
# graph_temporal_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEncoderCNN(nn.Module):
    """
    Tiny CNN for mel-spectrogram -> embedding vector.
    Input: (B, 1, Hm, Wm)
    Output: (B, d_audio)
    """
    def __init__(self, d_audio: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2, W/2

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4, W/4

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # (B, 128, 1, 1)
        )
        self.proj = nn.Linear(128, d_audio)

    def forward(self, mel):
        x = self.net(mel)           # (B, 128, 1, 1)
        x = x.flatten(1)            # (B, 128)
        x = self.proj(x)            # (B, d_audio)
        return x




class GraphConv(nn.Module):
    """
    Simple GraphConv: h_i' = W_self h_i + W_nei * mean_{j in N(i)} h_j
    A: (N, N) adjacency (0/1), no self-loops required; we add them internally.
    Works on batched tensors: (B, N, D).
    """
    def __init__(self, in_dim: int, out_dim: int, bias=True):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.nei_lin  = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, A):
        """
        x: (B, N, Din)
        A: (N, N) float/bool adjacency (same for all in batch)
        returns: (B, N, Dout)
        """
        B, N, D = x.shape
        device = x.device

        # Ensure float adjacency on the correct device
        if A.device != device:
            A = A.to(device)
        A = A.float()

        # Add self-loops (without double-counting later)
        I = torch.eye(A.size(0), device=device)
        A_hat = A + I

        # Degree-normalized neighbor average
        deg = A_hat.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (N, 1)
        nei_agg = torch.matmul(A_hat, x) / deg  # (B, N, Din)

        out = self.self_lin(x) + self.nei_lin(nei_agg)
        return out



class TemporalGraphPredictor(nn.Module):
    """
    Audio-conditioned temporal GNN that predicts anchor positions for frame 5.

    Inputs:
      - coords_seq: (B, T, N, 2) normalized coords for frames 1..T (T=4)
      - mel:        (B, 1, Hm, Wm)
      - A:          (N, N) adjacency (fixed graph)

    Output:
      - pred_xy_5:  (B, N, 2) predicted coords for frame 5
    """
    def __init__(self, num_nodes: int, d_hidden: int = 128, d_audio: int = 128):
        super().__init__()
        self.N = num_nodes
        self.d_hidden = d_hidden

        self.audio_enc = AudioEncoderCNN(d_audio=d_audio)

        # Input projector: [x_t, y_t, dx_t, dy_t] -> d_hidden
        self.in_lin = nn.Linear(4, d_hidden)

        # Condition (concat audio embedding per node)
        self.cond_lin = nn.Linear(d_hidden + d_audio, d_hidden)

        # One or more GNN layers; keep it simple with 2
        self.gnn1 = GraphConv(d_hidden, d_hidden)
        self.gnn2 = GraphConv(d_hidden, d_hidden)

        # Temporal accumulator (per node)
        self.gru = nn.GRUCell(d_hidden, d_hidden)

        # Output head predicts delta from frame T to frame T+1
        self.out_mlp = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, 2)
        )

    def forward(self, coords_seq, mel, A):
        """
        coords_seq: (B, T, N, 2)  -- frames 1..T (T=4)
        mel:        (B, 1, Hm, Wm)
        A:          (N, N)
        """
        B, T, N, _ = coords_seq.shape
        assert N == self.N, f"N mismatch: coords_seq has {N}, model expects {self.N}"

        # Audio embedding
        a = self.audio_enc(mel)  # (B, d_audio)

        # Prepare GRU hidden state (B*N, d_hidden)
        h = torch.zeros(B * N, self.d_hidden, device=coords_seq.device)

        # Time loop: t = 0..T-1 (corresponds to frames 1..T)
        prev = None
        for t in range(T):
            xt = coords_seq[:, t, :, :]  # (B, N, 2)
            if t == 0:
                dxt = torch.zeros_like(xt)
            else:
                dxt = xt - prev  # (B, N, 2)
            prev = xt

            # Feature: [x, y, dx, dy] -> d_hidden
            feat = torch.cat([xt, dxt], dim=-1)        # (B, N, 4)
            feat = self.in_lin(feat)                   # (B, N, d_hidden)

            # Audio conditioning (concat and project back to d_hidden)
            a_expand = a.unsqueeze(1).expand(-1, N, -1)  # (B, N, d_audio)
            feat = torch.cat([feat, a_expand], dim=-1)   # (B, N, d_hidden + d_audio)
            feat = self.cond_lin(feat)                   # (B, N, d_hidden)
            feat = F.relu(feat, inplace=True)

            # Graph message passing
            msg = self.gnn1(feat, A)
            msg = F.relu(msg, inplace=True)
            msg = self.gnn2(msg, A)
            msg = F.relu(msg, inplace=True)

            # GRU over time (per node)
            msg_flat = msg.reshape(B * N, -1)
            h = self.gru(msg_flat, h)  # (B*N, d_hidden)

        # Predict delta for next frame
        h_nodes = h.view(B, N, -1)
        delta = self.out_mlp(h_nodes)     # (B, N, 2)
        xT = coords_seq[:, -1, :, :]      # last input frame coords (frame 4)
        pred_xy_5 = xT + delta            # predict frame 5
        return pred_xy_5
