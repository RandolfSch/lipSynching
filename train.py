
# train_graph_lipsync.py
import os
import time
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import mediapipe as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from settings import RAW_IDS, RAWID2CLS, RAWID2IDX, N_NODES, mp_drawing, mp_connections, mp_face_mesh_module, USE_SETS, SETNAME2SET

# --- Your data module (change 'dl' to your filename) ---
from dataloader import LipSyncDataset, CollatePadMel  # or lipsync_collate_pad_mel
from LandmarkLoader import LipSyncLandmarkDataset, collate_landmark_batch_padmel


from Utils import to_uint8_rgb, collect_canonical_template_from_loader, save_checkpoint, load_checkpoint
from Utils import draw_points_and_intra_edges, make_black_image, save_viz_collage




# ------------------------ Silence noisy logs (optional) ------------------------
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass




def run_facemesh_landmarks(image_rgb: np.ndarray,
                           mp_face_mesh) -> Optional[np.ndarray]:
    """
    Runs FaceMesh on an image. Returns landmarks (468,2) normalized to [0,1], or None if no face.
    """
    h, w = image_rgb.shape[:2]
    res = mp_face_mesh.process(image_rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0]
    pts = np.array([(p.x, p.y) for p in lm.landmark], dtype=np.float32)  # normalized coords
    return pts  # shape (468, 2), in [0,1]

def extract_selected(landmarks_468_norm: np.ndarray,
                     raw_ids: List[int]) -> np.ndarray:
    """
    From full (468,2) normalized coords, gather selected nodes in order.
    Returns (N,2).
    """
    return landmarks_468_norm[raw_ids, :]

# ------------------------ Weighted adjacency (intra + inter) -------------------

def build_weighted_adjacency_from_template(
    template_xy: np.ndarray,              # (N,2) canonical coords for selected nodes (in [0,1])
    raw_ids: List[int],
    rawid2cls: Dict[int, int],
    k_inter: int = 3,
    w_intra: float = 1.0,
    w_inter: float = 0.3,
) -> np.ndarray:
    """
    Build weighted adjacency:
      - Intra-class edges from MediaPipe connection sets (weight = w_intra).
      - Inter-class edges: for each node, connect to k nearest nodes in other classes
        using distances in 'template_xy' (weight = w_inter).
    Returns A_w: (N, N) float32.
    """
    N = len(raw_ids)
    A = np.zeros((N, N), dtype=np.float32)

    # Intra-class from MP connections
    for setname in USE_SETS:
        S = SETNAME2SET[setname]
        for (i, j) in S:
            if i in RAWID2IDX and j in RAWID2IDX:
                a = RAWID2IDX[i]; b = RAWID2IDX[j]
                A[a, b] = max(A[a, b], w_intra)
                A[b, a] = max(A[b, a], w_intra)

    # Inter-class kNN from template
    # Precompute pairwise distances
    P = template_xy  # (N,2)
    diffs = P[:, None, :] - P[None, :, :]       # (N, N, 2)
    dists = np.linalg.norm(diffs, axis=-1)      # (N, N)

    cls_arr = np.array([rawid2cls[rid] for rid in raw_ids], dtype=np.int32)
    for i in range(N):
        mask_other = (cls_arr != cls_arr[i])
        dists_i = dists[i].copy()
        dists_i[~mask_other] = np.inf  # keep only other classes
        # Get k smallest indices
        js = np.argpartition(dists_i, k_inter)[:k_inter]
        for j in js:
            if math.isfinite(dists_i[j]):
                A[i, j] = max(A[i, j], w_inter)
                A[j, i] = max(A[j, i], w_inter)

    return A

# ------------------------ Model (audio-conditioned temporal GNN) ---------------

class AudioEncoderCNN(nn.Module):
    def __init__(self, d_audio: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(128, d_audio)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.net(mel)       # (B, 128, 1, 1)
        x = x.flatten(1)        # (B, 128)
        x = self.proj(x)        # (B, d_audio)
        return x

class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias=True):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.nei_lin  = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, A_w: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, N, Din)
        A_w: (N, N) weighted adjacency (float)
        """
        B, N, _ = x.shape
        device = x.device
        A = A_w.to(device).float()
        I = torch.eye(N, device=device)
        A_hat = A + I
        deg = A_hat.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # (N,1)
        nei_agg = torch.matmul(A_hat, x) / deg                  # (B, N, Din)
        return self.self_lin(x) + self.nei_lin(nei_agg)

class TemporalGraphPredictor(nn.Module):
    def __init__(self, num_nodes: int, d_hidden: int = 128, d_audio: int = 128):
        super().__init__()
        self.N = num_nodes
        self.d_hidden = d_hidden
        self.audio_enc = AudioEncoderCNN(d_audio=d_audio)
        self.in_lin = nn.Linear(4, d_hidden)               # [x,y,dx,dy]
        self.cond_lin = nn.Linear(d_hidden + d_audio, d_hidden)
        self.gnn1 = GraphConv(d_hidden, d_hidden)
        self.gnn2 = GraphConv(d_hidden, d_hidden)
        self.gru = nn.GRUCell(d_hidden, d_hidden)
        self.out_mlp = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, 2)
        )

    def forward(self, coords_seq: torch.Tensor, mel: torch.Tensor, A_w: torch.Tensor) -> torch.Tensor:
        """
        coords_seq: (B, T, N, 2)  (frames 1..T, T=4)
        mel:        (B, 1, Hm, Wm)
        A_w:        (N, N)
        Returns:
            pred_xy_5: (B, N, 2)
        """
        B, T, N, _ = coords_seq.shape
        assert N == self.N, f"N mismatch: coords_seq has {N}, model expects {self.N}"
        device = coords_seq.device

        a = self.audio_enc(mel)  # (B, d_audio)

        h = torch.zeros(B * N, self.d_hidden, device=device)
        prev = None
        for t in range(T):
            xt = coords_seq[:, t, :, :]                 # (B, N, 2)
            dxt = torch.zeros_like(xt) if t == 0 else (xt - prev)
            prev = xt
            feat = torch.cat([xt, dxt], dim=-1)         # (B, N, 4)
            feat = self.in_lin(feat)                    # (B, N, H)
            a_expand = a.unsqueeze(1).expand(-1, N, -1) # (B, N, d_audio)
            feat = self.cond_lin(torch.cat([feat, a_expand], dim=-1))
            feat = F.relu(feat, inplace=True)

            msg = F.relu(self.gnn1(feat, A_w), inplace=True)
            msg = F.relu(self.gnn2(msg, A_w), inplace=True)

            msg_flat = msg.reshape(B * N, -1)
            h = self.gru(msg_flat, h)

        h_nodes = h.view(B, N, -1)
        delta = self.out_mlp(h_nodes)                   # (B, N, 2)
        xT = coords_seq[:, -1, :, :]
        pred_xy_5 = xT + delta
        return pred_xy_5












# ------------------------ Training / main --------------------------------------



def main():
    # ----------------- Paths & hyper-params -----------------
    frames_dir = r"D:/Training Data/CarmenFrames/"
    mels_dir   = r"D:/Training Data/CarmenFrames/"
    out_dir    = r"D:/Training Data/CarmenFrames_out"
    os.makedirs(out_dir, exist_ok=True)

    batch_size = 8           # keep small while running MediaPipe in-loop
    sequence_length = 5
    lr = 2e-4
    weight_decay = 1e-5
    max_steps = 2000
    viz_interval = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




    # auto-resume if checkpoint exists
    latest = sorted([f for f in os.listdir(out_dir) if f.startswith("checkpoint_step")])
    if latest:
        last_path = os.path.join(out_dir, latest[-1])
        step, last_loss, A_w = load_checkpoint(model, opt, last_path)
        print(f"Resuming from step {step} (loss={last_loss})")
    else:
        step = 0




    # ----------------- Dataset & loader ---------------------
    # dataset = LipSyncDataset(
    #     frames_dir=frames_dir,
    #     mels_dir=mels_dir,
    #     sequence_length=sequence_length,
    #     enforce_consecutive=True,
    #     return_paths=True,
    #     recursive=True,
    #     # Leave transforms default (ToTensor) per your dataset class
    # )
    
    dataset = LipSyncLandmarkDataset(
        frames_dir="D:/Training Data/CarmenFrames/",
        mels_dir="D:/Training Data/CarmenFrames/",
        sequence_length=5,
        return_images4=True
    )


    
    # IMPORTANT on Windows + MediaPipe: keep workers=0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=collate_landmark_batch_padmel,     # <-- FIX
    )


    

    # ----------------- MediaPipe init -----------------------
    # mp_face_mesh = mp_face_mesh_module.FaceMesh(
    #     static_image_mode=True,
    #     max_num_faces=1,
    #     refine_landmarks=True,
    #     min_detection_confidence=0.5
    # )

    # ----------------- Build canonical template & adjacency -
    print("Building canonical template for inter-class edges (from precomputed coords) ...")
    template_xy = collect_canonical_template_from_loader(loader, max_samples=200)  # (N,2)
    
    A_w_np = build_weighted_adjacency_from_template(
        template_xy, RAW_IDS, RAWID2CLS, k_inter=3, w_intra=1.0, w_inter=0.3
    )
    A_w = torch.from_numpy(A_w_np)

    
    
    

    # ----------------- Model, loss, optimizer ----------------
    model = TemporalGraphPredictor(num_nodes=N_NODES, d_hidden=128, d_audio=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.SmoothL1Loss(beta=0.01)  # Huber

    # ----------------- Training loop ------------------------
    step = 0
    t0 = time.time()
    
    
    
    checkpoint_interval = 500   # or whatever you prefer

    while step < max_steps:
        for coords_seq_t, gt_xy5_t, mel_t, imgs4 in loader:
    
            model.train()
    
            coords_seq_t = coords_seq_t.to(device)   # (B,4,N,2)
            gt_xy5_t     = gt_xy5_t.to(device)       # (B,N,2)
            mel_t        = mel_t.to(device)          # (B,1,Hm,Wm)
            A_w_t        = A_w.to(device)            # (N,N)
    
            # --- forward ---
            pred_xy5 = model(coords_seq_t, mel_t, A_w_t)
    
            # --- loss ---
            loss = crit(pred_xy5, gt_xy5_t)
    
            # --- optimize ---
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    
            # --- logging ---
            if step % 10 == 0:
                dt = time.time() - t0
                print(f"step {step:05d} | loss {loss.item():.6f} | {dt:.1f}s")
    



            # --- visualization ---
            if step % viz_interval == 0:
                with torch.no_grad():
                    v0_imgs4   = imgs4[0]                                   # list of 4 images
                    v0_coords4 = coords_seq_t[0].detach().cpu().numpy()     # (4, N, 2)
                    v0_pred5   = pred_xy5[0].detach().cpu().numpy()         # (N, 2)
            
                out_path = os.path.join(out_dir, f"viz_step_{step:06d}.png")
                save_viz_collage(out_path, v0_imgs4, v0_coords4, v0_pred5, tile_size=(512, 512))
                print(f"[viz] saved {out_path}")


            # --- checkpoint saving ---
            if step % checkpoint_interval == 0 and step > 0:
                save_checkpoint(model, opt, step, loss.item(), A_w, out_dir)
                
                
    
            step += 1
            if step >= max_steps:
                break
    
    print("Training finished.")


if __name__ == "__main__":
    main()
