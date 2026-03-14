import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2



def collate_landmark_batch(batch):
    """
    batch = list of (coords_seq, gt_xy5, mel, imgs4)
    coords_seq: (4,N,2)
    gt_xy5:     (N,2)
    mel:        (1,H,W)
    imgs4:      list of 4 numpy RGB images
    """
    coords_seq_list = []
    gt_xy5_list = []
    mel_list = []
    imgs4_list = []

    for coords_seq, gt_xy5, mel, imgs4 in batch:
        coords_seq_list.append(coords_seq)
        gt_xy5_list.append(gt_xy5)
        mel_list.append(mel)
        imgs4_list.append(imgs4)  # keep lists intact!

    coords_seq_t = torch.stack(coords_seq_list, dim=0)  # (B,4,N,2)
    gt_xy5_t     = torch.stack(gt_xy5_list, dim=0)      # (B,N,2)
    mel_t        = torch.stack(mel_list, dim=0)         # (B,1,H,W)

    # imgs4_list stays Python-native: List[B][4][H][W][3]
    return coords_seq_t, gt_xy5_t, mel_t, imgs4_list
    
    
    
    

def collate_landmark_batch_padmel(batch, pad_value: float = 0.0):
    """
    batch = list of (coords_seq, gt_xy5, mel, imgs4)
      coords_seq: (4,N,2)
      gt_xy5:     (N,2)
      mel:        (1,H,W)  # H,W may differ across items
      imgs4:      list of 4 numpy RGB images
    Returns:
      coords_seq_t: (B,4,N,2)
      gt_xy5_t:     (B,N,2)
      mel_t:        (B,1,Hmax,Wmax)  # padded
      imgs4_list:   list of lists (B × 4) of images (unchanged)
    """
    coords_seq_list, gt_xy5_list, mel_list, imgs4_list = [], [], [], []
    for coords_seq, gt_xy5, mel, imgs4 in batch:
        coords_seq_list.append(coords_seq)
        gt_xy5_list.append(gt_xy5)
        mel_list.append(mel)
        imgs4_list.append(imgs4)

    coords_seq_t = torch.stack(coords_seq_list, dim=0)  # (B,4,N,2)
    gt_xy5_t     = torch.stack(gt_xy5_list, dim=0)      # (B,N,2)

    # ----- pad mel both dims -----
    Hs = [m.shape[-2] for m in mel_list]
    Ws = [m.shape[-1] for m in mel_list]
    Hmax, Wmax = max(Hs), max(Ws)

    padded = []
    for m in mel_list:
        _, H, W = m.shape
        if H == Hmax and W == Wmax:
            padded.append(m)
        else:
            out = torch.full((1, Hmax, Wmax), float(pad_value), dtype=m.dtype)
            out[:, :H, :W] = m  # top-left align
            padded.append(out)
    mel_t = torch.stack(padded, dim=0)  # (B,1,Hmax,Wmax)

    return coords_seq_t, gt_xy5_t, mel_t, imgs4_list






class LipSyncLandmarkDataset(Dataset):
    """
    Uses precomputed .npz landmark files stored next to the PNGs.
    Produces:
        coords_seq : (4, N, 2) from frames 1..4
        gt_xy5     : (N, 2)    from frame 5
        mel        : (1, Hm, Wm)
        imgs4      : list of 4 RGB uint8 images (optional, for visualization)
    """

    def __init__(self,
                 frames_dir: str,
                 mels_dir: str,
                 sequence_length: int = 5,
                 landmark_dim: int = 2,
                 mel_width_target: int = 86,
                 return_images4: bool = True):
        super().__init__()
        self.frames_dir = frames_dir
        self.mels_dir = mels_dir
        self.sequence_length = sequence_length
        self.landmark_dim = landmark_dim
        self.return_images4 = return_images4
        self.mel_width_target = mel_width_target

        # Collect available frame stems
        files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        stems = [os.path.splitext(f)[0] for f in files]
        self.stems = stems

        # Build list of valid sequences
        self.samples = self._build_sequences()

    def _build_sequences(self):
        samples = []
        for s in self.stems:
            # Example: vid_004_093 → prefix="vid_004_", idx="093"
            parts = s.split("_")
            if len(parts) != 3:
                continue
            vid = parts[1]
            idx = int(parts[2])

        # Now build sequences per video
        by_vid = {}
        for s in self.stems:
            parts = s.split("_")
            if len(parts) != 3:
                continue
            vid = parts[1]
            idx = int(parts[2])
            by_vid.setdefault(vid, {})[idx] = s

        samples = []
        L = self.sequence_length # typically 5
        for vid, idx2stem in by_vid.items():
            indices = sorted(idx2stem.keys())
            for end_idx in indices:
                start_idx = end_idx - (L - 1)
                seq = list(range(start_idx, end_idx + 1))
                if all(i in idx2stem for i in seq):
                    samples.append([ vid, seq, [idx2stem[i] for i in seq] ])
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_mel(self, stem):
        path = os.path.join(self.mels_dir, stem + ".tiff")
        mel = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED), dtype=np.float32)
        if mel.ndim == 3:
            mel = mel[..., 0]

        # Normalize width to target (crop)
        if mel.shape[1] > self.mel_width_target:
            mel = mel[:, :self.mel_width_target]

        return mel[None, :, :]  # (1, H, W)

    def __getitem__(self, index):
        vid, idx_list, stem_list = self.samples[index]

        coords_all = []
        imgs4 = []

        # Load 5 frames: 1..5
        valid = True
        for si, stem in enumerate(stem_list):
            npz_path = os.path.join(self.frames_dir, stem + ".npz")
            if not os.path.isfile(npz_path):
                valid = False
                break

            data = np.load(npz_path)
            if not data["valid"]:
                valid = False
                break

            coords = data["coords"]  # (N,2)
            if coords.ndim != 2:
                valid = False
                break

            coords_all.append(coords)

            if self.return_images4 and si < 4:
                img_path = os.path.join(self.frames_dir, stem + ".png")

                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARN] Could not load image: {img_path}")
                    # Skip this sample → force dataset to return another good one
                    return self.__getitem__((index + 1) % len(self))
                
                # Convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs4.append(img)

                
                

        if not valid:
            # Recursive retry
            return self.__getitem__((index + 1) % len(self))

        coords_all = np.stack(coords_all, axis=0)  # (5, N, 2)

        coords_seq = coords_all[:4, :, :]  # frames 1..4
        gt_xy5     = coords_all[4, :, :]   # frame 5

        mel = self._load_mel(stem_list[-1])  # (1,H,W)

        coords_seq_t = torch.from_numpy(coords_seq).float()
        gt_xy5_t     = torch.from_numpy(gt_xy5).float()
        mel_t        = torch.from_numpy(mel).float()

        return coords_seq_t, gt_xy5_t, mel_t, imgs4
