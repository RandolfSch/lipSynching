
import os
import cv2
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Utils import save_checkpoint2nd, load_checkpoint2nd


# ============================================================
# 1. Dataset
# ============================================================

class FramePredictDataset(Dataset):
    """
    Provides for each time step X:
        prev_small      = previous RGB frame (X-1) downscaled 1/4
        sketch_X        = 3-channel colored sketch at time X
        mel_X           = (1,H,W) mel spectrogram
        full_img_X      = ground truth full RGB frame (X)
    """
    def __init__(self,
                 frames_dir,
                 sketch_dir,
                 mel_dir,
                 downscale_factor=4,
                 mel_width_target=86):
        super().__init__()

        self.frames_dir = frames_dir
        self.sketch_dir = sketch_dir
        self.mel_dir    = mel_dir
        self.downscale_factor = downscale_factor
        self.mel_width_target = mel_width_target

        # All PNG frames
        files = sorted([f for f in os.listdir(frames_dir)
                        if f.lower().endswith(".png")])

        # Must skip index 0 because we need X-1
        self.frames = files[1:]

    def __len__(self):
        return len(self.frames)

    def _load_img(self, path):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Could not load {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def _load_mel(self, stem):
        mel_path = os.path.join(self.mel_dir, stem + ".tiff")
        mel = cv2.imread(mel_path, cv2.IMREAD_UNCHANGED)
        if mel is None:
            raise RuntimeError(f"Missing mel: {mel_path}")

        mel = mel.astype(np.float32)
        if mel.ndim == 3:
            mel = mel[..., 0]

        # Ensure consistent width
        if mel.shape[1] > self.mel_width_target:
            mel = mel[:, :self.mel_width_target]
        elif mel.shape[1] < self.mel_width_target:
            pad = self.mel_width_target - mel.shape[1]
            mel = np.pad(mel, ((0,0),(0,pad)), constant_values=0)

        mel = mel[None,:,:]     # (1,H,W)
        return mel


    def __getitem__(self, idx):
        fname_x = self.frames[idx]
        stem_x  = os.path.splitext(fname_x)[0]
    
        prev_fname = self.frames[idx - 1]
        prev_stem  = os.path.splitext(prev_fname)[0]
    
        # Load full image X and previous frame
        img_x     = self._load_img(os.path.join(self.frames_dir, fname_x))
        img_prev  = self._load_img(os.path.join(self.frames_dir, prev_fname))
    
        H, W = img_prev.shape[:2]
        small_prev = cv2.resize(
            img_prev,
            (W // self.downscale_factor, H // self.downscale_factor),
            interpolation=cv2.INTER_AREA
        )
    
        # -------- SAFE SKETCH LOADING + UPSCALE ------------------
        sketch_path = os.path.join(self.sketch_dir, stem_x + ".png")
    
        try:
            sketch = self._load_img(sketch_path)
        except Exception:
            # The sketch is missing or unreadable -> skip sample
            # "idx + 1" ensures the dataset moves forward
            return self.__getitem__((idx + 1) % len(self))
    
        # Now upscale 256x256 → 512x512
        sketch = cv2.resize(sketch, (512, 512), interpolation=cv2.INTER_LINEAR)
        # ----------------------------------------------------------
    
        # Load mel
        mel = self._load_mel(stem_x)
    
        # Convert to torch
        small_prev_t = torch.from_numpy(small_prev).permute(2,0,1)
        sketch_t     = torch.from_numpy(sketch).permute(2,0,1)
        mel_t        = torch.from_numpy(mel).float()
        img_x_t      = torch.from_numpy(img_x).permute(2,0,1)
    
        return small_prev_t, sketch_t, mel_t, img_x_t


        
        
        


# ============================================================
# 2. Model: UNet with Audio Conditioning
# ============================================================

class MelEncoder(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(128, d)

    def forward(self, mel):
        x = self.net(mel).flatten(1)
        return self.proj(x)   # (B,d)


class UNetFramePredictor(nn.Module):
    def __init__(self, mel_dim=128):
        super().__init__()
        self.mel_enc = MelEncoder(mel_dim)

        # Input channels:
        #   3 = previous frame (upsampled)
        #   3 = colored sketch
        #   mel_dim = tiled mel embedding
        self.in_ch = 3 + 3 + mel_dim

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.in_ch, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )

        # Decoder
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )

        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, prev_small, sketch, mel):
        B = prev_small.size(0)

        # Upscale prev frame to full size
        prev_up = F.interpolate(prev_small, size=(512,512),
                                mode="bilinear", align_corners=False)

        # Mel conditioning map
        mel_vec = self.mel_enc(mel)   # (B,mel_dim)
        mel_map = mel_vec[:,:,None,None].repeat(1, 1, 512, 512)

        x = torch.cat([prev_up, sketch, mel_map], dim=1)

        # UNet
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(d2)

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(d1)

        out = torch.sigmoid(self.final(d1))  # (B,3,512,512)
        return out


# ============================================================
# 3. Training loop
# ============================================================

def train():
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    
    out_dir = "C:/Temp/Crafter/New BOB/FramePredictor_out/"
    os.makedirs(out_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetFramePredictor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    # auto-resume if checkpoint exists
    latest = sorted([f for f in os.listdir(out_dir) if f.startswith("checkpoint_step")])
    if latest:
        last_path = os.path.join(out_dir, latest[-1])
        step, last_loss = load_checkpoint2nd(model, opt, last_path)
        print(f"Resuming from step {step} (loss={last_loss})")
    else:
        step = 0
        
    frames_dir = "C:/Temp/Crafter/New BOB/BobFrames/"
    sketch_dir = "C:/Temp/Crafter/New BOB/BobSketches/"   # <--- 3-channel sketches here
    mel_dir    = "C:/Temp/Crafter/New BOB/BobFrames/"

    dataset = FramePredictDataset(frames_dir, sketch_dir, mel_dir)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    

    # model = UNetFramePredictor().to(device)
    # opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    crit = nn.L1Loss()

    step = 0
    checkpoint_interval = 2000   # or whatever you prefer
    
    for epoch in range(10):
        for prev_small, sketch, mel, img_x in loader:
            prev_small = prev_small.to(device)
            sketch     = sketch.to(device)
            mel        = mel.to(device)
            img_x      = img_x.to(device)

            pred = model(prev_small, sketch, mel)
            loss = crit(pred, img_x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % 500 == 0:
                print(f"step {step} | loss {loss.item():.5f}")
                out = pred[0].detach().cpu().permute(1,2,0).numpy()
                out = (out * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(out_dir, f"pred_{step:06d}.png"),
                    cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                )
                
            # --- checkpoint saving ---
            if step % checkpoint_interval == 0 and step > 0:
                save_checkpoint2nd(model, opt, step, loss.item(), out_dir)

            step += 1


if __name__ == "__main__":
    train()
