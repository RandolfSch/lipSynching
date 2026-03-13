import torch
import numpy as np
import os


def to_uint8_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """img_tensor: (3,H,W) float [0,1]"""
    arr = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return arr



def collect_canonical_template_from_loader(
    loader,
    max_samples: int = 200
) -> np.ndarray:
    """
    Build a canonical template (N,2) by averaging the selected landmarks from the loader.
    Uses the 4th input frame (index 3) to reduce lip motion, just like before.

    loader yields: coords_seq_t, gt_xy5_t, mel_t, imgs4
    - coords_seq_t: (B, 4, N, 2)  frames 1..4 (normalized [0,1])
    """
    acc = []
    collected = 0

    for coords_seq_t, gt_xy5_t, mel_t, imgs4 in loader:
        # coords_seq_t: (B, 4, N, 2)
        coords_seq = coords_seq_t.detach().cpu().numpy()
        # take frame index 3 (4th image) per sample: shape (B, N, 2)
        coords_t4 = coords_seq[:, 3, :, :]
        acc.append(coords_t4)
        collected += coords_t4.shape[0]
        if collected >= max_samples:
            break

    if len(acc) == 0:
        raise RuntimeError("Could not build canonical template: got zero samples from loader.")

    all_coords = np.concatenate(acc, axis=0)  # (collected, N, 2)
    template = all_coords.mean(axis=0)        # (N, 2)
    return template



def save_checkpoint(model, optimizer, step, loss, A_w, out_dir):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "loss": float(loss),
        "A_w": A_w.cpu().numpy(),   # in case you rebuild graph later
    }
    fname = os.path.join(out_dir, f"checkpoint_step_{step:06d}.pth")
    torch.save(ckpt, fname)
    print(f"[checkpoint] saved {fname}")




def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]
    loss = ckpt["loss"]
    A_w = torch.from_numpy(ckpt["A_w"])
    print(f"[checkpoint] loaded {path} (step={step}, loss={loss})")
    return step, loss, A_w
    
    
    
    
    
    
    
    
    
    
def save_checkpoint2nd(model, optimizer, step, loss, out_dir):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "loss": float(loss)
    }
    fname = os.path.join(out_dir, f"checkpoint_step_{step:06d}.pth")
    torch.save(ckpt, fname)
    print(f"[checkpoint] saved {fname}")




def load_checkpoint2nd(model, optimizer, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]
    loss = ckpt["loss"]
    print(f"[checkpoint] loaded {path} (step={step}, loss={loss})")
    return step, loss
    
    
# ------------------------ Visualization helpers --------------------------------

def draw_points_and_intra_edges(img_rgb: np.ndarray,
                                coords_norm: np.ndarray,
                                color_points=(0, 255, 0),
                                thickness=1) -> np.ndarray:
    """
    Draws selected anchor points + intra-class edges (lips/eyes/nose) on the image.
    coords_norm: (N,2) normalized in [0,1]
    """
    h, w = img_rgb.shape[:2]
    out = img_rgb.copy()
    # Draw intra-class edges
    for setname in USE_SETS:
        S = SETNAME2SET[setname]
        for (i, j) in S:
            if i in RAWID2IDX and j in RAWID2IDX:
                a = RAWID2IDX[i]; b = RAWID2IDX[j]
                xa, ya = (coords_norm[a] * [w, h]).astype(int)
                xb, yb = (coords_norm[b] * [w, h]).astype(int)
                cv2.line(out, (xa, ya), (xb, yb), (255, 0, 255), thickness)  # magenta lines
    # Draw points
    for k in range(coords_norm.shape[0]):
        x, y = (coords_norm[k] * [w, h]).astype(int)
        cv2.circle(out, (x, y), 2, color_points, -1, lineType=cv2.LINE_AA)
    return out



def make_black_image(size=(512, 512)) -> np.ndarray:
    return np.zeros((size[1], size[0], 3), dtype=np.uint8)




def save_viz_collage(out_path: str,
                     imgs4: list,
                     coords4: np.ndarray,
                     pred5: np.ndarray,
                     tile_size=(512, 512)):
    """
    imgs4 : list of 4 RGB uint8 images
    coords4 : (4, N, 2) normalized coords
    pred5 : (N, 2) normalized coords
    """

    assert isinstance(imgs4, list), "imgs4 must be a list of images"
    assert len(imgs4) == 4, f"imgs4 must contain 4 images, got {len(imgs4)}"
    assert coords4.shape[0] == 4, f"coords4 must be shape (4,N,2), got {coords4.shape}"
    assert pred5.ndim == 2 and pred5.shape[1] == 2, "pred5 must be (N,2)"

    tiles = []

    # ---- frames 1..4 with ground truth landmarks ----
    for t in range(4):
        img = imgs4[t]
        assert isinstance(img, np.ndarray), "imgs4[t] must be a numpy array"

        # Resize frame for collage
        img_resized = cv2.resize(img, tile_size[::-1], interpolation=cv2.INTER_AREA)

        # Draw GT anchor points + intra-class edges
        annotated = draw_points_and_intra_edges(
            img_resized,
            coords4[t],  # <-- (N,2)
            thickness=1
        )
        tiles.append(annotated)

    # ---- frame 5 = black + predicted points ----
    black = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
    h, w = black.shape[:2]

    pred_img = black.copy()
    for k in range(pred5.shape[0]):
        x, y = (pred5[k] * [w, h]).astype(int)
        cv2.circle(pred_img, (x, y), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    tiles.append(pred_img)

    # ---- stack horizontally ----
    collage = np.concatenate(tiles, axis=1)

    # save as BGR
    cv2.imwrite(out_path, cv2.cvtColor(collage, cv2.COLOR_RGB2BGR))