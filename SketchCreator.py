import os
import argparse
import glob
import numpy as np
import cv2
from tqdm import tqdm

# We only import the connection constants; this does NOT run FaceMesh/TFLite
import mediapipe as mp
mp_connections = mp.solutions.face_mesh_connections

# ----------------------------
# Configuration (colors/RGB)
# ----------------------------
USE_SETS = ("LIPS", "LEFT_EYE", "RIGHT_EYE", "NOSE")
SETNAME2SET = {
    "LIPS": mp_connections.FACEMESH_LIPS,
    "LEFT_EYE": mp_connections.FACEMESH_LEFT_EYE,
    "RIGHT_EYE": mp_connections.FACEMESH_RIGHT_EYE,
    "NOSE": mp_connections.FACEMESH_NOSE,
}
# Colors in RGB (we’ll convert to BGR only when writing)
CLASS_COLORS = {
    "LIPS": (255, 0, 0),       # Red
    "LEFT_EYE": (0, 255, 0),   # Green
    "RIGHT_EYE": (0, 255, 0),  # Green (same as left; keeps 3-color scheme)
    "NOSE": (0, 0, 255),       # Blue
}


def build_selected_raw_ids_and_class_map():
    """
    Returns:
        RAW_IDS: sorted list of unique MP landmark ids across the selected sets.
        RAWID2CLS: dict raw_id -> class name (one of USE_SETS)
    """
    raw_ids = set()
    rawid2cls = {}
    for setname in USE_SETS:
        S = SETNAME2SET[setname]
        for (i, j) in S:
            raw_ids.add(i); raw_ids.add(j)
            if i not in rawid2cls:
                rawid2cls[i] = setname
            if j not in rawid2cls:
                rawid2cls[j] = setname
    raw_ids = sorted(raw_ids)
    return raw_ids, rawid2cls


RAW_IDS, RAWID2CLS = build_selected_raw_ids_and_class_map()
RAWID2IDX = {rid: k for k, rid in enumerate(RAW_IDS)}
N_NODES = len(RAW_IDS)


def draw_sketch(coords_norm: np.ndarray,
                H: int,
                W: int,
                thickness: int = 1,
                point_radius: int = 2) -> np.ndarray:
    """
    Draw a 3-channel colored sketch on a blank RGB canvas.

    Args:
        coords_norm: (N,2) normalized coordinates in [0,1] (order MUST match RAW_IDS).
        H, W: output canvas size (e.g., 512 x 512).
        thickness: line thickness.
        point_radius: landmark dot radius.

    Returns:
        RGB image (H, W, 3) uint8
    """
    assert coords_norm.ndim == 2 and coords_norm.shape[1] == 2, "coords must be (N,2)"
    if coords_norm.shape[0] != N_NODES:
        # If this happens, the .npz wasn't created with our selection; we'll still draw points in white.
        print(f"[WARN] coords length {coords_norm.shape[0]} != expected {N_NODES}. "
              "Edges will be skipped; drawing white points only.")
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for (x, y) in coords_norm:
            xi = int(np.clip(x, 0, 1) * (W - 1))
            yi = int(np.clip(y, 0, 1) * (H - 1))
            cv2.circle(img, (xi, yi), point_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        return img

    # Blank RGB
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Helper: get pixel coords by node index
    def idx_to_xy(idx: int):
        x, y = coords_norm[idx]
        xi = int(np.clip(x, 0, 1) * (W - 1))
        yi = int(np.clip(y, 0, 1) * (H - 1))
        return xi, yi

    # 1) Draw intra-class connections in their color
    for setname in USE_SETS:
        color = CLASS_COLORS[setname]
        for (i_raw, j_raw) in SETNAME2SET[setname]:
            if i_raw in RAWID2IDX and j_raw in RAWID2IDX:
                a = RAWID2IDX[i_raw]
                b = RAWID2IDX[j_raw]
                xa, ya = idx_to_xy(a)
                xb, yb = idx_to_xy(b)
                cv2.line(img, (xa, ya), (xb, yb), color, thickness, lineType=cv2.LINE_AA)

    # 2) Draw per-point colored dots (class-colored)
    for rid, idx in RAWID2IDX.items():
        setname = RAWID2CLS.get(rid, None)
        color = CLASS_COLORS.get(setname, (255, 255, 255))
        x, y = idx_to_xy(idx)
        cv2.circle(img, (x, y), point_radius, color, -1, lineType=cv2.LINE_AA)

    return img


def main():
    parser = argparse.ArgumentParser(description="Create 3-color sketches from precomputed landmark .npz files.")
    parser.add_argument("--in_dir",  type=str, required=True,
                        help="Input folder containing .npz landmark files.")
    parser.add_argument("--out_dir", type=str, default=r"D:\Training Data\BobSketches",
                        help="Output folder for sketch PNGs.")
    parser.add_argument("--width",   type=int, default=512, help="Output width.")
    parser.add_argument("--height",  type=int, default=512, help="Output height.")
    parser.add_argument("--thickness", type=int, default=1, help="Line thickness.")
    parser.add_argument("--point_radius", type=int, default=2, help="Landmark point radius.")
    parser.add_argument("--suffix",  type=str, default="",
                        help="Optional suffix for filename base before .png. "
                             "Default '' means exact replacement (.npz -> .png). "
                             "Use '_sketch' if you want 'name_sketch.png'.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(args.in_dir, "*.npz")))
    if not npz_files:
        print(f"[INFO] No .npz files found in {args.in_dir}")
        return

    print(f"[INFO] Found {len(npz_files)} .npz files. Writing sketches to: {args.out_dir}")

    for npz_path in tqdm(npz_files):
        base = os.path.basename(npz_path)
        stem = os.path.splitext(base)[0]

        # Output name: exact replacement or with suffix
        out_name = f"{stem}{args.suffix}.png" if args.suffix else f"{stem}.png"
        out_path = os.path.join(args.out_dir, out_name)

        try:
            data = np.load(npz_path)
            valid = bool(data.get("valid", True))
            if not valid or "coords" not in data:
                # create empty black sketch (or skip)
                # Here we skip and warn
                print(f"[WARN] Skipping invalid landmarks: {npz_path}")
                continue

            coords = data["coords"].astype(np.float32)  # (N,2) normalized
            img_rgb = draw_sketch(coords, H=args.height, W=args.width,
                                  thickness=args.thickness, point_radius=args.point_radius)
            # Save (convert to BGR for cv2)
            cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"[ERROR] Failed on {npz_path}: {e}")

    print("[DONE] Sketch generation complete.")


if __name__ == "__main__":
    main()



# python SketchCreator.py --in_dir "D:\Training Data\BobFrames" --out_dir "D:\Training Data\BobSketches" --width 256 --height 256 --thickness 2 --point_radius 3
# python SketchCreator.py --in_dir "D:/Training Data/CarmenFrames" --out_dir "D:/Training Data/Carmenketches" --width 256 --height 256 --thickness 2 --point_radius 3
