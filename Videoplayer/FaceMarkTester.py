
import os
import cv2
import numpy as np
import mediapipe as mp

# -------- Input --------
IMG_PATH = r"D:/Training Data/CarmenFrames/frame_000100.png"
NPZ_PATH = os.path.splitext(IMG_PATH)[0] + ".npz"

# Colors (BGR) for the four sets
COLOR_MAP = {
    "LIPS":       (255, 255, 0),   # cyan
    "LEFT_EYE":   (255, 0, 255),   # magenta
    "RIGHT_EYE":  (0, 255, 255),   # yellow
    "NOSE":       (0, 255, 0),     # lime
}
POINT_RADIUS = 2
THICKNESS = -1
ALPHA = 0.85

# --- Rebuild exactly the same RAW_IDS and set membership as in your extractor ---
mp_connections = mp.solutions.face_mesh_connections
USE_SETS = ("LIPS", "LEFT_EYE", "RIGHT_EYE", "NOSE")
SETNAME2SET = {
    "LIPS": mp_connections.FACEMESH_LIPS,
    "LEFT_EYE": mp_connections.FACEMESH_LEFT_EYE,
    "RIGHT_EYE": mp_connections.FACEMESH_RIGHT_EYE,
    "NOSE": mp_connections.FACEMESH_NOSE,
}

# Build RAW_IDS = sorted(unique ids from all requested sets)
_raw_ids = set()
for S in USE_SETS:
    for (i, j) in SETNAME2SET[S]:
        _raw_ids.add(i); _raw_ids.add(j)
RAW_IDS = np.array(sorted(_raw_ids), dtype=np.int32)  # (N,)

# Per-set global ids
IDS_PER_SET = {
    name: np.array(sorted({i for (i, j) in SETNAME2SET[name]} |
                          {j for (i, j) in SETNAME2SET[name]}), dtype=np.int32)
    for name in USE_SETS
}

# Map global id -> row index in coords (coords rows follow RAW_IDS order)
RID2ROW = {rid: k for k, rid in enumerate(RAW_IDS)}

def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(IMG_PATH)

    data = np.load(NPZ_PATH)
    if not bool(data["valid"]):
        print("Landmarks not valid for this frame.")
        cv2.imshow("Overlay", img); cv2.waitKey(0); cv2.destroyAllWindows()
        return

    coords = data["coords"]  # (N,2) normalized to the image
    H, W = img.shape[:2]
    pts_px = (coords * np.array([W, H], dtype=np.float32)).astype(int)

    overlay = img.copy()
    # Draw each set in its color by mapping set global ids -> row indices in coords
    for name in USE_SETS:
        color = COLOR_MAP[name]
        ids = IDS_PER_SET[name]
        rows = [RID2ROW[rid] for rid in ids if rid in RID2ROW]
        for r in rows:
            x, y = int(pts_px[r, 0]), int(pts_px[r, 1])
            cv2.circle(overlay, (x, y), POINT_RADIUS, color, THICKNESS)

    blended = cv2.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0)

    # Legend
    y0 = 24
    for name in USE_SETS:
        cv2.putText(blended, name, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MAP[name], 2, cv2.LINE_AA)
        y0 += 22

    cv2.imshow("LMK overlay (per-set colors)", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
