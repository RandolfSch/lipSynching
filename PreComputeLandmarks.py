import os
import glob
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

# ----------------------
# MediaPipe Setup
# ----------------------
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

mp_connections = mp.solutions.face_mesh_connections
USE_SETS = ("LIPS", "LEFT_EYE", "RIGHT_EYE", "NOSE")
SETNAME2SET = {
    "LIPS": mp_connections.FACEMESH_LIPS,
    "LEFT_EYE": mp_connections.FACEMESH_LEFT_EYE,
    "RIGHT_EYE": mp_connections.FACEMESH_RIGHT_EYE,
    "NOSE": mp_connections.FACEMESH_NOSE,
}

# Build selected raw ids
raw_ids = set()
for S in USE_SETS:
    for (i, j) in SETNAME2SET[S]:
        raw_ids.add(i); raw_ids.add(j)
RAW_IDS = sorted(raw_ids)

def run_facemesh(image_rgb):
    res = mp_face_mesh.process(image_rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0]
    pts = np.array([(p.x, p.y) for p in lm.landmark], dtype=np.float32)
    return pts  # (468,2) normalized [0,1]

def extract_selected(full468):
    return full468[RAW_IDS, :]  # (N,2)

# ----------------------
# MAIN
# ----------------------

def precompute_landmarks(frames_dir):
    png_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    print(f"Found {len(png_files)} PNG frames.")

    for path in tqdm(png_files):
        stem = os.path.splitext(path)[0]
        out_npz = stem + ".npz"

        # Skip if exists
        if os.path.isfile(out_npz):
            continue

        img = cv2.imread(path)
        if img is None:
            print(f"Warning: cannot read {path}")
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lm_468 = run_facemesh(rgb)

        if lm_468 is None:
            np.savez_compressed(out_npz, valid=False)
            continue

        sel = extract_selected(lm_468)  # (N,2)
        np.savez_compressed(out_npz, valid=True, coords=sel)

    print("Precomputation complete.")

if __name__ == "__main__":
    frames_dir = r"D:/Training Data/BobFrames/"
    precompute_landmarks(frames_dir)
