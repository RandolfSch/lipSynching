from torch.utils.data import DataLoader
import torchvision.transforms as T
from dl import LipSyncDataset, CollatePadMel  # ensure these are top-level in dl.py

import cv2
import os
import numpy as np


import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh_module = mp.solutions.face_mesh
mp_connections = mp.solutions.face_mesh_connections

FACEMESH_LIPS = mp_connections.FACEMESH_LIPS
FACEMESH_LEFT_EYE = mp_connections.FACEMESH_LEFT_EYE
FACEMESH_RIGHT_EYE = mp_connections.FACEMESH_RIGHT_EYE
FACEMESH_NOSE = mp_connections.FACEMESH_NOSE


frames_dir = "D:/Training Data/BobFrames/"
mels_dir   = "D:/Training Data/BobFrames/"
out_dir = "D:/Training Data/BobFrames_Out/"


os.makedirs(out_dir, exist_ok=True)

# Prepare FaceMesh once (static_image_mode=True for still frames)
mp_face_mesh = mp_face_mesh_module.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)



# Drawing specs (different colors per class, lines only)
spec_lips = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=0)    # magenta
spec_eye_l = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=0)     # green
spec_eye_r = mp_drawing.DrawingSpec(color=(0, 200, 200), thickness=1, circle_radius=0)   # cyan-ish
spec_nose = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=0)    # yellow



def to_uint8_rgb(img_tensor):
    """
    img_tensor: torch.Tensor (3, H, W) in [0,1] float
    returns: np.ndarray (H, W, 3) uint8 RGB
    """
    # Ensure on CPU and contiguous
    arr = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return arr

def annotate_face_parts(image_rgb):
    """
    Draw only lips, left/right eyes, and nose connections on a copy of image_rgb.
    image_rgb: np.uint8 RGB
    returns: np.uint8 RGB (annotated)
    """
    res = mp_face_mesh.process(image_rgb)
    annotated = image_rgb.copy()

    if not res.multi_face_landmarks:
        return annotated  # no face, return original

    face_landmarks = res.multi_face_landmarks[0]

    # Draw each class separately (no cross-class connections)
    mp_drawing.draw_landmarks(
        annotated, face_landmarks, FACEMESH_LIPS,
        landmark_drawing_spec=None,
        connection_drawing_spec=spec_lips
    )
    mp_drawing.draw_landmarks(
        annotated, face_landmarks, FACEMESH_LEFT_EYE,
        landmark_drawing_spec=None,
        connection_drawing_spec=spec_eye_l
    )
    mp_drawing.draw_landmarks(
        annotated, face_landmarks, FACEMESH_RIGHT_EYE,
        landmark_drawing_spec=None,
        connection_drawing_spec=spec_eye_r
    )
    mp_drawing.draw_landmarks(
        annotated, face_landmarks, FACEMESH_NOSE,
        landmark_drawing_spec=None,
        connection_drawing_spec=spec_nose
    )
    return annotated







def main():
    dataset = LipSyncDataset(
        frames_dir=frames_dir,
        mels_dir=mels_dir,
        sequence_length=5,
        transform=T.ToTensor(),
        mel_transform=None,
        enforce_consecutive=True,
        mel_height_target=None,
        mel_pad_value=0.0,
        return_paths=True,
        recursive=True,
    )

    # For the very first run, keep workers=0 so we can verify everything else is fine
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,                 # <-- start with 0 to rule out multiprocessing issues
        pin_memory=False,              # <-- flip to True later if you like
        persistent_workers=False,      # <-- safer during iteration
        collate_fn=CollatePadMel(pad_value=0.0),  # <-- no lambda
    )


        
        
    # ---- Your existing loop, now with saving ----
    for batch_idx, (imgs, mel, meta) in enumerate(loader):
        print(imgs.shape, mel.shape, meta[0])
        
        # imgs: (B, L, 3, H, W)
        B, L, C, H, W = imgs.shape
    
        for b in range(B):
            # If your dataset was created with return_paths=True, you have paths here:
            seq_paths = meta[b].get("frame_paths", [None] * L)
    
            for t in range(L):
                # Convert tensor to uint8 RGB
                img_rgb = to_uint8_rgb(imgs[b, t])
    
                # Annotate with MP FaceMesh
                annotated_rgb = annotate_face_parts(img_rgb)
    
                # Build output filename
                if seq_paths and seq_paths[t] is not None:
                    base = os.path.basename(seq_paths[t])
                    name, ext = os.path.splitext(base)
                    out_name = f"{name}_annotated{ext}"
                else:
                    out_name = f"b{batch_idx}_s{b}_t{t}.png"
    
                out_path = os.path.join(out_dir, out_name)
    
                # Save; cv2 expects BGR, so convert
                cv2.imwrite(out_path, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))

        # If you really want to stop after the first batch:
        break
        
        
        
        

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
    # Close MediaPipe resources
    mp_face_mesh.close()

