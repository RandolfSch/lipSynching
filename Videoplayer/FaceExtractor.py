
import os
import cv2
import math
import numpy as np

# If mediapipe is not installed: pip install mediapipe
import mediapipe as mp

# -------- Settings --------
VIDEO_PATH = "D:/Training Data/CP1E03/040.mxf"        # <-- your MXF path
OUT_DIR = "D:/Training Data/CarmenFrames"                 # output folder for crops
TARGET_SIZE = (512, 512)        # (width, height)
FRAME_STRIDE = 1                # process every Nth frame; e.g., 5 to process ~20%
MARGIN_FRAC = 0.10              # expand bbox by 10% on each side
STOP_AFTER = None               # set to an int to stop after N crops, e.g., 1
SAVE_FORMAT = "png"             # "png" or "jpg"
JPEG_QUALITY = 95               # if using jpg
# --------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def expand_and_clip_bbox(xmin, ymin, w, h, img_w, img_h, margin_frac):
    """
    Expand bbox by margin_frac, then clip to image bounds.
    Input bbox is in pixel coordinates (not normalized).
    Returns (x0, y0, x1, y1) as ints within image bounds.
    """
    cx = xmin + w * 0.5
    cy = ymin + h * 0.5
    # Expand the half-size by margin fraction
    half_w = (w * 0.5) * (1.0 + margin_frac * 2.0)
    half_h = (h * 0.5) * (1.0 + margin_frac * 2.0)

    # Convert to top-left / bottom-right
    x0 = int(round(cx - half_w))
    y0 = int(round(cy - half_h))
    x1 = int(round(cx + half_w))
    y1 = int(round(cy + half_h))

    # Clip to image bounds
    x0 = max(0, min(x0, img_w - 1))
    y0 = max(0, min(y0, img_h - 1))
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))

    # Ensure non-empty box
    if x1 <= x0: x1 = min(img_w, x0 + 1)
    if y1 <= y0: y1 = min(img_h, y0 + 1)

    return x0, y0, x1, y1

def pick_largest_detection(detections, img_w, img_h):
    """
    From mediapipe detections, select the one with largest area.
    Returns a tuple (xmin, ymin, w, h) in pixel coords, or None if no detections.
    """
    if not detections:
        return None

    best = None
    best_area = -1
    for det in detections:
        # MediaPipe FaceDetection returns relative bbox [0..1]
        rel = det.location_data.relative_bounding_box
        xmin = max(0.0, rel.xmin)
        ymin = max(0.0, rel.ymin)
        w = max(0.0, rel.width)
        h = max(0.0, rel.height)

        # Convert to pixel bbox
        px = int(round(xmin * img_w))
        py = int(round(ymin * img_h))
        pw = int(round(w * img_w))
        ph = int(round(h * img_h))

        area = pw * ph
        if area > best_area:
            best_area = area
            best = (px, py, pw, ph)

    return best

def main():
    ensure_dir(OUT_DIR)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    # MediaPipe Face Detection: short range is suitable for near-field faces.
    mp_fd = mp.solutions.face_detection
    face_det = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    total_saved = 0
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # Skip frames if stride > 1
        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        img_h, img_w = frame_bgr.shape[:2]

        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Run detection
        result = face_det.process(frame_rgb)

        if result.detections:
            # Pick the largest face
            bbox = pick_largest_detection(result.detections, img_w, img_h)
            if bbox is not None:
                px, py, pw, ph = bbox
                # Expand and clip with margin
                x0, y0, x1, y1 = expand_and_clip_bbox(px, py, pw, ph, img_w, img_h, MARGIN_FRAC)

                # Crop and resize to target
                crop = frame_bgr[y0:y1, x0:x1]
                resized = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)

                # Save
                out_name = f"frame_{frame_idx:06d}.{SAVE_FORMAT}"
                out_path = os.path.join(OUT_DIR, out_name)
                if SAVE_FORMAT.lower() == "jpg" or SAVE_FORMAT.lower() == "jpeg":
                    cv2.imwrite(out_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                else:
                    cv2.imwrite(out_path, resized)

                total_saved += 1

                if STOP_AFTER is not None and total_saved >= STOP_AFTER:
                    break

        frame_idx += 1

    cap.release()
    face_det.close()
    print(f"Done. Saved {total_saved} face crops to: {OUT_DIR}")

if __name__ == "__main__":
    main()
