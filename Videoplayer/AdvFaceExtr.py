
import os
import cv2
import numpy as np
import mediapipe as mp

# === Audio / mel dependencies ===
import av  # PyAV for audio decode
import librosa
import tifffile

# -------- Settings --------
INPUT_DIR   = r"D:/Training Data/CP1E03"          # <-- folder of .mxf files
OUT_DIR     = r"D:/Training Data/CarmenFrames"    # output folder
TARGET_SIZE = (512, 512)                          # (w, h)
FRAME_STRIDE = 1                                  # process every Nth decoded frame
MARGIN_FRAC  = 0.10
STOP_AFTER    = None                               # max saved samples per video; None = all
SAVE_FORMAT   = "png"                              # "png" | "jpg"
JPEG_QUALITY  = 95
VIDEO_INDEX_BASE = 0                               # vid numbering: 0→vid_000_..., or set to 1

# --- Log-mel configuration (balanced defaults for lip-sync) ---
AUDIO_TARGET_SR   = 16000   # Hz
N_MELS            = 80
WIN_LENGTH_MS     = 25      # STFT window (ms)
HOP_LENGTH_MS     = 10      # STFT hop (ms)
MEL_WIDTH         = 86      # fixed hop count per frame (controls stripe width)
MEL_FMIN, MEL_FMAX = 50, 7600  # mel band; tweak to taste

# ----------------------------------------------------------------
# MediaPipe FaceMesh subset (exactly like your legacy precompute)
# ----------------------------------------------------------------
mp_connections = mp.solutions.face_mesh_connections
USE_SETS = ("LIPS", "LEFT_EYE", "RIGHT_EYE", "NOSE")
SETNAME2SET = {
    "LIPS": mp_connections.FACEMESH_LIPS,
    "LEFT_EYE": mp_connections.FACEMESH_LEFT_EYE,
    "RIGHT_EYE": mp_connections.FACEMESH_RIGHT_EYE,
    "NOSE": mp_connections.FACEMESH_NOSE,
}
_raw_ids = set()
for S in USE_SETS:
    for (i, j) in SETNAME2SET[S]:
        _raw_ids.add(i); _raw_ids.add(j)
RAW_IDS = np.array(sorted(_raw_ids), dtype=np.int32)  # rows of coords follow this order

# --------- Helpers ---------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_mxf_files(folder: str):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".mxf")]
    files.sort()
    return [os.path.join(folder, f) for f in files]

def expand_and_clip_bbox(xmin, ymin, w, h, img_w, img_h, margin_frac):
    cx = xmin + w * 0.5
    cy = ymin + h * 0.5
    half_w = (w * 0.5) * (1.0 + margin_frac * 2.0)
    half_h = (h * 0.5) * (1.0 + margin_frac * 2.0)
    x0 = int(round(cx - half_w)); y0 = int(round(cy - half_h))
    x1 = int(round(cx + half_w)); y1 = int(round(cy + half_h))
    x0 = max(0, min(x0, img_w - 1)); y0 = max(0, min(y0, img_h - 1))
    x1 = max(0, min(x1, img_w));     y1 = max(0, min(y1, img_h))
    if x1 <= x0: x1 = min(img_w, x0 + 1)
    if y1 <= y0: y1 = min(img_h, y0 + 1)
    return x0, y0, x1, y1

def pick_largest_detection(detections, img_w, img_h):
    if not detections:
        return None
    best = None; best_area = -1
    for det in detections:
        rel = det.location_data.relative_bounding_box
        xmin = max(0.0, rel.xmin); ymin = max(0.0, rel.ymin)
        w = max(0.0, rel.width);   h = max(0.0, rel.height)
        px = int(round(xmin * img_w)); py = int(round(ymin * img_h))
        pw = int(round(w * img_w));    ph = int(round(h * img_h))
        area = pw * ph
        if area > best_area:
            best_area = area; best = (px, py, pw, ph)
    return best

def run_facemesh_on_rgb(image_rgb, mp_face_mesh):
    res = mp_face_mesh.process(image_rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([(p.x, p.y) for p in lm], dtype=np.float32)  # normalized [0,1]
    return pts

def extract_selected(full468):
    return full468[RAW_IDS, :]  # (N,2) normalized

# --------- Audio → log-mel (whole clip) ---------
def decode_audio_mono(path):
    """
    Return mono float32 audio and its sample rate, or (None, None) if no audio.
    """
    container = av.open(path)
    astream = next((s for s in container.streams if s.type == "audio"), None)
    if astream is None:
        container.close()
        return None, None
    chunks = []
    sr = None
    for frame in container.decode(audio=astream.index):
        arr = frame.to_ndarray()
        if arr.ndim == 1:
            mono = arr.astype(np.float32)
        else:
            mono = arr.mean(axis=0).astype(np.float32)
        # Normalize if integer type
        if np.issubdtype(mono.dtype, np.integer):
            maxv = float(np.iinfo(mono.dtype).max)
            mono = mono / maxv
        else:
            max_abs = np.max(np.abs(mono)) if mono.size else 1.0
            if max_abs > 1.0:
                mono = mono / max_abs
        chunks.append(mono)
        if sr is None:
            sr = frame.sample_rate
    container.close()
    if not chunks:
        return None, None
    audio = np.concatenate(chunks)
    return audio, sr

def compute_logmel_fullclip(y, sr):
    """
    Resample to AUDIO_TARGET_SR, compute log-mel over the entire clip.
    Returns (mel: [n_mels, T], hop_len_samples, sr_out)
    """
    if sr != AUDIO_TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=AUDIO_TARGET_SR, res_type="kaiser_best")
        sr = AUDIO_TARGET_SR

    win_length = int(round(WIN_LENGTH_MS * 1e-3 * sr))
    hop_length = int(round(HOP_LENGTH_MS * 1e-3 * sr))
    # Hann window STFT → power mel
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2 ** int(np.ceil(np.log2(win_length))),
        hop_length=hop_length, win_length=win_length, window="hann",
        n_mels=N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX, power=2.0, center=True, pad_mode="reflect"
    )
    logmel = librosa.power_to_db(S, ref=np.max).astype(np.float32)  # (n_mels, T)
    return logmel, hop_length, sr

class MelStripeExtractor:
    def __init__(self, mel, hop_len, sr):
        self.mel = mel                  # (n_mels, T)
        self.hop_len = hop_len          # samples
        self.sr = sr
        self.dt = hop_len / float(sr)   # seconds per hop
        self.T = mel.shape[1]

    def slice_centered(self, t_sec, width=MEL_WIDTH, pad_value=0.0):
        """
        Return (n_mels, width) stripe centered at t_sec, zero-padded at edges.
        """
        if t_sec is None:
            t_sec = 0.0
        center = int(round(t_sec / self.dt))
        half = width // 2
        # center-inclusive symmetric window: [center-half, center+half-1] if width even
        start = center - half
        end = start + width  # exclusive
        # Pad if necessary
        pad_left = max(0, -start)
        pad_right = max(0, end - self.T)
        start_clamped = max(0, start)
        end_clamped = min(self.T, end)
        stripe = self.mel[:, start_clamped:end_clamped]
        if pad_left or pad_right:
            stripe_padded = np.full((self.mel.shape[0], width), pad_value, dtype=np.float32)
            stripe_padded[:, pad_left:width - pad_right] = stripe
            return stripe_padded
        return stripe

# --------- Robust "save all-or-nothing" ---------
def save_triplet_atomic(img_bgr, coords_sel, mel_stripe, stem, saved_count_ref):
    """
    Save PNG/JPG, NPZ, and TIFF. If any fails, delete others.
    Increments saved_count_ref[0] only when all succeed.
    """
    img_out = os.path.join(OUT_DIR, f"{stem}.{SAVE_FORMAT}")
    npz_out = os.path.join(OUT_DIR, f"{stem}.npz")
    tif_out = os.path.join(OUT_DIR, f"{stem}.tiff")

    saved_paths = []

    try:
        # Save TIFF (float32 log-mel)
        tifffile.imwrite(tif_out, mel_stripe.astype(np.float32))
        saved_paths.append(tif_out)

        # Save NPZ (strict legacy format)
        np.savez_compressed(npz_out, valid=True, coords=coords_sel.astype(np.float32))
        saved_paths.append(npz_out)

        # Save image
        ok = False
        if SAVE_FORMAT.lower() in ("jpg", "jpeg"):
            ok = cv2.imwrite(img_out, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        else:
            ok = cv2.imwrite(img_out, img_bgr)
        if not ok:
            raise RuntimeError("cv2.imwrite returned False")
        saved_paths.append(img_out)

        saved_count_ref[0] += 1
        return True

    except Exception as e:
        print(f"[WARN] Saving failed for stem {stem}: {e}")
        # Roll back any partial files
        for p in saved_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e2:
                print(f"[WARN] Could not remove partial file {p}: {e2}")
        return False

# --------- Core: per-video processing ---------
def process_video_file(video_path: str, vid_idx: int, face_det, mp_face_mesh):
    """
    For each decoded frame:
      - name stem = vid_{XXX}_{YYY} where XXX=vid_idx (3 digits), YYY=frame index in video (3 digits, starting at 000).
      - if FaceMesh succeeds: save PNG+NPZ+TIFF; else save nothing.
    """
    # 1) Decode audio and precompute log-mel (whole clip)
    y, sr = decode_audio_mono(video_path)
    if y is None or sr is None:
        print(f"[WARN] No audio found in {os.path.basename(video_path)} → skipping this video.")
        return 0
    mel_full, hop_len, sr_out = compute_logmel_fullclip(y, sr)
    melx = MelStripeExtractor(mel_full, hop_len, sr_out)

    # 2) OpenCV for frames (keep your original video pipeline)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return 0

    frame_idx = 0
    saved_count = [0]  # mutable ref for atomic saver

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            stem = f"vid_{(vid_idx + VIDEO_INDEX_BASE):03d}_{frame_idx:03d}"

            if (frame_idx % FRAME_STRIDE) == 0:
                img_h, img_w = frame_bgr.shape[:2]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Face detection (bbox)
                result = face_det.process(frame_rgb)
                if result.detections:
                    bbox = pick_largest_detection(result.detections, img_w, img_h)
                    if bbox is not None:
                        px, py, pw, ph = bbox
                        x0, y0, x1, y1 = expand_and_clip_bbox(px, py, pw, ph, img_w, img_h, MARGIN_FRAC)

                        crop_bgr = frame_bgr[y0:y1, x0:x1]
                        if crop_bgr.size != 0:
                            resized_bgr = cv2.resize(crop_bgr, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                            resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

                            # FaceMesh on crop
                            lm468 = run_facemesh_on_rgb(resized_rgb, mp_face_mesh)
                            if lm468 is not None:
                                sel = extract_selected(lm468).astype(np.float32)  # (N,2) normalized

                                # Timestamp (sec) for current frame
                                # OpenCV pos in ms is aligned to the *last retrieved* frame
                                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                                t_sec = float(t_ms) / 1000.0 if t_ms and t_ms > 0 else (frame_idx / max(1.0, cap.get(cv2.CAP_PROP_FPS)))

                                # Extract fixed-width mel stripe centered at t_sec
                                mel_stripe = melx.slice_centered(t_sec, width=MEL_WIDTH, pad_value=0.0)  # (N_MELS, MEL_WIDTH)

                                # Save all three atomically
                                ok_all = save_triplet_atomic(resized_bgr, sel, mel_stripe, stem, saved_count)
                                if not ok_all:
                                    # nothing to do; files already rolled back
                                    pass

                                if STOP_AFTER is not None and saved_count[0] >= STOP_AFTER:
                                    break

            frame_idx += 1

    finally:
        cap.release()

    print(f"[INFO] Video #{vid_idx:03d}: saved {saved_count[0]} samples.")
    return saved_count[0]

def main():
    ensure_dir(OUT_DIR)

    # Create detectors once
    mp_fd = mp.solutions.face_detection
    face_det = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    # Enumerate videos
    mxf_paths = list_mxf_files(INPUT_DIR)
    if not mxf_paths:
        print(f"[WARN] No .mxf files found in: {INPUT_DIR}")
        return

    total_saved_all = 0
    try:
        for vid_idx, path in enumerate(mxf_paths):
            print(f"[INFO] Processing ({vid_idx+1}/{len(mxf_paths)}): {os.path.basename(path)}")
            total_saved_all += process_video_file(path, vid_idx, face_det, mp_face_mesh)
    finally:
        face_det.close()
        mp_face_mesh.close()

    print(f"[DONE] Saved a total of {total_saved_all} samples to: {OUT_DIR}")

if __name__ == "__main__":
    main()
