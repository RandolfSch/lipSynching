import cv2
import av
import numpy as np

# ========= Paths / UI =========
VIDEO_PATH = "D:/Training Data/CP1E03/726.mxf"   # <-- your file
WINDOW_NAME = "Speech/Silence Tint"

# ========= Tint behavior =========
# Silence -> red tint (boost R, dim G and B)
# Speech  -> green tint (boost G, dim R and B)
TINT_UP   = 1.6     # boost factor on the "active" channel
TINT_DOWN = 0.6     # dim factor on the other two channels
DRAW_LABEL = True

# ========= Silence detector (your parameters) =========
# Matches the logic from your detect_silence.py:
#   - moving maximum over windows of raw waveform
#   - per-window silence flag
#   - merge close silent windows
#   - keep only segments longer than min_silence_duration
WINDOW_SIZE_MS        = 100     # moving-max window size (ms)
SILENCE_THRESHOLD     = 0.75   # max level for a window to be considered silent
MERGE_THRESHOLD_SEC   = 0.12     # merge silence gaps shorter than this (s)
MIN_SILENCE_DURATION  = 0.10    # discard silence shorter than this (s)

# ======================================================

def decode_audio_mono(path):
    """
    Decode the first audio stream from `path` to mono float32 in [-1, 1].
    Returns (audio, sr). If none, returns (None, None).
    """
    container = av.open(path)
    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if audio_stream is None:
        container.close()
        return None, None

    chunks = []
    sr = None
    for frame in container.decode(audio=audio_stream.index):
        arr = frame.to_ndarray()  # shape (channels, samples) or (samples,)
        if arr.ndim == 1:
            mono = arr.astype(np.float32)
        else:
            mono = arr.mean(axis=0).astype(np.float32)

        # Normalize to [-1,1] if integer; clamp if float goes beyond
        if np.issubdtype(mono.dtype, np.integer):
            maxv = float(np.iinfo(mono.dtype).max)
            mono = mono.astype(np.float32) / maxv
        else:
            m = np.max(np.abs(mono)) if mono.size else 1.0
            if m > 1.0:
                mono = mono / m

        chunks.append(mono)
        if sr is None:
            sr = frame.sample_rate

    container.close()
    if not chunks:
        return None, None
    return np.concatenate(chunks), sr

def moving_max_1d(x, win):
    """
    Sliding maximum with O(n) deque algorithm; avoids SciPy dependency.
    x: 1-D numpy array
    win: window length (samples) >= 1
    Returns array of same length as x (valid mode with padding on edges).
    """
    from collections import deque
    n = len(x)
    if win <= 1 or n == 0:
        return x.copy()

    dq = deque()
    out = np.empty(n, dtype=x.dtype)

    # Initialize deque for first window
    for i in range(win):
        while dq and x[i] >= x[dq[-1]]:
            dq.pop()
        dq.append(i)
    out[win-1] = x[dq[0]]

    # Slide
    for i in range(win, n):
        # remove out-of-window
        while dq and dq[0] <= i - win:
            dq.popleft()
        # push new
        while dq and x[i] >= x[dq[-1]]:
            dq.pop()
        dq.append(i)
        out[i] = x[dq[0]]

    # For the first win-2 outputs, copy the first computed max (simple leading pad)
    out[:win-1] = out[win-1]
    return out

def detect_silence_segments_from_wave(y, sr,
                                      window_size_ms=WINDOW_SIZE_MS,
                                      silence_threshold=SILENCE_THRESHOLD,
                                      merge_threshold=MERGE_THRESHOLD_SEC,
                                      min_silence_duration=MIN_SILENCE_DURATION):
    """
    Implements the same algorithm as your detect_silence.py but operates directly on the waveform array.
    Returns: list of (start, end) silent segments in seconds.
    """
    # 1) Moving maximum over raw waveform (like your maximum_filter1d)
    win_samples = max(1, int(round(sr * window_size_ms / 1000.0)))
    # If SciPy is available, you can uncomment and prefer it:
    # from scipy.ndimage import maximum_filter1d
    # moving_max = maximum_filter1d(y, size=win_samples)
    moving_max = moving_max_1d(y, win_samples)

    # 2) Split into non-overlapping windows and flag per-window silence
    num_windows = len(moving_max) // win_samples
    moving_max = moving_max[:num_windows * win_samples]
    if num_windows == 0:
        return []

    # reshape to (num_windows, win_samples) -> mean of (moving_max < threshold) per window
    mm2 = moving_max.reshape(num_windows, win_samples)
    is_silence_window = (mm2 < silence_threshold).mean(axis=1) > 0.95  # same 95% rule

    # 3) Build raw segments from silent windows
    silent_segments = []
    in_sil = False
    start_t = 0.0
    for i, flag in enumerate(is_silence_window):
        if flag and not in_sil:
            start_t = (i * win_samples) / sr
            in_sil = True
        elif not flag and in_sil:
            end_t = (i * win_samples) / sr
            silent_segments.append((start_t, end_t))
            in_sil = False
    if in_sil:
        # file ends with silence
        end_t = len(y) / sr
        silent_segments.append((start_t, end_t))

    if not silent_segments:
        return []

    # 4) Merge close silences (gaps shorter than merge_threshold)
    merged = []
    cur_s, cur_e = silent_segments[0]
    for s, e in silent_segments[1:]:
        if s - cur_e < merge_threshold:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # 5) Keep only segments >= min_silence_duration
    final_segments = [(s, e) for (s, e) in merged if (e - s) >= min_silence_duration]
    return final_segments

def make_silence_lookup(segments):
    """
    Build a callable that returns True if time t is inside any [start, end) silence segment.
    Uses binary search over starts for O(log N) queries.
    """
    if not segments:
        return lambda t: False

    starts = np.array([s for s, _ in segments], dtype=np.float64)
    ends   = np.array([e for _, e in segments], dtype=np.float64)

    def is_silence(t):
        if t is None:
            return False
        # index of first segment with start > t
        i = int(np.searchsorted(starts, t, side='right')) - 1
        if i < 0 or i >= len(starts):
            return False
        return starts[i] <= t < ends[i]

    return is_silence

def tint_bgr(img_bgr, speech, up=TINT_UP, down=TINT_DOWN):
    """
    Red tint on silence, green on speech.
    OpenCV uses BGR channel order.
    """
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    if speech:
        # Green tint
        g *= up; r *= down; b *= down
        label = ("SPEECH", (0, 255, 0))
    else:
        # Red tint
        r *= up; g *= down; b *= down
        label = ("SILENCE", (0, 0, 255))
    out = cv2.merge((b, g, r))
    out = np.clip(out, 0, 255).astype(np.uint8)

    if DRAW_LABEL:
        text, color = label
        cv2.putText(out, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
    return out

def main():
    # --- Prepass: audio -> silence segments (your method) ---
    y, sr = decode_audio_mono(VIDEO_PATH)
    if y is None or sr is None:
        # No audio: treat as all silence
        is_silence_at = lambda t: True
    else:
        segments = detect_silence_segments_from_wave(
            y, sr,
            window_size_ms=WINDOW_SIZE_MS,
            silence_threshold=SILENCE_THRESHOLD,
            merge_threshold=MERGE_THRESHOLD_SEC,
            min_silence_duration=MIN_SILENCE_DURATION
        )
        is_silence_at = make_silence_lookup(segments)

    # --- Video playback with tint ---
    container = av.open(VIDEO_PATH)
    video_stream = next((s for s in container.streams if s.type == "video"), None)
    if video_stream is None:
        container.close()
        raise RuntimeError("No video stream found.")

    # approximate pacing
    fps = float(video_stream.average_rate) if video_stream.average_rate else 25.0
    delay_ms = max(1, int(round(1000.0 / fps)))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        for frame in container.decode(video=video_stream.index):
            bgr = frame.to_ndarray(format="bgr24")

            # Prefer precise timestamp if .time is None
            t = frame.time
            if t is None and frame.pts is not None and frame.time_base is not None:
                t = float(frame.pts * frame.time_base)

            silence = is_silence_at(t)
            tinted = tint_bgr(bgr, speech=not silence)

            cv2.imshow(WINDOW_NAME, tinted)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        container.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
