import cv2

# --- Settings ---
VIDEO_PATH = "D:/Training Data/CP1E03/040.mxf"  # <-- change this
SCALE = 0.25               # 0.5x width and height => 1/4 pixels
WINDOW_NAME = "MXF Preview (1/4 res)"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}\nTip: Ensure your OpenCV uses the FFmpeg backend that supports MXF.")

# Try to fetch FPS to pace playback; fallback if missing.
fps = cap.get(cv2.CAP_PROP_FPS)
delay_ms = int(1000 / fps) if fps and fps > 0 else 1

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

while True:
    ok, frame = cap.read()
    if not ok:
        break  # end of stream or read error

    # Resize to ~1/4 pixel count by halving each dimension
    resized = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

    cv2.imshow(WINDOW_NAME, resized)

    # Keyboard controls:
    #   q or ESC: quit
    #   space: pause/resume
    key = cv2.waitKey(delay_ms) & 0xFF
    if key in (27, ord('q')):  # ESC or q
        break
    elif key == ord(' '):  # space to pause
        # Wait until another key press; break on q/ESC
        while True:
            k2 = cv2.waitKey(50) & 0xFF
            if k2 in (27, ord('q'), ord(' ')):
                if k2 in (27, ord('q')):
                    cap.release()
                    cv2.destroyAllWindows()
                    raise SystemExit
                else:
                    break

cap.release()
cv2.destroyAllWindows()
