"""
Meat type detector - Flask backend (CES117V capstone).

Loads the fine-tuned YOLOv8n weights (best.pt), accepts an uploaded video or
photo, draws a box and label on every detected piece of raw meat, and returns
the annotated file plus a short summary of what was found.

Routes:
  /               the web page (templates/index.html)
  /health         quick check that the app is running and which classes the model knows
  /predict        POST a video or photo in the form field "file"; returns JSON
  /outputs/<name> serves the annotated video or photo
"""
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from ultralytics import YOLO

# ---------- 1. SETTINGS ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "best.pt"))
WORK_DIR = Path(os.environ.get("WORK_DIR", Path(tempfile.gettempdir()) / "meat_detector"))
UPLOAD_DIR = WORK_DIR / "uploads"
OUTPUT_DIR = WORK_DIR / "outputs"

CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.35))  # ignore weaker detections
IMG_SIZE = int(os.environ.get("IMG_SIZE", 416))                 # MUST match the image size used in training
TARGET_FPS = 10        # frames analysed per second of video (keeps CPU hosting fast)
MAX_FRAMES = 300       # about 30 seconds of video at TARGET_FPS
MAX_SIDE = 960         # longest side of the output, in pixels
MAX_UPLOAD_MB = 100
FILE_LIFETIME = 60 * 60  # delete results after one hour

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Box colours (BGR for OpenCV). The web page uses the same colours.
CLASS_COLORS = {
    "beef": (43, 57, 192),
    "pork": (159, 135, 232),
    "chicken": (0, 180, 244),
    "lamb": (194, 87, 126),
    "fish": (210, 118, 25),
}
# Used for any class name that is not in CLASS_COLORS (in case the dataset names differ).
EXTRA_COLORS = [(75, 180, 60), (0, 130, 200), (255, 105, 180), (0, 128, 255), (128, 128, 0), (255, 0, 255)]
FALLBACK_COLOR = (115, 107, 91)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model weights not found at {MODEL_PATH}. Copy your trained best.pt next to app.py."
    )

# ---------- 2. LOAD THE MODEL ONCE, WHEN THE APP STARTS ----------
model = YOLO(str(MODEL_PATH))


def build_color_map(names):
    """Give every class the model knows a box colour (known names first, then spare colours)."""
    colors, spare = {}, 0
    for name in names.values():
        key = str(name).lower()
        if key in CLASS_COLORS:
            colors[key] = CLASS_COLORS[key]
        else:
            colors[key] = EXTRA_COLORS[spare % len(EXTRA_COLORS)]
            spare += 1
    return colors


COLOR_MAP = build_color_map(model.names)
MODEL_LOCK = threading.Lock()  # one analysis at a time on the shared model

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# ---------- 3. HELPER FUNCTIONS ----------
def color_for(label):
    return COLOR_MAP.get(label.lower(), FALLBACK_COLOR)


def text_color_for(bgr):
    b, g, r = bgr
    return (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 150 else (255, 255, 255)


def cleanup_old_files():
    cutoff = time.time() - FILE_LIFETIME
    for folder in (UPLOAD_DIR, OUTPUT_DIR):
        for path in folder.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass


def draw_detections(frame, result):
    """Draw boxes and labels on the frame. Returns {label: best confidence in this frame}."""
    height, width = frame.shape[:2]
    thickness = max(2, round(min(height, width) / 300))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(height, width) / 900)
    best = {}

    for box in result.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        label = str(result.names[int(box.cls[0])])
        conf = float(box.conf[0])
        best[label] = max(best.get(label, 0.0), conf)

        color = color_for(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        text = f"{label} {conf:.0%}"
        (tw, th), base = cv2.getTextSize(text, font, font_scale, 1)
        top = max(0, y1 - th - base - 6)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 8, top + th + base + 6), color, -1)
        cv2.putText(frame, text, (x1 + 4, top + th + 2), font, font_scale,
                    text_color_for(color), 1, cv2.LINE_AA)
    return best


def summarise(tally, frames_seen):
    """Turn {label: [best confidence per frame]} into a sorted list for the web page."""
    rows = []
    for label, confs in tally.items():
        rows.append({
            "label": label,
            "frames": len(confs),
            "share": round(len(confs) / max(frames_seen, 1), 3),
            "avg_conf": round(sum(confs) / len(confs), 3),
            "max_conf": round(max(confs), 3),
        })
    rows.sort(key=lambda row: row["frames"], reverse=True)
    return rows


def make_browser_safe(raw_path, final_path):
    """Re-encode to H.264 so every browser can play the result. Falls back to the raw file."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(raw_path),
               "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(final_path)]
        if subprocess.run(cmd).returncode == 0:
            raw_path.unlink(missing_ok=True)
            return
    raw_path.replace(final_path)


def process_image(upload_path, job):
    image = cv2.imread(str(upload_path))
    if image is None:
        raise ValueError("That image could not be read. Try a JPG or PNG photo.")

    scale = min(1.0, MAX_SIDE / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    result = model.predict(image, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)[0]
    best = draw_detections(image, result)

    out_path = OUTPUT_DIR / f"{job}.jpg"
    cv2.imwrite(str(out_path), image)
    tally = {label: [conf] for label, conf in best.items()}
    return {
        "kind": "image",
        "url": url_for("serve_output", name=out_path.name),
        "summary": summarise(tally, 1),
        "frames_analysed": 1,
        "truncated": False,
    }


def process_video(upload_path, job):
    cap = cv2.VideoCapture(str(upload_path))
    if not cap.isOpened():
        raise ValueError("That video could not be opened. Try an MP4 or MOV clip.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        cap.release()
        raise ValueError("That video has no readable frames.")

    scale = min(1.0, MAX_SIDE / max(width, height))
    out_w = max(2, int(width * scale) // 2 * 2)   # H.264 needs even dimensions
    out_h = max(2, int(height * scale) // 2 * 2)

    stride = max(1, round(fps / TARGET_FPS))
    raw_path = OUTPUT_DIR / f"{job}_raw.mp4"
    final_path = OUTPUT_DIR / f"{job}.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / stride, (out_w, out_h))

    tally = {}
    analysed = 0
    index = 0
    truncated = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            if analysed >= MAX_FRAMES:
                truncated = True
                break
            if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            result = model.predict(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)[0]
            for label, conf in draw_detections(frame, result).items():
                tally.setdefault(label, []).append(conf)
            writer.write(frame)
            analysed += 1
        index += 1

    cap.release()
    writer.release()

    if analysed == 0:
        raw_path.unlink(missing_ok=True)
        raise ValueError("No frames could be read from that video.")

    make_browser_safe(raw_path, final_path)
    return {
        "kind": "video",
        "url": url_for("serve_output", name=final_path.name),
        "summary": summarise(tally, analysed),
        "frames_analysed": analysed,
        "truncated": truncated,
    }


# ---------- 4. PAGES, PREDICTION AND RESULT FILES ----------
@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_UPLOAD_MB)


@app.get("/health")
def health():
    classes = []
    for name in model.names.values():
        b, g, r = color_for(str(name))
        classes.append({"name": str(name), "color": f"#{r:02x}{g:02x}{b:02x}"})
    return jsonify(status="ok", classes=classes)


@app.post("/predict")
def predict():
    cleanup_old_files()

    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(error="Choose a video or photo first."), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in VIDEO_EXT | IMAGE_EXT:
        return jsonify(error="Unsupported file type. Use MP4, MOV, AVI, MKV, WEBM, JPG, PNG or WEBP."), 400

    job = uuid.uuid4().hex[:12]
    upload_path = UPLOAD_DIR / f"{job}{ext}"
    file.save(upload_path)
    started = time.time()

    try:
        with MODEL_LOCK:
            payload = process_video(upload_path, job) if ext in VIDEO_EXT else process_image(upload_path, job)
    except ValueError as exc:
        return jsonify(error=str(exc)), 422
    except Exception:
        app.logger.exception("Detection failed")
        return jsonify(error="Something went wrong while analysing the file. Try a shorter clip."), 500
    finally:
        upload_path.unlink(missing_ok=True)

    payload["seconds"] = round(time.time() - started, 1)
    return jsonify(payload)


@app.get("/outputs/<name>")
def serve_output(name):
    return send_from_directory(OUTPUT_DIR, name, conditional=True)


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error=f"That file is too large. Keep uploads under {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
