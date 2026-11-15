import cv2
import numpy as np
import onnxruntime as ort
import threading
import time
import json
import csv
import os
import re
from collections import deque, Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "best.onnx"
CAMERA = 1
PORT = 8000
CONF = 0.80
IOU = 0.45
TRIAL_SECONDS = 10
STABLE_WINDOW = 5
STABLE_REQUIRED = 3
NAMES = ["Person", "Dog", "Cattle", "Squirrel"]

OUT = "experiment_results"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/frames", exist_ok=True)

session = ort.InferenceSession(
    MODEL, providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Cannot open USB webcam")

lock = threading.RLock()
latest_jpg = None
latest_frame = None
active = None
last_trial = None

def append_csv(filename, row):
    path = f"{OUT}/{filename}"
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def detect(frame):
    start = time.perf_counter()
    h, w = frame.shape[:2]

    scale = min(640 / w, 640 / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))

    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    px = (640 - nw) // 2
    py = (640 - nh) // 2
    canvas[py:py+nh, px:px+nw] = resized

    image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))[None]

    output = session.run(None, {input_name: image})[0]
    predictions = output[0].T
    detections = []

    for class_id in range(4):
        boxes, scores, coordinates = [], [], []

        for pred in predictions:
            score = float(pred[4 + class_id])
            if score < CONF:
                continue

            cx, cy, bw, bh = pred[:4]
            x1 = (cx - bw / 2 - px) / scale
            y1 = (cy - bh / 2 - py) / scale
            x2 = (cx + bw / 2 - px) / scale
            y2 = (cy + bh / 2 - py) / scale

            x1, x2 = np.clip([x1, x2], 0, w)
            y1, y2 = np.clip([y1, y2], 0, h)

            if x2 <= x1 or y2 <= y1:
                continue

            box = [
                int(x1), int(y1),
                int(x2 - x1), int(y2 - y1)
            ]
            boxes.append(box)
            scores.append(score)
            coordinates.append((int(x1), int(y1), int(x2), int(y2)))

        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF, IOU)

        if len(indices):
            for idx in np.array(indices).flatten():
                x1, y1, x2, y2 = coordinates[idx]
                score = scores[idx]

                detections.append({
                    "class_id": class_id,
                    "class_name": NAMES[class_id],
                    "confidence": score,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2
                })

    elapsed_ms = (time.perf_counter() - start) * 1000

    annotated = frame.copy()
    for d in detections:
        cv2.rectangle(
            annotated, (d["x1"], d["y1"]),
            (d["x2"], d["y2"]), (0, 255, 0), 2
        )
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        cv2.putText(
            annotated, label,
            (d["x1"], max(d["y1"] - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

    return annotated, detections, elapsed_ms

def finish_trial(reason):
    global active, last_trial

    if active is None:
        return

    t = active
    elapsed = time.time() - t["start"]
    counts = Counter(t["class_counts"])
    predicted = counts.most_common(1)[0][0] if counts else ""

    outcome = "detected" if reason == "stable_detection" else "missed"

    summary = {
        "trial_id": t["trial_id"],
        "stimulus_id": t["stimulus_id"],
        "expected_class": t["expected_class"],
        "expected_count": t["expected_count"],
        "predicted_class": predicted,
        "outcome": outcome,
        "end_reason": reason,
        "duration_seconds": round(elapsed, 3),
        "frames_processed": t["frames"],
        "detection_events": t["events"],
        "start_time": t["start_iso"],
        "end_time": datetime.now().isoformat()
    }

    if latest_frame is not None:
        path = f'{OUT}/frames/{t["trial_id"]}_final.jpg'
        cv2.imwrite(path, latest_frame)

    append_csv("trials.csv", summary)
    last_trial = summary
    active = None
    print("Trial finished:", json.dumps(summary))

def camera_loop():
    global latest_jpg, latest_frame

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        annotated, detections, latency = detect(frame)
        now = time.time()

        with lock:
            latest_frame = annotated.copy()

            if active is not None:
                t = active
                t["frames"] += 1
                frame_number = t["frames"]
                elapsed = now - t["start"]

                expected_id = t["expected_id"]
                target_present = (
                    expected_id is not None and
                    any(d["class_id"] == expected_id for d in detections)
                )

                t["latest_detections"] = [
                    {
                        "class_name": d["class_name"],
                        "confidence": round(d["confidence"], 5)
                    }
                    for d in detections
                ]

                if expected_id is not None:
                    t["stability"].append(target_present)

                t["stability_hits"] = sum(t["stability"])

                for d in detections:
                    t["events"] += 1
                    t["class_counts"].append(d["class_name"])

                    append_csv("detections.csv", {
                        "trial_id": t["trial_id"],
                        "stimulus_id": t["stimulus_id"],
                        "frame_number": frame_number,
                        "elapsed_seconds": round(elapsed, 4),
                        "timestamp": datetime.now().isoformat(),
                        "class_id": d["class_id"],
                        "class_name": d["class_name"],
                        "confidence": round(d["confidence"], 5),
                        "x1": d["x1"], "y1": d["y1"],
                        "x2": d["x2"], "y2": d["y2"],
                        "inference_ms": round(latency, 3)
                    })

                if detections:
                    frame_path = (
                        f'{OUT}/frames/{t["trial_id"]}_'
                        f'{frame_number:04d}.jpg'
                    )
                    cv2.imwrite(frame_path, annotated)

                if (
                    expected_id is not None and
                    len(t["stability"]) == STABLE_WINDOW and
                    sum(t["stability"]) >= STABLE_REQUIRED
                ):
                    finish_trial("stable_detection")

                elif elapsed >= TRIAL_SECONDS:
                    finish_trial("timeout")

            ok, encoded = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            if ok:
                latest_jpg = encoded.tobytes()

class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        global latest_jpg

        if self.path == "/status":
            with lock:
                self.send_json({
                    "active_trial": active["trial_id"] if active else None,
                    "active_details": ({
                        "stimulus_id": active["stimulus_id"],
                        "frames_processed": active["frames"],
                        "detection_events": active["events"],
                        "latest_detections": active["latest_detections"],
                        "stability_hits": active["stability_hits"],
                        "stable_required": STABLE_REQUIRED,
                        "elapsed_seconds": round(time.time() - active["start"], 3)
                    } if active else None),
                    "last_trial": last_trial,
                    "confidence": CONF,
                    "iou": IOU
                })
            return

        if self.path == "/":
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()

            try:
                while True:
                    with lock:
                        jpg = latest_jpg
                    if jpg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(
                        b"Content-Type: image/jpeg\r\n\r\n"
                    )
                    self.wfile.write(jpg + b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_error(404)

    def do_POST(self):
        global active

        if self.path == "/start_trial":
            data = self.read_json()

            required = ["trial_id", "stimulus_id", "expected_class"]
            if any(k not in data for k in required):
                self.send_json({"error": "Missing required fields"}, 400)
                return

            trial_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(data["trial_id"]))
            expected = str(data["expected_class"])
            expected_id = NAMES.index(expected) if expected in NAMES else None

            with lock:
                if active is not None:
                    self.send_json({"error": "A trial is already active"}, 409)
                    return

                active = {
                    "trial_id": trial_id,
                    "stimulus_id": str(data["stimulus_id"]),
                    "expected_class": expected,
                    "expected_count": int(data.get("expected_count", 0)),
                    "expected_id": expected_id,
                    "start": time.time(),
                    "start_iso": datetime.now().isoformat(),
                    "frames": 0,
                    "events": 0,
                    "class_counts": [],
                    "stability": deque(maxlen=STABLE_WINDOW),
                    "stability_hits": 0,
                    "latest_detections": []
                }

            self.send_json({"started": True, "trial_id": trial_id})
            return

        if self.path == "/stop_trial":
            with lock:
                if active is None:
                    self.send_json({"error": "No active trial"}, 409)
                    return
                finish_trial("manual_stop")
            self.send_json({"stopped": True})
            return

        self.send_error(404)

    def log_message(self, format, *args):
        print(format % args)

threading.Thread(target=camera_loop, daemon=True).start()

server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
print(f"Experiment server running on port {PORT}")
print(f"Confidence={CONF}, IoU={IOU}")
print("Stream: http://<raspberry-pi-ip>:8000/")
print("Status: http://<raspberry-pi-ip>:8000/status")

try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    server.server_close()
