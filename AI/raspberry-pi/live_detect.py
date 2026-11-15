import cv2
import numpy as np
import onnxruntime as ort
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "best.onnx"
CAMERA = 0
PORT = 8000
CONF = 0.25
IOU = 0.45
NAMES = ["Person", "Dog", "Cattle", "Squirrel"]

session = ort.InferenceSession(
    MODEL, providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Cannot open USB webcam")

def detect(frame):
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

    for class_id in range(4):
        boxes, scores = [], []

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

            boxes.append([
                int(x1), int(y1),
                int(x2 - x1), int(y2 - y1)
            ])
            scores.append(score)

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, CONF, IOU
        )

        if len(indices):
            for idx in np.array(indices).flatten():
                x, y, bw, bh = boxes[idx]
                score = scores[idx]

                cv2.rectangle(
                    frame, (x, y), (x+bw, y+bh),
                    (0, 255, 0), 2
                )
                label = f"{NAMES[class_id]} {score:.2f}"
                cv2.putText(
                    frame, label, (x, max(y-8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2
                )

    return frame

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = detect(frame)
                ok, jpg = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if not ok:
                    continue

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(
                    b"Content-Type: image/jpeg\r\n\r\n"
                )
                self.wfile.write(jpg.tobytes())
                self.wfile.write(b"\r\n")

        except (BrokenPipeError, ConnectionResetError):
            pass

server = ThreadingHTTPServer(("0.0.0.0", PORT), StreamHandler)
print(f"Live detection server running on port {PORT}")
print("Open http://<raspberry-pi-ip>:8000/ on a device on the same network")

try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    server.server_close()
