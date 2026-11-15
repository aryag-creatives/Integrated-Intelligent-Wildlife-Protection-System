
import os
import cv2
import numpy as np
import onnxruntime as ort
import requests
import time
import uuid
import json
import threading

# =====================================================
# 1. SYSTEM CONFIGURATION
# =====================================================

MODEL = "best.onnx"
CAMERA = 0

CONF = 0.25
IOU = 0.45

NAMES = ["Person", "Dog", "Cattle", "Squirrel"]

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = str(os.getenv("CHAT_ID", ""))

ESP32_IP = os.getenv("ESP32_IP")
ESP32_API_KEY = os.getenv("ESP32_API_KEY")

TELEGRAM_URL = (
    f"https://api.telegram.org/bot{BOT_TOKEN}"
    if BOT_TOKEN else None
)

# Safety configuration
GATE_CONTROL_ENABLED = os.getenv("GATE_CONTROL_ENABLED", "true").lower() in ("1", "true", "yes")
AUTHORIZED_USER_ID = str(os.getenv("AUTHORIZED_USER_ID", CHAT_ID))

# Detection configuration
ALERT_COOLDOWN = 30
APPROVAL_TIMEOUT = 60

# PIR-triggered camera operation
PIR_POLL_INTERVAL = 0.5          # Poll ESP32 PIR status twice per second
CAMERA_ACTIVE_AFTER_MOTION = 30 # Keep YOLO active 30 seconds after last PIR HIGH

# MPU6050 fence tamper monitoring (pitch values are degrees)
MPU_NORMAL_PITCH = float(os.getenv("MPU_NORMAL_PITCH", "-2.34"))
MPU_TILT_THRESHOLD = float(os.getenv("MPU_TILT_THRESHOLD", "15.0"))
MPU_TAMPER_CONFIRM_SECONDS = float(os.getenv("MPU_TAMPER_CONFIRM_SECONDS", "2.0"))

# Animal danger configuration
DANGER_CLASSES = {"Dog", "Cattle"}
DANGER_MIN_CONFIDENCE = 0.40
DANGER_CLEAR_DELAY = 5

# =====================================================
# 2. SHARED SYSTEM STATE
# =====================================================

state_lock = threading.Lock()

pending_events = {}
event_decisions = {}
last_alert_time = {}

running = True

danger_active = False
last_danger_seen = 0
last_led_attempt = 0
mpu_tilt_started_at = None
mpu_tamper_alerted = False

# =====================================================
# 3. INITIALIZE YOLO
# =====================================================

print("Loading YOLO model...")

session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("YOLO model loaded successfully.")

# =====================================================
# 4. CAMERA STATE
# =====================================================

# The USB camera is opened only after the ESP32 reports PIR motion.
# It is released after the configured no-motion hold period.
cap = None

def start_camera():
    global cap

    if cap is not None and cap.isOpened():
        return True

    camera = cv2.VideoCapture(CAMERA)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        camera.release()
        print("Camera could not be opened.")
        return False

    cap = camera
    print("PIR triggered: camera active, YOLO detection started.")
    return True


def stop_camera():
    global cap
import os
import cv2
import numpy as np
import onnxruntime as ort
import requests
import time
import uuid
import json
import threading

# =====================================================
# 1. SYSTEM CONFIGURATION
# =====================================================

MODEL = "best.onnx"
CAMERA = 0

CONF = 0.25
IOU = 0.45

NAMES = ["Person", "Dog", "Cattle", "Squirrel"]

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = str(os.getenv("CHAT_ID", ""))

ESP32_IP = os.getenv("ESP32_IP")
ESP32_API_KEY = os.getenv("ESP32_API_KEY")

TELEGRAM_URL = (
    f"https://api.telegram.org/bot{BOT_TOKEN}"
    if BOT_TOKEN else None
)

# Safety configuration
GATE_CONTROL_ENABLED = os.getenv("GATE_CONTROL_ENABLED", "true").lower() in ("1", "true", "yes")
AUTHORIZED_USER_ID = str(os.getenv("AUTHORIZED_USER_ID", CHAT_ID))

# Detection configuration
ALERT_COOLDOWN = 30
APPROVAL_TIMEOUT = 60

# PIR-triggered camera operation
PIR_POLL_INTERVAL = 0.5          # Poll ESP32 PIR status twice per second
CAMERA_ACTIVE_AFTER_MOTION = 30 # Keep YOLO active 30 seconds after last PIR HIGH

# MPU6050 fence tamper monitoring (pitch values are degrees)
MPU_NORMAL_PITCH = float(os.getenv("MPU_NORMAL_PITCH", "-2.34"))
MPU_TILT_THRESHOLD = float(os.getenv("MPU_TILT_THRESHOLD", "15.0"))
MPU_TAMPER_CONFIRM_SECONDS = float(os.getenv("MPU_TAMPER_CONFIRM_SECONDS", "2.0"))

# Animal danger configuration
DANGER_CLASSES = {"Dog", "Cattle"}
DANGER_MIN_CONFIDENCE = 0.40
DANGER_CLEAR_DELAY = 5

# =====================================================
# 2. SHARED SYSTEM STATE
# =====================================================

state_lock = threading.Lock()

pending_events = {}
event_decisions = {}
last_alert_time = {}

running = True

danger_active = False
last_danger_seen = 0
last_led_attempt = 0
mpu_tilt_started_at = None
mpu_tamper_alerted = False

# =====================================================
# 3. INITIALIZE YOLO
# =====================================================

print("Loading YOLO model...")

session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("YOLO model loaded successfully.")

# =====================================================
# 4. CAMERA STATE
# =====================================================

# The USB camera is opened only after the ESP32 reports PIR motion.
# It is released after the configured no-motion hold period.
cap = None

    if cap is not None:
        cap.release()
        cap = None
        print("No recent PIR motion: camera released, system in standby.")

# =====================================================
# 5. YOLO DETECTION FUNCTION
# =====================================================

def detect(frame):

    h, w = frame.shape[:2]

    scale = min(640 / w, 640 / h)

    nw = int(w * scale)
    nh = int(h * scale)

    resized = cv2.resize(frame, (nw, nh))

    canvas = np.full(
        (640, 640, 3),
        114,
        dtype=np.uint8
    )

    px = (640 - nw) // 2
    py = (640 - nh) // 2

    canvas[py:py+nh, px:px+nw] = resized

    image = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )[None]

    output = session.run(
        None,
        {input_name: image}
    )[0]
    predictions = output[0].T

    detections = []

    for class_id in range(4):

        boxes = []
        scores = []

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
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1)
            ])

            scores.append(score)

        if not boxes:
            continue

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            CONF,
            IOU
        )

        if len(indices):

            for idx in np.array(indices).flatten():

                x, y, bw, bh = boxes[idx]

                detections.append({
                    "class": NAMES[class_id],
                    "confidence": scores[idx],
                    "box": (x, y, bw, bh)
                })

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + bw, y + bh),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{NAMES[class_id]} {scores[idx]:.2f}",
                    (x, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    return frame, detections


# =====================================================
# 6. ESP32 COMMUNICATION
# =====================================================

def send_tamper_telegram_alert(pitch_value, deviation):
    """Send one Telegram message when sustained fence tilt is confirmed."""
    if not TELEGRAM_URL or not CHAT_ID:
        print("MPU TAMPER ALERT: Telegram credentials missing.")
        return False

    message = (
        "⚠️ FENCE TAMPER ALERT\n\n"
        "Abnormal fence movement detected by the MPU6050.\n"
        f"Current pitch: {pitch_value:.2f}°\n"
        f"Normal pitch: {MPU_NORMAL_PITCH:.2f}°\n"
        f"Deviation: {deviation:.2f}°\n"
        f"Threshold: {MPU_TILT_THRESHOLD:.2f}°\n\n"
        "This alert does not operate the gate."
    )

    try:
        response = requests.post(
            f"{TELEGRAM_URL}/sendMessage",
            data={"chat_id": CHAT_ID, "text": message},
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        if result.get("ok"):
            print("MPU TAMPER ALERT sent to Telegram.")
            return True
        print("Telegram rejected MPU tamper alert:", result)
    except Exception as exc:
        print("Could not send MPU tamper alert:", exc)
    return False


def monitor_mpu_tamper(status):
    """Detect sustained pitch deviation; independent of gate control."""
    global mpu_tilt_started_at, mpu_tamper_alerted

    if not status or not status.get("mpu_available"):
        mpu_tilt_started_at = None
        return

    try:
        pitch_value = float(status["pitch"])
    except (KeyError, TypeError, ValueError):
        return

    deviation = abs(pitch_value - MPU_NORMAL_PITCH)
    now = time.monotonic()

    if deviation >= MPU_TILT_THRESHOLD:
        if mpu_tilt_started_at is None:
            mpu_tilt_started_at = now
            print(
                f"MPU: abnormal tilt started ({deviation:.2f}° from baseline); "
                "confirming for 2 seconds."
            )

        elif (not mpu_tamper_alerted and
              now - mpu_tilt_started_at >= MPU_TAMPER_CONFIRM_SECONDS):
            mpu_tamper_alerted = True
            send_tamper_telegram_alert(pitch_value, deviation)
    else:
        if mpu_tilt_started_at is not None:
            print("MPU: pitch returned within the normal range.")
        mpu_tilt_started_at = None
        mpu_tamper_alerted = False


def get_esp32_status():

    if not ESP32_IP:
        return None

    try:

        response = requests.get(
            f"http://{ESP32_IP}/status",
            timeout=2
        )

        response.raise_for_status()

        return response.json()

    except Exception as e:

        print("ESP32 status error:", e)

        return None


def esp32_command(command):
    if not ESP32_IP or not ESP32_API_KEY:
        print("ESP32 credentials missing.")
        return False

    try:

        response = requests.post(
            f"http://{ESP32_IP}/command",
            data={
                "key": ESP32_API_KEY,
                "cmd": command
            },
            timeout=5
        )

        result = response.json()

        if response.ok and result.get("ok"):

            print(
                "ESP32:",
                result.get("message")
            )

            return True

        print("ESP32 command rejected:", result)

        return False

    except Exception as e:

        print("ESP32 communication error:", e)

        return False




def request_gate_open():
    """
    Open only when the ESP32 is reachable, fault-free, and confirms
    that the gate is fully closed at the home limit.
    """
    if not GATE_CONTROL_ENABLED:
        print("Gate control is disabled in configuration.")
        return False, "Gate control is disabled."

    if not ESP32_IP or not ESP32_API_KEY:
        print("ESP32 credentials missing. Gate command withheld.")
        return False, "ESP32 connection settings are missing."

    status = get_esp32_status()
    if not status:
        return False, "Cannot reach ESP32. Gate remains unchanged."

    if status.get("status") != "online":
        return False, "ESP32 is not online. Gate remains unchanged."

    if status.get("error") or status.get("gate_state") == "FAULT":
        return False, "ESP32 reports a fault. Gate remains unchanged."

    if status.get("gate_state") != "CLOSED_IDLE":
        return (
            False,
            f"Gate is not confirmed closed (state: {status.get('gate_state')}). "
            "No OPEN command sent."
        )

    if status.get("home_limit") is not True or status.get("far_limit") is not False:
        return False, "Limit switches do not confirm the closed position. No OPEN command sent."

    if not esp32_command("OPEN"):
        return False, "ESP32 rejected or did not confirm OPEN. Check gate status."

    return True, "Gate opening command accepted. It will close automatically after 15 seconds fully open."


# =====================================================
# 7. ANIMAL DANGER LED CONTROL
# =====================================================

def update_danger_state(detections):

    global danger_active
    global last_danger_seen
    global last_led_attempt

    current_time = time.time()

    danger_detected = any(
        item["class"] in DANGER_CLASSES
        and item["confidence"] >= DANGER_MIN_CONFIDENCE
        for item in detections
    )

    if danger_detected:

        last_danger_seen = current_time

        if not danger_active:

            # Limit retries if ESP32 is unavailable
            if current_time - last_led_attempt < 5:
                return

            last_led_attempt = current_time

            status = get_esp32_status()

            if not status:
                print("Cannot verify ESP32. LED command withheld.")
                return

            if status.get("status") != "online":
                print("ESP32 offline. LED command withheld.")
                return

            if status.get("error"):
                print("ESP32 reports a hardware error. LED command withheld.")
                return

            if status.get("gate_state") == "FAULT":
                print("ESP32 FAULT. LED command withheld.")
                return

            if esp32_command("LED_ON"):

                danger_active = True

                print("DANGER ACTIVE: FENCE INDICATOR LED ON")

    elif danger_active:

        if current_time - last_danger_seen >= DANGER_CLEAR_DELAY:

            if current_time - last_led_attempt < 5:
                return

            last_led_attempt = current_time

            if esp32_command("LED_OFF"):

                danger_active = False

                print("DANGER CLEARED: FENCE INDICATOR LED OFF")


# =====================================================
# 8. TELEGRAM PHOTO ALERT
# =====================================================

def send_telegram_alert(frame, detections):

    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram credentials missing.")
        return

    event_id = str(uuid.uuid4())[:8]

    detected_classes = list(set(
        item["class"] for item in detections
    ))

    has_person = "Person" in detected_classes

    caption = (
        "GHARIAL WILDLIFE ALERT\n\n"
        f"Event ID: {event_id}\n"
        f"Detected: {', '.join(detected_classes)}\n"
    )

    keyboard = None

    if has_person:

        with state_lock:

            pending_events[event_id] = {                "status": "PENDING",
                "created_at": time.time()
            }

        keyboard = {
            "inline_keyboard": [[
                {
                    "text": "APPROVE",
                    "callback_data": f"APPROVE_{event_id}"
                },
                {
                    "text": "DENY",
                    "callback_data": f"DENY_{event_id}"
                }
            ]]
        }

        caption += "\nAwaiting human approval."

    else:

        caption += "\nAnimal detection notification."

    success, image_data = cv2.imencode(
        ".jpg",
        frame
    )

    if not success:

        print("Image encoding failed.")

        if has_person:
            with state_lock:
                pending_events.pop(event_id, None)

        return

    payload = {
        "chat_id": CHAT_ID,
        "caption": caption
    }

    if keyboard:

        payload["reply_markup"] = json.dumps(keyboard)

    try:

        response = requests.post(
            f"{TELEGRAM_URL}/sendPhoto",
            data=payload,
            files={
                "photo": (
                    "wildlife_alert.jpg",
                    image_data.tobytes(),
                    "image/jpeg"
                )
            },
            timeout=15
        )

        result = response.json()

        if result.get("ok"):

            print(
                "Telegram alert sent. Event ID:",
                event_id
            )

        else:

            print("Telegram error:", result)

            if has_person:

                with state_lock:
                    pending_events.pop(event_id, None)

    except Exception as e:

        print("Telegram notification failed:", e)

        if has_person:

            with state_lock:
                pending_events.pop(event_id, None)


# =====================================================
# 9. APPROVAL EVENT PROCESSING
# =====================================================

def process_decision(event_id, decision):

    # Reserve the event first so repeated button presses cannot issue
    # multiple gate commands while the first approval is being processed.
    with state_lock:
        event = pending_events.get(event_id)

        if event is None:
            return "Unknown event ID."

        if event["status"] != "PENDING":
            return "Event already processed or expired."

        if time.time() - event["created_at"] > APPROVAL_TIMEOUT:
            event["status"] = "EXPIRED"
            event_decisions[event_id] = "EXPIRED"
            return "Approval expired. Gate unchanged."

        if decision == "APPROVED":
            event["status"] = "PROCESSING"
        elif decision == "DENIED":
            event["status"] = "DENIED"
            event_decisions[event_id] = "DENIED"
        else:
            return "Invalid decision."

    print("\nPERSON EVENT DECISION")
    print("Event ID:", event_id)
    print("Decision:", decision)

    if decision == "DENIED":
        print("Gate remains closed. No gate command sent.")
        return f"Event {event_id}: DENIED. Gate remains closed."

    # Approval is not enough by itself: verify the ESP32 and closed
    # limit-switch state before sending OPEN.
    opened, message = request_gate_open()

    with state_lock:
        event = pending_events.get(event_id)
        if event is not None:
            event["status"] = "APPROVED" if opened else "FAILED"
        event_decisions[event_id] = "APPROVED" if opened else "FAILED"

    if opened:
        print("Gate OPEN command accepted by ESP32.")
        return f"Event {event_id}: APPROVED. {message}"

    print("Approval received, but gate was NOT opened:", message)
    return f"Event {event_id}: approval received, but gate was NOT opened. {message}"


def expire_pending_events():

    current_time = time.time()

    with state_lock:

        for event_id, event in list(pending_events.items()):

            if event["status"] != "PENDING":
                continue

            if current_time - event["created_at"] > APPROVAL_TIMEOUT:

                event["status"] = "EXPIRED"

                event_decisions[event_id] = "EXPIRED"

                print(
                    "Approval expired:",
                    event_id
                )


# =====================================================
# 10. TELEGRAM CALLBACK LISTENER
# =====================================================

def telegram_listener():

    print("Telegram approval listener started.")

    offset = None
    while running:

        try:

            params = {
                "timeout": 25
            }

            if offset is not None:

                params["offset"] = offset

            response = requests.get(
                f"{TELEGRAM_URL}/getUpdates",
                params=params,
                timeout=30
            )

            data = response.json()

            if not data.get("ok"):

                print("Telegram API error:", data)

                time.sleep(3)

                continue

            for update in data.get("result", []):

                offset = update["update_id"] + 1

                callback = update.get("callback_query")

                if not callback:
                    continue

                callback_id = callback["id"]

                message = callback.get("message", {})

                user_chat_id = str(
                    message.get("chat", {}).get("id", "")
                )
                clicking_user_id = str(
                    callback.get("from", {}).get("id", "")
                )

                choice = callback.get("data", "")

                if user_chat_id != CHAT_ID or clicking_user_id != AUTHORIZED_USER_ID:

                    reply = "Unauthorized user."

                elif choice.startswith("APPROVE_"):

                    event_id = choice[len("APPROVE_"):]

                    reply = process_decision(
                        event_id,
                        "APPROVED"
                    )

                elif choice.startswith("DENY_"):

                    event_id = choice[len("DENY_"):]

                    reply = process_decision(
                        event_id,
                        "DENIED"
                    )

                else:

                    reply = "Unknown button action."

                requests.post(
                    f"{TELEGRAM_URL}/answerCallbackQuery",
                    json={
                        "callback_query_id": callback_id,
                        "text": reply
                    },
                    timeout=10
                )

        except Exception as e:

            if running:

                print("Telegram listener error:", e)

                time.sleep(3)


# =====================================================
# 11. PIR-TRIGGERED YOLO DETECTION LOOP
# =====================================================

def detection_loop():

    print("PIR monitoring started. Camera is in standby.")
    print("PIR HIGH -> camera ON. Camera stays active 30 seconds after last motion.")

    last_status_poll = 0.0
    last_motion_time = 0.0
    last_camera_retry = 0.0

    while running:

        now = time.monotonic()

        # Poll the ESP32 for PIR state at a fixed interval.
        if now - last_status_poll >= PIR_POLL_INTERVAL:
            last_status_poll = now
            status = get_esp32_status()
            monitor_mpu_tamper(status)

            if status and status.get("status") == "online":
                if status.get("pir") is True:
                    last_motion_time = now

                    if cap is None:
                        # Avoid repeatedly hammering a missing camera.
                        if now - last_camera_retry >= 5:
                            last_camera_retry = now
                            start_camera()
            else:
                # If ESP32 status cannot be verified, do not wake the camera
                # based on stale PIR data. An already-active camera remains
                # active until its existing hold timer expires.
                print("PIR status unavailable; waiting for ESP32.")

        # Camera remains in standby until PIR reports motion.
        if cap is None:
            expire_pending_events()
            time.sleep(0.05)
            continue

        # After the last PIR HIGH, keep the camera active for 30 seconds.
        if last_motion_time and now - last_motion_time >= CAMERA_ACTIVE_AFTER_MOTION:
            # Give the animal LED logic a final chance to clear before standby.
            update_danger_state([])
            stop_camera()
            last_motion_time = 0.0
            continue

        ret, frame = cap.read()

        if not ret:
            print("Camera frame unavailable. Reinitializing camera.")
            stop_camera()
            last_camera_retry = now
            time.sleep(0.5)
            continue

        annotated_frame, detections = detect(frame)

        # Animal LED response continues while the camera is active.
        update_danger_state(detections)

        # Expire old approval requests even while detection is active.
        expire_pending_events()

        current_time = time.time()

        detected_classes = set(
            item["class"] for item in detections
        )

        # Select classes whose alert cooldown has elapsed.
        eligible_classes = {
            name for name in detected_classes
            if current_time - last_alert_time.get(name, 0)
            >= ALERT_COOLDOWN
        }

        if eligible_classes:
            alert_detections = [
                item for item in detections                if item["class"] in eligible_classes
            ]

            if alert_detections:
                send_telegram_alert(
                    annotated_frame.copy(),
                    alert_detections
                )

                for name in eligible_classes:
                    last_alert_time[name] = current_time

        time.sleep(0.01)


# =====================================================
# 12. MAIN SYSTEM STARTUP
# =====================================================

def main():

    global running

    if not BOT_TOKEN or not CHAT_ID:

        raise RuntimeError(
            "BOT_TOKEN or CHAT_ID is missing."
        )

    if not ESP32_IP or not ESP32_API_KEY:

        print(
            "WARNING: ESP32 credentials missing. "
            "ESP32 LED and gate control unavailable."
        )

    listener_thread = threading.Thread(
        target=telegram_listener,
        daemon=True
    )

    listener_thread.start()

    try:

        print("\n================================")

        print("GHARIAL SYSTEM ONLINE")

        print("PIR MONITORING: ACTIVE")

        print("MPU6050 TAMPER MONITORING: ACTIVE")
        print(f"MPU BASELINE: {MPU_NORMAL_PITCH:.2f} degrees")
        print(f"MPU TILT THRESHOLD: {MPU_TILT_THRESHOLD:.2f} degrees for {MPU_TAMPER_CONFIRM_SECONDS:.1f} seconds")
        print("MPU ALERTS: TELEGRAM (NO GATE CONTROL)")

        print("CAMERA: PIR-TRIGGERED (30-SECOND HOLD)")

        print("YOLO: ACTIVE WHEN CAMERA IS AWAKE")

        print("TELEGRAM: ACTIVE")

        print("ESP32 LED CONTROL: ACTIVE"
              if ESP32_IP and ESP32_API_KEY
              else "ESP32 LED CONTROL: UNAVAILABLE")

        print("GATE CONTROL: ACTIVE" if GATE_CONTROL_ENABLED and ESP32_IP and ESP32_API_KEY else "GATE CONTROL: UNAVAILABLE / DISABLED")

        print("================================\n")

        detection_loop()

    except KeyboardInterrupt:

        print("\nShutdown requested.")

    finally:

        running = False

        stop_camera()

        print("Camera released.")

        print("Gharial system stopped.")


# =====================================================
# 13. PROGRAM ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
