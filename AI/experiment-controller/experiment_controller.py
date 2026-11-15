
import cv2
import csv
import time
import random
import os
import requests
from pathlib import Path
from datetime import datetime

# ---------------- CONFIGURATION ----------------

PI_URL = os.getenv("PI_URL", "http://raspberrypi.local:8000")

IMAGE_DIR = Path.home() / "YOLO_Experiment/YOLO_camera_test_export/images"
OUTPUT_DIR = Path.home() / "YOLO_Experiment/results"

SCREEN_W = 1280
SCREEN_H = 720

CLASSES = ["person", "dog", "cattle", "squirrel"]

CLASS_NAMES = {
    "person": "Person",
    "dog": "Dog",
    "cattle": "Cattle",
    "squirrel": "Squirrel"
}

SEED = 42
SETTLE_TIME = 1
TRIAL_LIMIT = 10

# True = test one image
# False = run all 160 images
TEST_MODE = False

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Unique ID prevents confusion with previous runs
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------- LOAD STIMULI ----------------

stimuli = []

for class_name in CLASSES:
    folder = IMAGE_DIR / class_name

    for extension in ("*.jpg", "*.jpeg", "*.png"):
        for image in folder.glob(extension):
            stimuli.append({
                "path": image,
                "class": CLASS_NAMES[class_name]
            })

if len(stimuli) != 160:
    raise RuntimeError(f"Expected 160 stimuli, found {len(stimuli)}")

random.Random(SEED).shuffle(stimuli)

if TEST_MODE:
    stimuli = stimuli[:1]

print(f"Loaded {len(stimuli)} stimuli")

# ---------------- CHECK PI ----------------

response = requests.get(f"{PI_URL}/status", timeout=5)
response.raise_for_status()

if response.json().get("active_trial") is not None:
    raise RuntimeError("Pi already has an active trial. Stop it before proceeding.")

print("Raspberry Pi connected")

# ---------------- DISPLAY WINDOW ----------------

window = "YOLO11n Experiment"

cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, SCREEN_W, SCREEN_H)

# ---------------- RESULTS ----------------

results_file = OUTPUT_DIR / (
    "test_trial.csv" if TEST_MODE else "trials.csv"
)

fields = [
    "trial_id",
    "stimulus_id",
    "expected_class",
    "predicted_class",
    "outcome",
    "end_reason",
    "duration_seconds",
    "frames_processed",
    "detection_events",
    "start_time",
    "end_time",
    "image_path"
]

# ---------------- RUN EXPERIMENT ----------------

with open(results_file, "w", newline="") as f:

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    try:
        for number, stimulus in enumerate(stimuli, start=1):

            trial_id = f"RUN{RUN_ID}_TRIAL_{number:03d}"

            image_path = stimulus["path"]
            expected_class = stimulus["class"]

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            # Resize while preserving aspect ratio
            h, w = image.shape[:2]

            scale = min(
                SCREEN_W / w,
                SCREEN_H / h
            )

            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(image, (new_w, new_h))

            left = (SCREEN_W - new_w) // 2
            right = SCREEN_W - new_w - left
            top = (SCREEN_H - new_h) // 2
            bottom = SCREEN_H - new_h - top

            display = cv2.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

            # Show stimulus and enter fullscreen
            cv2.imshow(window, display)
            cv2.setWindowProperty(
                window,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
            cv2.waitKey(1)

            print(
                f"\nTrial {number}/{len(stimuli)}: "
                f"{expected_class} - {image_path.name}"
            )

            # Allow TV image to stabilize
            settle_start = time.time()

            while time.time() - settle_start < SETTLE_TIME:
                cv2.imshow(window, display)
                cv2.waitKey(1)
                time.sleep(0.02)

            # Start Pi trial
            payload = {
                "trial_id": trial_id,
                "stimulus_id": image_path.name,
                "expected_class": expected_class,
                "expected_count": 0
            }

            response = requests.post(
                f"{PI_URL}/start_trial",
                json=payload,
                timeout=5
            )
            response.raise_for_status()

            # First wait for Pi to acknowledge this exact trial
            active_confirmed = False
            deadline = time.time() + 5

            while time.time() < deadline:

                cv2.imshow(window, display)
                cv2.waitKey(1)

                response = requests.get(
                    f"{PI_URL}/status",
                    timeout=5
                )
                response.raise_for_status()

                status = response.json()

                if status.get("active_trial") == trial_id:
                    active_confirmed = True
                    break

                time.sleep(0.1)

            if not active_confirmed:
                raise RuntimeError(
                    f"Pi did not confirm that {trial_id} started"
                )

            # Wait for this trial to finish
            trial_result = None
            deadline = time.time() + TRIAL_LIMIT + 8

            while time.time() < deadline:

                # Keep stimulus visible throughout the trial
                cv2.imshow(window, display)
                cv2.waitKey(1)

                response = requests.get(
                    f"{PI_URL}/status",
                    timeout=5
                )
                response.raise_for_status()

                status = response.json()

                if (
                    status.get("active_trial") is None
                    and status.get("last_trial") is not None
                    and status["last_trial"].get("trial_id") == trial_id
                ):
                    trial_result = status["last_trial"]
                    break

                time.sleep(0.05)

            if trial_result is None:
                raise RuntimeError(
                    f"No completed result received for {trial_id}"
                )

            # Save result
            row = {
                "trial_id": trial_id,
                "stimulus_id": image_path.name,
                "expected_class": expected_class,
                "predicted_class": trial_result["predicted_class"],
                "outcome": trial_result["outcome"],
                "end_reason": trial_result["end_reason"],
                "duration_seconds": trial_result["duration_seconds"],
                "frames_processed": trial_result["frames_processed"],
                "detection_events": trial_result["detection_events"],
                "start_time": trial_result["start_time"],
                "end_time": trial_result["end_time"],
                "image_path": str(image_path)
            }

            writer.writerow(row)
            f.flush()

            print(
                f"Result: {row['outcome']} | "
                f"Reason: {row['end_reason']} | "
                f"Duration: {row['duration_seconds']}s"
            )

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")

    except Exception as error:
        print(f"\nExperiment error: {error}")

    finally:
        if TEST_MODE:
            print("Test complete. Press any key to close the display.")
            cv2.waitKey(0)

        cv2.destroyAllWindows()

print(f"\nResults saved to {results_file}")
