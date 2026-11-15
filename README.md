# Integrated Intelligent Wildlife Protection System

An IoT-based wildlife protection and monitoring system integrating smart sensors, embedded systems, camera-based surveillance, and automated deterrence to reduce human-wildlife conflict and enhance wildlife safety.

## Repository contents

This repository currently includes the AI wildlife-detection and camera-experiment subsystem.

- `AI/raspberry-pi/`: live camera inference and the Raspberry Pi experiment server.
- `AI/experiment-controller/`: controller used to run camera-based detection trials.
- `AI/training/`: YOLO training configuration snapshots.
- `results/training/`: training metrics exported as CSV files.
- `results/experiments/`: trial-level and detection-level experiment logs.
- `results/visualizations/`: model evaluation and training plots.
- `docs/dataset-analysis/`: dataset audit and analysis summaries.

The sensor-network, fence-monitoring, and deterrence modules should be added as those modules are ready. The presence of AI files here does not imply that the complete wildlife-protection system has been deployed.

## AI subsystem

The current detection scripts use an ONNX model named `best.onnx` and recognise the classes `Person`, `Dog`, `Cattle`, and `Squirrel`.

The trained model file is not included in this upload bundle. Place your approved model file alongside the relevant Raspberry Pi script before running inference. Check the model's size and dataset/model licence before publishing it.

## Setup

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r AI/requirements.txt
```

Set the Raspberry Pi server address before running the experiment controller:

```bash
# Linux/macOS
export PI_URL="http://<raspberry-pi-ip>:8000"

# Windows PowerShell
$env:PI_URL = "http://<raspberry-pi-ip>:8000"
```

Run the relevant script from the project root:

```bash
python AI/raspberry-pi/live_detect.py
python AI/raspberry-pi/experiment_server.py
python AI/experiment-controller/experiment_controller.py
```

Use the appropriate script for your hardware setup. The live detection script and experiment server both access a camera and load `best.onnx`; do not run both against the same camera simultaneously unless the camera supports multiple clients.

## Important notes

- Update the camera index in the relevant script to match the connected camera.
- Keep the Raspberry Pi and experiment-controller computer on the same trusted network.
- The experiment server currently listens on all network interfaces and does not implement authentication. Do not expose port `8000` to the public internet.
- Dataset photographs and the full dataset are not included. Confirm dataset redistribution permissions before publishing sample images or annotations.
- Training configuration files are historical snapshots. Review dataset paths, model checkpoints, and output directories before rerunning training.

## Results

Training metrics, experiment CSV logs, and evaluation plots are included for documentation. They are experimental records, not a guarantee of field performance.

## Contributions

Use branches and pull requests for changes. Keep `main` for reviewed and stable code, and document hardware changes, model versions, and experiment settings in commit messages.
