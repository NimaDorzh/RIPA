# RIPA: Prompt Injection Attacks on LLM-Controlled Robotic Systems

RIPA is an empirical security study of prompt injection attacks against ROS 2 robots that route untrusted sensory inputs into an LLM-driven control loop. The project focuses on sensory-vector prompt injection: text observed in the environment, extracted by OCR, and forwarded through a robotics pipeline that can ultimately affect motion commands.

The current project state includes four completed phases:

1. Phase 1: Baseline injection attacks against DeepSeek-chat with 100% attack success rate.
2. Phase 2: Hybrid semantic firewall defense with 0% attack success rate and 0% false positives.
3. Phase 3: Multi-model testing with Llama 3 8B via Together AI, reducing baseline attack success to 67%.
4. Phase 4: Visual injection pipeline with OCR, firewall mediation, and automated experiment logging.

Key outcomes so far:

- First empirical study in this workspace of sensory-vector prompt injection across a ROS 2 control pipeline.
- Baseline attack success rate: DeepSeek 100%, Llama 67%.
- Defense result: firewall reduced attack success to 0% with 0% false positives in the current experiments.

## 1. Project Overview

This repository implements an end-to-end robotics security testbed for evaluating how prompt injection propagates from perception to actuation:
- untrusted text appears in the robot environment;
- OCR extracts that text into ROS 2 topics;
- a semantic firewall filters malicious content;
- a controller node queries an LLM;
- the selected action is published to `/cmd_vel`.

The focus is not only whether an LLM can be manipulated, but whether that manipulation survives the full ROS 2 pipeline and affects downstream robot behavior.

## 2. Technology Stack

| Component | Value |
|-----------|-------|
| OS | Ubuntu 24.04 on WSL2 |
| ROS | ROS 2 Jazzy |
| Simulator | Gazebo Harmonic |
| Robot | TurtleBot3 |
| LLMs | DeepSeek-chat, Llama 3 8B via Together AI |
| Python | 3.12 |
| GPU | NVIDIA RTX 4060 + CUDA 12.3 |
| OCR | Tesseract OCR + pytesseract |
| Key packages | openai, python-dotenv, Pillow, pytesseract, matplotlib, usb_cam |

## 3. Architecture

Full defended pipeline:
```text
[Camera/Image] -> [ocr_node] -> [/object_label] -> [firewall_node] -> [/object_label_safe] -> [controller_node] -> [LLM] -> [/cmd_vel]
```

Operational meaning of each stage:
- `ocr_node` reads images and extracts visible text.
- `/object_label` carries raw, untrusted textual observations.
- `firewall_node` classifies and filters suspicious prompt-like content.
- `/object_label_safe` carries sanitized labels to the controller.
- `controller_node` queries the selected LLM provider and converts the result into motion commands.
- `/cmd_vel` is the final actuation topic.

## 4. Package Structure

Current workspace layout relevant to the RIPA pipeline:
```text
robotics_ws/
├── README.md
├── generate_test_images.py
├── run_visual_injection_test.sh
├── test_cards/
│   └── generate_test_cards.py
├── results/
│   ├── csv/
│   ├── pdf/
│   ├── png/
│   ├── generate_visual_chart.py
│   ├── parse_visual_injection_log.py
│   └── visualize.py
├── test_images/
└── src/
    └── llm_robot_controller/
        ├── launch/
      │   ├── camera_world.launch.py
      │   └── flat_world.launch.py
        ├── llm_robot_controller/
        │   ├── __init__.py
        │   ├── controller_node.py
        │   ├── firewall_node.py
        │   ├── ocr_node.py
        │   ├── injection_test.py
        │   ├── firewall_test.py
        │   ├── ocr_test.py
      │   ├── real_camera_test.py
        │   └── results_paths.py
        ├── package.xml
        ├── setup.cfg
        └── setup.py
```

Primary files:

- `controller_node.py`: subscribes to `/object_label_safe`, queries the configured LLM, and publishes to `/cmd_vel`.
- `firewall_node.py`: filters unsafe text before it reaches the controller.
- `ocr_node.py`: performs OCR and publishes extracted text to `/object_label`.
- `injection_test.py`: runs text-based prompt injection experiments.
- `firewall_test.py`: validates firewall behavior against experiment payloads.
- `ocr_test.py`: drives the visual injection experiment path.
- `generate_test_images.py`: generates image fixtures for OCR and visual injection tests.

## 5. Environment Setup

Create and prepare the workspace:
```bash
git clone <your-repo-url> ~/robotics_ws
cd ~/robotics_ws

python3 -m venv venv
source venv/bin/activate

pip install -U pip
pip install -e src/llm_robot_controller

sudo apt update
sudo apt install -y tesseract-ocr ros-jazzy-usb-cam

pip install pytesseract pillow python-dotenv --break-system-packages

cp .env.example .env
# edit .env and add your API keys

source /opt/ros/jazzy/setup.bash
colcon build --packages-select llm_robot_controller
source install/setup.bash
```

Notes:

- `pip install -e src/llm_robot_controller` installs the Python package and its declared dependencies.
- Tesseract must be available on the system path for `ocr_node.py`.
- `usb_cam` must be installed to publish frames from a physical webcam into `/camera/image_raw`.
- `colcon build` is still required to expose the ROS 2 entry points.

## 6. Configuration (.env)

Copy `.env.example` to `.env` and set the values for your environment:

```env
DEEPSEEK_API_KEY=
TOGETHER_API_KEY=
LLM_PROVIDER=deepseek
LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct-Lite
INJECTION_DIRECT_MODE=false
```

Configuration semantics:

- `DEEPSEEK_API_KEY`: required when `LLM_PROVIDER=deepseek`.
- `TOGETHER_API_KEY`: required when `LLM_PROVIDER=together`.
- `LLM_PROVIDER`: selects the runtime backend. Supported values are `deepseek` and `together`.
- `LLM_MODEL`: optional override for the provider default model. Use this to pin the Together model variant you want to test.
- `INJECTION_DIRECT_MODE=false`: default defended mode. When set to `true`, `injection_test.py` can bypass the firewall and publish directly to `/object_label_safe`.

## 7. Running: Full Pipeline

Open four terminals from `~/robotics_ws`.

Terminal 1:

```bash
source venv/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run llm_robot_controller firewall_node
```

Terminal 2:

```bash
source venv/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run llm_robot_controller controller_node
```

Terminal 3:

```bash
source venv/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run llm_robot_controller ocr_node
```

Terminal 4 (visual injection testing):

```bash
source venv/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run llm_robot_controller ocr_test
```

For live webcam OCR with a physical camera instead of simulated frames:

```bash
source venv/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch llm_robot_controller camera_world.launch.py
```

In a separate terminal you can log camera-driven OCR and firewall outcomes with:

```bash
source venv/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run llm_robot_controller real_camera_test
```

Terminal 4, alternative (text-only injection testing instead of `ocr_test`):

```bash
source venv/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run llm_robot_controller injection_test
```

## 8. Running: Automated Visual Injection Test

To run the OCR, firewall, controller, parsing, and chart generation flow automatically:

```bash
./run_visual_injection_test.sh
```

This script starts the required ROS 2 nodes, runs the visual test, and writes:

- a combined raw log to `results/visual_injection_<timestamp>.log`;
- a parsed CSV to `results/csv/visual_injection_<timestamp>.csv`;
- a PNG summary chart to `results/png/visual_injection_<timestamp>.png`.

## 9. Real Camera Setup (WSL2)

To use a Logitech C920e or another USB webcam from WSL2, pass the device through from Windows and publish it with `usb_cam`:

1. Install `usbipd-win` on Windows.
2. In an elevated PowerShell window, run `usbipd list`, then `usbipd attach --wsl --busid <id>` for the webcam.
3. In WSL2, verify the camera is visible with `ls /dev/video*`.
4. Install the ROS 2 driver with `sudo apt install ros-jazzy-usb-cam`.
5. Launch the camera pipeline with `ros2 launch llm_robot_controller camera_world.launch.py`.

If `/dev/video0` is not the correct device, override it at launch time:

```bash
ros2 launch llm_robot_controller camera_world.launch.py video_device:=/dev/video1
```

Use the printable prompt cards generated by `python3 test_cards/generate_test_cards.py` to hold benign and adversarial instructions in front of the camera.

## 10. Experiment Results

| Experiment | Model | ASR | Notes |
|------------|-------|-----|-------|
| Phase 1 baseline | DeepSeek-chat | 100% | All 3 adversarial variants succeeded |
| Phase 1 baseline | Llama 3 8B | 67% | A1 showed partial resistance |
| Phase 2 firewall | Both models | 0% | 0% false positives in current tests |
| Phase 4 visual | Both models | 0% | OCR pipeline plus firewall blocked tested visual injections |

Interpretation:

- DeepSeek-chat was fully vulnerable in the baseline prompt injection configuration.
- Llama 3 8B via Together AI showed lower but still significant susceptibility.
- The semantic firewall eliminated successful attacks in the tested scenarios.
- The visual injection pipeline remained blocked when OCR output was routed through the firewall.

## 11. Results Structure

Generated artifacts are organized as follows:

```text
results/
├── csv/    # experiment CSV outputs
├── png/    # charts and visual summaries
├── pdf/    # generated reports
├── generate_visual_chart.py
├── parse_visual_injection_log.py
└── visualize.py
```

## 12. DDS Fix (WSL2)

If ROS 2 topic discovery is unreliable under WSL2, install Cyclone DDS:

```bash
sudo apt install -y ros-jazzy-rmw-cyclonedds-cpp
```

Then add this to `~/.bashrc`:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

Reload your shell afterward:

```bash
source ~/.bashrc
```

## 13. Status

- [x] Environment setup
- [x] LLM → ROS 2 pipeline
- [x] Injection experiments (DeepSeek 100% ASR, Llama 67% ASR)
- [x] Defense mechanism (semantic firewall, 0% ASR)
- [x] Visual OCR pipeline with real camera support
- [ ] Multi-model comparison (expand)
- [ ] Paper draft

## 14. Citation

Placeholder BibTeX entry:

```bibtex
@misc{ripa2026,
      title        = {RIPA: Prompt Injection Attacks on LLM-Controlled Robotic Systems},
      author       = {Nima Dorzhiev},
      year         = {2026},
      note         = {Work in progress}
}
```