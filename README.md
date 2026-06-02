# RIPA: Robotic Injection via Pipeline Attack

**Empirical study of sensory-vector prompt injection attacks on ROS 2 LLM-controlled robots.**

RIPA evaluates how adversarial text injected through a robot's physical sensors — camera, microphone, LiDAR — propagates through a ROS 2 pipeline and manipulates LLM-driven motion commands. The project covers three attack channels, five LLMs, a hybrid semantic firewall, and a firewall bypass taxonomy across 19 obfuscated payloads.

> Paper: *RIPA: Robotic Injection via Pipeline Attack* (preprint forthcoming)  
> Code: https://github.com/NimaDorzh/RIPA

---

## Key Results

| Channel | Vector | Models tested | ASR |
|---------|--------|---------------|-----|
| Channel 1 | Visual (OCR) | DeepSeek, Llama 3.1 8B, Llama 3.3 70B, Qwen 2.5 7B, Gemma-3n-4B | 67–100% |
| Channel 2 | Audio (Whisper STT) | DeepSeek-chat | 97–100% per variant |
| Channel 3 | LiDAR sensor context | DeepSeek-chat | 100% |
| Firewall (known patterns) | — | Both models | 0% ASR, 0% false positives |
| Firewall bypass (obfuscated) | 19 payloads, N=30 | Llama 3.3 70B (controller) | 52.6% bypass rate |

**Multi-model sweep (Channel 1, N=100 per variant via Together AI):**

| Model | Params | A1 | A2 | A3 | Overall |
|-------|--------|----|----|----|---------|
| DeepSeek-chat | ~67B (MoE) | 100% | 100% | 100% | 100% |
| Llama-3.3-70B | 70B | 100% | 100% | 100% | 100% |
| Qwen-2.5-7B | 7B | 100% | 100% | 100% | 100% |
| Gemma-3n-4B | 4B | 100% | 100% | 100% | 100% |
| Llama-3.1-8B | 8B | 60% | 80% | 60% | 66.7% |

---

## Architecture

### Channel 1 — Visual (OCR)
```
[Camera / Image] → [ocr_node] → /object_label → [firewall_node] → /object_label_safe → [controller_node] → [LLM] → /cmd_vel
```

### Channel 2 — Audio (Whisper STT)
```
[Microphone / WAV] → [audio_listener_node] → Whisper base → /object_label → [controller_node] → [LLM] → /cmd_vel
```

### Channel 3 — LiDAR Sensor Context Poisoning
```
[sensor_spoof_node] → /scan (fake LaserScan) → [sensor_context_node] → /sensor_context → [sensor_controller_node] → [LLM system prompt] → /cmd_vel
```

---

## Technology Stack

| Component | Value |
|-----------|-------|
| OS | Ubuntu 24.04 (WSL2) |
| ROS | ROS 2 Jazzy + Cyclone DDS |
| Simulator | Gazebo Harmonic |
| Robot | TurtleBot3 Waffle |
| LLMs | DeepSeek-chat (Platform API + Together AI), Llama 3.1 8B, Llama 3.3 70B, Qwen 2.5 7B, Gemma-3n-4B |
| Python | 3.12 |
| GPU | NVIDIA RTX 4060 Laptop + CUDA 12.3 |
| OCR | Tesseract 5 + pytesseract |
| STT | OpenAI Whisper base |
| LiDAR | sensor_msgs/LaserScan (simulated, TurtleBot3 Waffle) |
| Key packages | openai, python-dotenv, gtts, pydub, openai-whisper, Pillow, pytesseract, matplotlib |

---

## Repository Structure

```
robotics_ws/
├── README.md
├── results/
│   ├── csv/                          # all experiment outputs
│   ├── png/                          # charts and visualizations
│   └── pdf/
├── test_images/                      # OCR test fixtures
├── test_cards/
│   └── generate_test_cards.py        # printable adversarial cards
└── src/llm_robot_controller/
    └── llm_robot_controller/
        ├── controller_node.py         # Channel 1 LLM controller
        ├── firewall_node.py           # hybrid semantic firewall
        ├── ocr_node.py                # OCR → /object_label
        ├── injection_test.py          # Channel 1 baseline experiment
        ├── multi_model_sweep.py       # 5-model sweep via Together AI
        ├── firewall_test.py           # firewall validation
        ├── firewall_bypass_test.py    # 19-payload bypass taxonomy (N=30)
        ├── ocr_test.py                # visual injection experiment
        ├── real_camera_test.py        # live webcam injection
        ├── audio_listener_node.py     # Channel 2: Whisper STT node
        ├── audio_injection_test.py    # Channel 2: end-to-end audio experiment
        ├── sensor_spoof_node.py       # Channel 3: fake LaserScan publisher
        ├── sensor_context_node.py     # Channel 3: LaserScan → text context
        ├── sensor_controller_node.py  # Channel 3: LLM controller
        └── sensor_injection_test.py   # Channel 3: experiment runner
```

---

## Setup

```bash
git clone https://github.com/NimaDorzh/RIPA.git ~/robotics_ws
cd ~/robotics_ws

python3 -m venv venv
source venv/bin/activate
pip install -U pip

# System dependencies
sudo apt update
sudo apt install -y tesseract-ocr ros-jazzy-usb-cam

# Python dependencies
pip install openai python-dotenv pytesseract pillow matplotlib \
            gtts pydub openai-whisper --break-system-packages

# ROS 2 build
source /opt/ros/jazzy/setup.bash
colcon build --packages-select llm_robot_controller
source install/setup.bash

# Configuration
cp .env.example .env
# Add your API keys to .env
```

### .env configuration

```env
DEEPSEEK_API_KEY=your_key_here
TOGETHER_API_KEY=your_key_here
LLM_PROVIDER=together
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
INJECTION_DIRECT_MODE=false
```

### WSL2 DDS fix

```bash
sudo apt install -y ros-jazzy-rmw-cyclonedds-cpp
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc
source ~/.bashrc
```

---

## Running Experiments

### Channel 1 — Baseline injection (text)

```bash
source venv/bin/activate && source /opt/ros/jazzy/setup.bash && source install/setup.bash
python3 src/llm_robot_controller/llm_robot_controller/injection_test.py
```

### Channel 1 — Multi-model sweep (5 models, N=100)

```bash
python3 src/llm_robot_controller/llm_robot_controller/multi_model_sweep.py --runs 100
```

### Channel 1 — Firewall bypass taxonomy (19 payloads, N=30)

```bash
python3 src/llm_robot_controller/llm_robot_controller/firewall_bypass_test.py --runs 30
```

### Channel 2 — Audio injection

```bash
python3 src/llm_robot_controller/llm_robot_controller/audio_injection_test.py
```

### Channel 3 — LiDAR sensor context poisoning

```bash
python3 src/llm_robot_controller/llm_robot_controller/sensor_injection_test.py
```

### Channel 1 — Full visual pipeline (OCR + firewall + controller)

Start nodes in separate terminals:
```bash
ros2 run llm_robot_controller firewall_node
ros2 run llm_robot_controller controller_node
ros2 run llm_robot_controller ocr_node
ros2 run llm_robot_controller ocr_test
```

---

## Experiment Results (CSV)

All raw results are in `results/csv/`:

| File | Description |
|------|-------------|
| `experiment_deepseek_flash_100runs_*.csv` | Channel 1 baseline, DeepSeek Platform API, N=100 |
| `experiment_together_lite_100runs_*.csv` | Channel 1 baseline, Llama 3.1 8B, N=100 |
| `multi_model_sweep_*.csv` | 5-model sweep, N=100 per variant |
| `firewall_bypass_*.csv` | Bypass taxonomy, 19 payloads, N=30 |
| `audio_injection_*.csv` | Channel 2 results, N=30 |
| `channel3_sensor_injection_*.csv` | Channel 3 results, N=30 |

---

## Completed Work

- [x] Channel 1: Visual injection via OCR (5 models, N=100)
- [x] Channel 2: Audio injection via Whisper STT (DeepSeek, N=30)
- [x] Channel 3: LiDAR sensor context poisoning (DeepSeek, N=30)
- [x] Hybrid semantic firewall (0% ASR, 0% false positives)
- [x] Firewall bypass taxonomy (19 payloads × N=30, 52.6% bypass rate)
- [x] Real camera OCR validation (Logitech C920e, WSL2)
- [ ] WER/CER metrics for audio channel
- [ ] arXiv preprint

---

## Citation

```bibtex
@misc{dorzhiev2026ripa,
      title  = {RIPA: Robotic Injection via Pipeline Attack — Empirical Study
                of Sensory-Vector Prompt Injection on ROS 2 LLM-Controlled Robots},
      author = {Dorzhiev, Nima},
      year   = {2026},
      note   = {Preprint. https://github.com/NimaDorzh/RIPA}
}
```
