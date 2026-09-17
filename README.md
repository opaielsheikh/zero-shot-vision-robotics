# Zero-Shot Vision-Driven Robotics Simulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Simulator](https://img.shields.io/badge/simulator-PyBullet%20Physics-orange.svg)](https://pybullet.org/)
[![Model](https://img.shields.io/badge/model-TypeSafe%20Jev%20System%20One-purple.svg)](https://typesafe.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, vision-driven autonomous robotics simulation combining **PyBullet** 3D physics with **TypeSafe's Jev System One Model (`jev-latest` / `jev-1.13.0`)**.

The robotic agent makes real-time physical decisions based **strictly on raw RGB visual observations** captured by a virtual synthetic camera overlooking the workspace—operating without privileged state information (such as ground-truth world coordinates or bounding boxes).

---

## 📸 Perception-Action Trajectory

The model visualizes the workspace, performs zero-shot spatial reasoning, and drives the 7-DOF manipulator directly to the target object:

| Step 1: Perception & Alignment | Step 2: Approach Corridor Extension |
| :---: | :---: |
| ![Step 01](assets/images/step_01.jpg) | ![Step 02](assets/images/step_02.jpg) |
| **Jev Decision**: `ALIGN_BASE`<br>**Confidence**: 49.0% \| **Safety**: 67.0% | **Jev Decision**: `DESCEND_HOVER`<br>**Confidence**: 83.0% \| **Safety**: 66.0% |

| Step 3: Descent & Fine Precision | Step 4: Visual Target Acquisition |
| :---: | :---: |
| ![Step 03](assets/images/step_03.jpg) | ![Step 04](assets/images/step_04.jpg) |
| **Jev Decision**: `DESCEND_HOVER`<br>**Confidence**: 83.0% \| **Safety**: 43.0% (Near Table) | **Jev Decision**: `TARGET_ACQUIRED`<br>**Confidence**: 91.0% \| **Safety**: 74.0% |

---

## 🧠 Live Jev System One Telemetry

On every decision step, the simulation pauses physics, extracts the RGB frame, and queries TypeSafe's Jev model for structured action choices and confidence scores:

```text
┌────────────────────────────────────────────────────────────────────┐
│ 🧠 TYPE-SAFE JEV MODEL INFERENCE [jev-1.13.0]                       │
│ Request ID: req_01a0afa03994711c82812e59ce5fe7a4                   │
├────────────────────────────────────────────────────────────────────┤
│ 🎯 Decision Phase    : ALIGN_BASE                                   │
│ 📊 Alignment Conf.   : 49.0% (Noul Score)                           │
│ 🛡️ Safety Clearance  : 67.0% (Noul Score)                          │
│ ⚡ Speed Profile     : standard_approach                            │
├────────────────────────────────────────────────────────────────────┤
│ Raw Model Output:                                                  │
│   Choices: {"action_phase": "align_base", "speed_profile": "standard_approach"}│
│   Nouls  : {"alignment_confidence": 0.49, "safety_clearance": 0.67}│
└────────────────────────────────────────────────────────────────────┘
```

### Real-Time In-Simulation 3D HUD
The simulation renders a dynamic **3D floating HUD** and an **interactive targeting laser beam** directly inside the PyBullet 3D viewport:
- **Title**: `JEV [jev-1.13.0]: ALIGN_BASE`
- **Telemetry**: `Conf: 83% | Safety: 66% | Speed: standard_approach`
- **Laser Guide**: Real-time 3D vector connecting the end-effector tip to the red cube.
- **Goal Completion**: Displays `🎯 TARGET ACQUIRED (GOAL REACHED)` upon task completion.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│                   PyBullet Physics                     │
│  - Tabletop Workspace (Table URDF)                     │
│  - 7-DOF Robotic Arm (KUKA LBR iiwa)                   │
│  - Target Object (Vibrant Red Cube)                    │
└──────────────────────────┬─────────────────────────────┘
                           │ 1. Pause Physics & Render Frame
                           ▼
┌────────────────────────────────────────────────────────┐
│                  Synthetic Camera                      │
│  - View Matrix: `computeViewMatrixFromYawPitchRoll`    │
│  - Projection: `computeProjectionMatrixFOV`            │
│  - Formats: uint8 NumPy (H, W, 3) ➔ PIL ➔ Base64 JPEG   │
└──────────────────────────┬─────────────────────────────┘
                           │ 2. Multimodal State & Observation
                           ▼
┌────────────────────────────────────────────────────────┐
│             TypeSafe Jev System One Model              │
│  - Evaluates spatial context, alignment, & safety      │
│  - Returns structured Choices & Noul confidence:       │
│    { action_phase, alignment_confidence, ... }         │
└──────────────────────────┬─────────────────────────────┘
                           │ 3. Dispatch Joint Targets
                           ▼
┌────────────────────────────────────────────────────────┐
│               Synchronous Physics Stepping             │
│  - Motor Control: `setJointMotorControlArray`          │
│  - Deterministic step window (120 ticks @ 240 Hz)      │
│  - Eliminates perception-action latency                │
└────────────────────────────────────────────────────────┘
```

---

## ⏱️ Overcoming Perception-Action Latency

In standard continuous simulation loops, external API round trips (typically 200ms–800ms) introduce severe perception delay. If the simulation steps during deliberation, the robot acts on outdated observations.

### Synchronous Turn-Based Control Loop
This framework eliminates lag through step-based physics synchronization:
1. **Pause**: Simulation physics pauses while the camera frame is captured.
2. **Deliberate**: Jev processes the observation and computes action vectors.
3. **Step**: Target joint angles are dispatched, and physics advances synchronously for a controlled window ($\Delta t = 120 \text{ ticks} \times \frac{1}{240}\text{s} = 0.5\text{s}$) to reach mechanical steady-state before the next observation is captured.

---

## 📐 Virtual Camera Formulation

The synthetic camera is positioned with spherical coordinates relative to the workspace center:

$$
\mathbf{p}_{\text{target}} = \begin{bmatrix} 0.0 & 0.05 & 0.65 \end{bmatrix}^T, \quad d = 1.4\text{m}, \quad \text{yaw} = 50^\circ, \quad \text{pitch} = -35^\circ
$$

PyBullet computes the view matrix $\mathbf{V}$ and perspective projection matrix $\mathbf{P}$:
```python
view_matrix = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=[0.0, 0.05, 0.65],
    distance=1.4,
    yaw=50.0,
    pitch=-35.0,
    roll=0.0,
    upAxisIndex=2,  # Z-axis is UP
)

proj_matrix = p.computeProjectionMatrixFOV(
    fov=60.0,
    aspect=640.0 / 480.0,
    nearVal=0.1,
    farVal=3.5,
)
```

---

## 📦 Directory Layout

```
zero-shot-vision-robotics/
├── assets/
│   └── images/                   # Trajectory frames for README
│       ├── step_01.jpg
│       ├── step_02.jpg
│       ├── step_03.jpg
│       └── step_04.jpg
├── vision_robotics/              # Core Simulation Package
│   ├── __init__.py
│   ├── agent.py                  # TypeSafe Jev & multimodal policy integration
│   ├── camera.py                 # Camera matrices, RGB extraction, & Base64
│   ├── config.py                 # Simulation, camera, and robot dataclasses
│   └── environment.py            # Tabletop environment, 3D HUD, & joint control
├── .env.example                  # Template for API keys (safe to commit)
├── .gitignore                    # Strictly ignores .env and local caches
├── main.py                       # Executable perception-action loop
├── requirements.txt              # Production dependencies
├── run.sh                        # One-command runner script
└── test_simulation.py            # Automated test suite (including live Jev test)
```

---

## ⚡ Quickstart

### 1. Clone & Setup
```bash
git clone https://github.com/opaielsheikh/zero-shot-vision-robotics.git
cd zero-shot-vision-robotics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Key
Copy `.env.example` to `.env` and insert your TypeSafe API key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
TYPESAFE_API_KEY=your_typesafe_api_key_here
TYPESAFE_MODEL=jev-latest
```

### 3. Run Interactive 3D Simulation
```bash
./run.sh
```
*Launches the 3D PyBullet visualizer with real-time physics, 3D floating HUD, and targeting beam.*

### 4. Run Headless Mode
```bash
./run.sh --headless --steps 4 --output-dir captured_frames
```

### 5. Run Automated Tests
```bash
python test_simulation.py
```
*Validates camera matrices, frame formatting, motor controls, and the live Jev System One connection.*

---

## 🛡️ Security Note
Your private API keys (stored in `.env`) are **strictly ignored** by `.gitignore` and are never committed or pushed to GitHub. Always use `.env.example` as a public template.

---

## 📜 License
MIT License. Feel free to use and extend for robotics research and vision-language development.
