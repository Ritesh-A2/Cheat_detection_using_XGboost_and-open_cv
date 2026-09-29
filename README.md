# 🖥️ Advanced Multi-Stream Exam Integrity & Cheat Detection System

[![Python 3.9+](https://shields.io)](https://python.org)
[![YOLOv11](https://shields.io)](https://github.com)
[![XGBoost](https://shields.io)](https://readthedocs.io)

An enterprise-grade, AI-powered **Classroom Cheat Detection System** designed to monitor academic integrity during examinations. By leveraging high-fidelity human pose estimation and sequential behavior classification, the system identifies suspicious activities across multiple live camera feeds in real time.

---

## 🌟 Key Technical Features

* **Multi-Stream Batch Inference (GPU Optimization):** Concurrently processes up to 4 live video streams. Instead of initializing independent, heavy object-detection models per stream, the engine aggregates frames into a single tensor batch for unified GPU inference, drastically reducing VRAM overhead and system latency.
* **Spatial/Location-Independent AI Model:** Unlike primitive models that overfit to absolute screen coordinates (pixel coordinates), this architecture converts raw keypoints into **Relative Bounding Box Space**. This structural normalization ensures the system remains robust across unseen classrooms, varying camera installation angles, and varying student seating depths (front vs. back benches).
* **Temporal Tracking & Score Smoothing:** Integrates a custom multi-object tracker with a mathematical `EventEngine`. Single-frame detection anomalies (pose noise) are mitigated through a **Rolling Score Smoothing Window**, preventing volatile false-positive flags.
* **Automated Asynchronous Email Alerting:** When a specific student's cheating index breaks the temporal accumulation threshold, the backend detaches a non-blocking thread to dispatch an instantaneous automated email notification (complete with frame metadata) to the proctor or supervisor.
* **Pro-Tier Dark Dashboard:** Features an interactive dashboard built on top of Python's UI ecosystem, rendering multi-stream bounding box grids, synchronized real-time incident logs, active candidate registries, and comprehensive session metrics exportable via CSV.

---

## 🏗️ System Workflow Architecture

```text
[ Live Video Feeds ] ──> [ Asynchronous Frame Reader Threads ] 
                                      │
                                      ▼ (Bounded Queue Buffer)
                         [ Batched GPU Inference Tensor ] 
                                      │ (YOLOv11-Pose Engine)
                                      ▼
                       [ Bounding-Box Keypoint Normalization ]
                                      │
                                      ▼ (Spatial Invariant Features)
                         [ Gradient-Boosted Classifier ] ──> (XGBoost Pipeline)
                                      │
                                      ▼ (Per-Frame Inference Matrix)
                           [ Custom Tracking Engine ] 
                                      │ (Identity Continuity)
                                      ▼
                         [ Temporal Smoothing Filter ] ──> [ Real-Time UI Logs & Smptlib Alerts ]
```

---

## 📁 Repository Directory Structure

```text
Cheat_detection_using_XGboost_and-open_cv/
│
├── 📂 Model_training/                  # Model R&D and Feature Engineering Pipeline
│   ├── dataset.csv                     # Compiled structural pose dataset
│   └── train.py                        # Localized preprocessing & XGBoost training script
│
└── 📂 project/                         # Live Real-Time Application Package
    ├── 📂 classifier/                  # Batched XGBoost predictor implementation
    ├── 📂 config/                      # Global hyper-parameters & tuning matrices (settings.py)
    ├── 📂 detection/                   # YOLOv11 pose-estimation pipeline encapsulation
    ├── 📂 events/                      # Temporal smoothing & anomaly accumulation engine
    ├── 📂 models/                      # Deep learning & tree-based binary serialized weights
    ├── 📂 ui/                          # Tkinter multithreaded logging widgets & GUI assets
    ├── app.py                          # Primary application runtime entry-point
    └── requirements.txt                # Fixed pinning of required deployment dependencies
```

---

## 🚀 Installation & Deployment Instructions

### 1. System Prerequisites
Ensure your local deployment workspace meets the following baselines:
* **Operating System:** Windows 10/11 or Ubuntu Linux
* **Python Runtime:** Python 3.9 or higher
* **Hardware Acceleration:** NVIDIA GPU with CUDA Toolkit installed (Highly Recommended)

### 2. Dependency Resolution
Clone the workspace repository, navigate to the project root directory, and run the following command to install the locked dependencies:
```bash
pip install -r project/requirements.txt
```

### 3. Model Training & Compilation (Optional)
To execute feature engineering and re-compile the tree-based classification topology over your custom dataset using bounding-box spatial adjustments:
```bash
python Model_training/train.py
```

### 4. Initializing the Application Runtime
To kick off the primary graphical console interface and initiate live batch analysis, run:
```bash
cd project
python app.py
```

---

## 🛠️ Core Technology Stack
* **Computer Vision Core:** OpenCV, Ultralytics YOLOv11-Pose Framework
* **Machine Learning Engine:** XGBoost, Scikit-Learn Data Pipeline, NumPy Core, Pandas
* **UI/UX Engineering:** Tkinter Desktop Toolkit, TTKBootstrap, CVZone Asset Engine
* **Concurrency Architecture:** Python Standard `threading` Modules & Memory Bounded `queue` Structs
* **Communication Layer:** Network `smtplib` Implementation (MIME Object Encapsulation)
