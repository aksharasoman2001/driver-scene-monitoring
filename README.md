🚗 Driver & Scene Monitoring System (DMS + ADAS)
> End-to-end real-time automotive perception pipeline combining driver monitoring, lane detection, and object detection — optimized for edge deployment on embedded hardware.
>
> A portfolio project targeting **Embedded AI / Automotive AI Engineer** roles in Europe.
---
🎬 Live Demo
![Unified Pipeline Demo](results/demo.gif)
All three models running together on a driving video — DMS alerts + lane detection + object detection — on an Intel i3 CPU (~2-3 FPS).
---
✅ Project Status — Complete
Module	Model	Dataset	Status
👁️ Driver Monitoring (DMS)	MobileNetV3-Small + MediaPipe	Kaggle Drowsiness	✅ Trained · Optimized · Deployed
🛣️ Lane Detection	Ultra-Fast-Lane-Detection V1 (ResNet-18)	TuSimple	✅ Trained · FP16 · ONNX
🚙 Object Detection (ADAS)	YOLOv11-nano	BDD100K (10 classes)	✅ Trained · ONNX
🔗 Unified Pipeline	All 3 merged	Video + Webcam	✅ Tested on CPU
---
🎯 Results Summary
Module 1 — Driver Monitoring System
Metric	Score
Test Accuracy	100%
Validation F1	0.9944
False Negative Rate	0.66%
False Positive Rate	0.20%
Model Size (FP32)	5.93 MB
Model Size (ONNX)	6.11 MB
Parameters	1,519,906
📊 Experiment tracking: W&B Dashboard
---
Module 2 — Lane Detection
Metric	Value
Architecture	Ultra-Fast-Lane-Detection V1
Backbone	ResNet-18
Dataset	TuSimple
Parameters	64.6 million
Final IoU	0.857
Optimization journey:
Stage	Size	Reduction
Original checkpoint (weights + optimizer state)	775.6 MB	—
After removing optimizer state	258.6 MB	66.6% ↓
After FP16 conversion	129.3 MB	83.3% ↓
ONNX export	✅ Complete	—
---
Module 3 — Object Detection (ADAS)
Chose YOLOv11-nano specifically for embedded deployment — the smallest model in the YOLOv11 family (~2.6M parameters), optimized for low-latency edge inference over peak accuracy.
Metric	Value
Architecture	YOLOv11-nano (Ultralytics)
Dataset	BDD100K
Classes	10 (car, bus, truck, person, rider, bike, motor, traffic light, traffic sign, train)
Training epochs	25
Training time	~5.5 hours (Colab T4)
Image size	640×640
mAP@50	0.381
mAP@50-95	0.208
Precision	0.615
Recall	0.357
F1 peak	0.41 (at confidence 0.198)
Export format	ONNX (opset 12, simplified)
Per-class accuracy (normalized confusion matrix diagonal):
Class	Accuracy	Notes
Car	0.66	Strongest — most frequent class in BDD100K
Traffic light	0.49	Reliable for ADAS
Traffic sign	0.41	Reliable for ADAS
Truck	0.38	Moderate
Person	0.36	Critical ADAS class
Bus	0.36	Moderate
Bike	0.21	Rare class, harder
Rider	0.20	Often confused with person
Motor	0.16	Very rare in BDD100K
Train	0.00	Almost absent in dataset
Design choice — accuracy vs edge deployability:
BDD100K is one of the most challenging autonomous driving benchmarks due to extreme class imbalance, diverse weather/lighting conditions, and small object sizes. YOLOv11-nano was chosen deliberately over larger variants (YOLOv11-s/m/l) to prioritize low-latency edge inference — a realistic tradeoff for embedded automotive deployment rather than peak benchmark accuracy. These metrics align with the expected performance of YOLOv11-nano on BDD100K.
---
Module 4 — Unified Pipeline
All three models integrated and tested together:
Test	Result
Input	Driving video + live webcam
Hardware	Intel i3 CPU (no GPU)
Runtime	ONNX Runtime (CPU)
Performance	~2-3 FPS combined
Outputs	DMS alert + lane overlay + object bounding boxes
---
🧠 Architecture Overview
Module 1 — DMS (Two-stage pipeline)
```
Camera → MediaPipe FaceMesh (468 landmarks)
       ├── EAR (Eye Aspect Ratio) → blink detection
       ├── MAR (Mouth Aspect Ratio) → yawning detection
       ├── Head Pose (Yaw / Pitch / Roll) → distraction
       └── Eye crop → MobileNetV3-Small → open / closed
                                            ↓
                        Alert Logic (🟢 Safe / 🟡 Warning / 🔴 Danger)
```
Why two-stage? MediaPipe handles facial geometry (fast, no training needed), while MobileNetV3 gives a second-opinion visual confirmation on eye state — more robust than EAR alone.
Module 2 — Lane Detection
Ultra-Fast-Lane-Detection treats lane detection as a row-wise classification problem rather than segmentation — dramatically faster for real-time edge deployment. Trained on TuSimple highway benchmark.
Module 3 — Object Detection
YOLOv11-nano is the smallest model in the YOLOv11 family (~2.6M parameters), ideal for embedded deployment. Trained on BDD100K for diverse road scenes (day/night/weather) with 10 ADAS-relevant classes.
Module 4 — Unified Pipeline
All three ONNX models run in parallel per video frame, outputs composited into a single annotated frame. Runs on standard laptop CPU without GPU.
---
📁 Project Structure
```
driver-scene-monitoring/
├── src/
│   ├── dms/                          ← Driver Monitoring
│   │   ├── __init__.py
│   │   ├── ear_calculator.py         ← Eye Aspect Ratio
│   │   ├── mar_calculator.py         ← Mouth Aspect Ratio
│   │   ├── head_pose.py              ← Head Pose Estimation
│   │   └── dms_pipeline.py           ← Full DMS pipeline
│   ├── lane_detection/               ← Lane Detection
│   │   ├── __init__.py
│   │   └── README.md
│   ├── object_detection/             ← Object Detection
│   │   ├── __init__.py
│   │   └── README.md
│   └── unified_pipeline/             ← Merged DMS + Lane + YOLO
│       ├── __init__.py
│       ├── unified_pipeline.py
│       └── README.md
├── notebooks/
│   ├── 01_DMS_training.ipynb
│   ├── 02_DMS_onnx_export.ipynb
│   ├── 03_lane_training.ipynb
│   ├── 04_lane_optimization.ipynb
│   └── 05_yolo_training.ipynb
├── results/
│   ├── demo.gif                      ← Unified pipeline demo
│   └── metrics/                      ← Loss curves, confusion matrices
├── docs/
│   └── DMS_model_card.md
├── checkpoints/                      ← Weights via GitHub Releases
├── requirements.txt
├── .gitignore
└── README.md
```
---
🚀 How To Run
1. Clone the repo
```bash
git clone https://github.com/aksharasoman2001/driver-scene-monitoring.git
cd driver-scene-monitoring
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Download model weights
Download all ONNX models from the Releases page and place them in `checkpoints/`.
4. Run individual modules
```bash
python src/dms/dms_pipeline.py                    # DMS only (webcam)
python src/unified_pipeline/unified_pipeline.py   # All 3 merged (video/webcam)
```
---
🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-blue)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11-yellow)
Training: PyTorch · Ultralytics · W&B · Google Colab (T4 GPU)  
Optimization: ONNX · FP16 quantization · ONNX Simplifier  
Deployment runtime: ONNX Runtime (CPU) · OpenCV · MediaPipe  
Hardware tested: Intel i3 laptop (CPU-only inference)
---
🔗 Related Project
> This project builds on my embedded AI research at CNRS:
>
> 📦 [Underwater Fish Detection — Embedded AI on Raspberry Pi 5](https://github.com/aksharasoman2001/underwater-fish-detection-embedded-ai)  
> Achieved mAP@0.5 = 0.98 · INT8 quantization · 3.26 FPS on CPU-only Raspberry Pi 5
---
👩‍💻 Author
Akshara Soman — M.Sc. Automotive Embedded Systems, ESIGELEC France
🔗 LinkedIn: linkedin.com/in/akshara-soman
📧 Email: somanakshara7@gmail.com
💼 Open to: Embedded AI · Computer Vision · ADAS Engineer roles in Europe
---
📜 License
MIT License — see LICENSE file for details.
