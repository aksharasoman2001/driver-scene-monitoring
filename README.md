# driver-scene-monitoring
# 🚗 Driver & Scene Monitoring System (DMS)

> Real-time driver drowsiness and distraction detection using deep learning and computer vision.
> Built as part of a 12-week Automotive AI portfolio project targeting Automotive AI Engineer roles in Europe.

---

## 📌 What This Project Does

Detects driver drowsiness and distraction in real-time using:
- 👁️ **Eye state detection** — MobileNetV3-Small (99.44% F1 score)
- 📐 **EAR** — Eye Aspect Ratio for blink detection
- 👄 **MAR** — Mouth Aspect Ratio for yawn detection
- 🗣️ **Head Pose** — Yaw/Pitch/Roll angles for distraction detection
- ⚠️ **3-level alert system** — Green (Safe) / Yellow (Warning) / Red (Danger)

---

## 🎯 Model Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 100% |
| Validation F1 | 0.9944 |
| False Negative Rate (FNR) | 0.66% |
| False Positive Rate (FPR) | 0.20% |
| Model Size | 5.93 MB |
| Total Parameters | 1,519,906 |

---

## 🧠 Model Details

- **Architecture:** MobileNetV3-Small (pretrained ImageNet)
- **Task:** Binary classification — open eye vs closed eye
- **Dataset:** [Drowsiness Detection — Kaggle](https://www.kaggle.com/datasets/kutaykutlu/drowsiness-detection)
- **Training platform:** Google Colab T4 GPU
- **Experiment tracking:** [W&B Dashboard](https://wandb.ai/aksharasoman1966-esigelec/dms-eye-detection/runs/93rqa2h9)

---

## 📁 Project Structure
```
driver-scene-monitoring/
├── src/
│   └── dms/
│       ├── ear_calculator.py     ← Eye Aspect Ratio
│       ├── mar_calculator.py     ← Mouth Aspect Ratio
│       ├── head_pose.py          ← Head Pose Estimation
│       └── dms_pipeline.py       ← Complete DMS Pipeline
├── notebooks/
│   └── 02_DMS_training.ipynb    ← Training notebook (Colab)
├── docs/
│   └── DMS_model_card.md        ← Model card (coming soon)
└── README.md
```

---

## 🚀 How To Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/driver-scene-monitoring.git
cd driver-scene-monitoring
```

**2. Install dependencies**
```bash
pip install opencv-python mediapipe torch torchvision numpy
```

**3. Download model weights**
- Download `best_model.pt` from the Releases page
- Place it in `checkpoints/` folder

**4. Run DMS pipeline**
```bash
python src/dms/dms_pipeline.py
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-blue)

---

## 📅 Project Roadmap

- [x] Week 1 — EAR + MAR + Head Pose ✅
- [x] Week 2 — Dataset + Data Pipeline ✅
- [x] Week 3 — MobileNetV3 Training (99.44% F1) ✅
- [x] Week 4 — Real-time DMS Pipeline ✅
- [ ] Week 5 — Lane Detection (BDD100K)
- [ ] Week 6 — Object Detection (YOLOv11-nano)
- [ ] Week 7 — Combined DMS + ADAS Pipeline
- [ ] Week 8 — CARLA Simulator Demo
- [ ] Week 9 — Quantization (FP32 → INT8)
- [ ] Week 10 — ONNX + TFLite + OpenVINO Export
- [ ] Week 11 — Docker ARM Deployment Simulation
- [ ] Week 12 — Documentation + Portfolio Polish

---

## 👩‍💻 Author

**Your Name**
- LinkedIn: your LinkedIn link
- Email: your email

---

*🎯 Target: Automotive AI Engineer Role in Europe · Timeline: 12 Weeks*
