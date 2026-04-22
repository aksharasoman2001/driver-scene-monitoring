# Model Card — DMS Eye Classifier

## Model Details

| Field | Value |
|-------|-------|
| Architecture | MobileNetV3-Small |
| Task | Binary classification — eye open / closed |
| Framework | PyTorch → ONNX |
| Input | Eye crop · 224×224 · RGB |
| Output | [p_open, p_closed] |
| Parameters | 1,519,906 |
| Model size (FP32) | 5.93 MB |
| Model size (ONNX) | 6.11 MB |

## Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 100% |
| Validation F1 | 0.9944 |
| False Negative Rate | 0.66% |
| False Positive Rate | 0.20% |

## Training

- Dataset: Kaggle Drowsiness Detection
- Optimizer: Adam
- Experiment tracking: [W&B Dashboard](https://wandb.ai/aksharasoman1966-esigelec/dms-eye-detection/runs/93rqa2h9)

## Pipeline Role

This model is the second stage of a two-stage DMS pipeline:

1. **MediaPipe FaceMesh** → extracts eye landmarks → computes EAR
2. **MobileNetV3-Small (this model)** → visual confirmation on eye crop

## ARM64 Deployment
