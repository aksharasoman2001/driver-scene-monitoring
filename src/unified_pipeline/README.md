# 🔗 Unified Pipeline

All three models — DMS, Lane Detection, and Object Detection — running 
together in a single real-time pipeline.

## How To Run

```bash
# On webcam
python src/unified_pipeline/unified_pipeline.py --source webcam

# On video file
python src/unified_pipeline/unified_pipeline.py --source path/to/video.mp4
```

## What Runs Per Frame

| Step | Model | Output |
|------|-------|--------|
| 1 | MediaPipe FaceMesh | 468 face landmarks |
| 2 | MobileNetV3-Small (ONNX) | Eye open / closed |
| 3 | Ultra-Fast-Lane-Detection (ONNX) | Lane overlays |
| 4 | YOLOv11-nano (ONNX) | Bounding boxes (10 classes) |

## Performance

| Hardware | FPS |
|----------|-----|
| Intel i3 CPU (no GPU) | ~2-3 FPS combined |

## Alert Logic

| Status | Condition |
|--------|-----------|
| 🟢 SAFE | EAR normal + eyes open + no yawning |
| 🟡 WARNING | Yawning detected (MAR > 0.6 for 15 frames) |
| 🔴 DANGER | Eyes closed for 20 consecutive frames |

## Models Required

Place all ONNX models in `checkpoints/`:

```
checkpoints/
├── dms_eye.onnx
├── lane_detection_final.onnx
└── yolo_detection.onnx
```
