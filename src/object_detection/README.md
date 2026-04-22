# 🚙 Object Detection Module (ADAS)

**Architecture:** YOLOv11-nano (Ultralytics)  
**Dataset:** BDD100K  
**Classes:** 10  
**mAP@50:** 0.381  
**ONNX size:** 10.6 MB  
**Parameters:** ~2.6 million  

## Classes

`car · bus · truck · person · rider · bike · motor · traffic light · traffic sign · train`

## Training

| Parameter | Value |
|-----------|-------|
| Epochs | 25 |
| Training time | ~5.5 hours (Kaggle T4) |
| Image size | 640×640 |
| mAP@50 | 0.381 |
| mAP@50-95 | 0.208 |
| Precision | 0.615 |
| Recall | 0.357 |

## Design Choice — Nano for Edge

YOLOv11-nano was chosen deliberately over larger variants (s/m/l) to prioritize 
low-latency edge inference. BDD100K is one of the most challenging autonomous 
driving benchmarks — extreme class imbalance, diverse weather and lighting, 
and small objects.

## ARM64 Deployment

Validated on ARM64 via Docker QEMU simulation:

```
Architecture : aarch64 ✅
Model size   : 10.6 MB
Inference    : confirmed on linux/arm64 ✅
```
