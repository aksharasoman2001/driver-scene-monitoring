# 🛣️ Lane Detection Module

**Architecture:** Ultra-Fast-Lane-Detection V1 (ResNet-18 backbone)  
**Dataset:** TuSimple  
**Final IoU:** 0.857  
**Parameters:** 64.6 million  

## Model Optimization

| Stage | Size | Reduction |
|-------|------|-----------|
| Original checkpoint (weights + optimizer state) | 775.6 MB | — |
| After removing optimizer state | 258.6 MB | 66.6% ↓ |
| After FP16 conversion | 129.3 MB | 83.3% ↓ |
| ONNX export | ✅ Complete | — |

## ARM64 Deployment

Validated on ARM64 via Docker QEMU simulation:
Architecture   : aarch64 ✅
Model size     : 258.5 MB
Inference time : 168311 ms (QEMU overhead — not real hardware speed)
Output shape   : (1, 101, 56, 4) ✅


## How It Works

Ultra-Fast-Lane-Detection treats lane detection as a **row-wise classification 
problem** rather than pixel segmentation. For each row anchor and each lane, 
the model predicts which column grid cell the lane passes through — making it 
much faster than segmentation-based approaches.

Output tensor shape: `(1, 101, 56, 4)` — 101 grid columns × 56 row anchors × 4 lanes.
