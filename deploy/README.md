# 🐳 ARM64 Deployment — Docker + QEMU Simulation

All three ONNX models deployed and validated on a simulated **NVIDIA Jetson 
Nano / Raspberry Pi 5 ARM64 environment** using Docker + QEMU — without 
physical embedded hardware.

## Deployment Approach

Docker with QEMU multi-architecture emulation runs each model inside a real 
`linux/arm64` (`aarch64`) container on a standard Intel i3 laptop. This 
validates ARM64 compatibility and confirms each model loads and executes 
correctly in an embedded-style environment.

```bash
# Step 1 — Enable ARM64 emulation (one time)
docker run --privileged --rm tonistiigi/binfmt --install arm64

# Step 2 — Run model on ARM64 (example: DMS)
docker run --platform linux/arm64 --rm \
  -v /path/to/models:/models \
  arm64v8/python:3.10-slim \
  bash -c "pip install onnxruntime numpy -q && python3 /models/run_dms.py"
```

## ARM64 Deployment Results

| Model | ONNX Size | Architecture Confirmed | Output |
|-------|-----------|------------------------|--------|
| DMS — MobileNetV3-Small | 6.11 MB | `aarch64` ✅ | EYE OPEN / CLOSED |
| Lane Detection — ResNet-18 | 258.5 MB | `aarch64` ✅ | `(1, 101, 56, 4)` |
| Object Detection — YOLOv11-nano | 10.6 MB | `aarch64` ✅ | `(1, N, 85)` detections |

**Note on QEMU inference times:** Times measured inside QEMU reflect 
instruction-translation overhead on an Intel host — not real hardware 
performance. The `aarch64` confirmation in the output is the deployment 
proof, not the QEMU-inflated latency. On actual Jetson Nano or Raspberry 
Pi 5 hardware, inference would be significantly faster.

## What Was Validated

- ✅ All three ONNX models load successfully on ARM64
- ✅ `aarch64` architecture confirmed at runtime for each model
- ✅ Correct input/output tensor shapes verified per model
- ✅ End-to-end inference pipeline runs without errors on ARM Linux


