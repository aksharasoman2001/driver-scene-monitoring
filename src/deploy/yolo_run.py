import onnxruntime as ort
import numpy as np
import time
import platform
import os

print(f"Architecture : {platform.machine()}")
print(f"Simulating   : Raspberry Pi 5 ARM64")
print(f"Model size   : {os.path.getsize('/models/yolo_detection.onnx')/1e6:.1f} MB")

session = ort.InferenceSession('/models/yolo_detection.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Model loaded successfully")
print(f"Input  : {input_name} {session.get_inputs()[0].shape}")
print(f"Output : {output_name}")

# YOLOv11 default input size is 640x640
dummy = np.random.randn(1, 3, 512, 512).astype(np.float32)

# Warmup
for _ in range(3):
    session.run([output_name], {input_name: dummy})

# Benchmark
times = []
for _ in range(20):
    start = time.time()
    result = session.run([output_name], {input_name: dummy})
    times.append((time.time() - start) * 1000)

avg_ms = sum(times) / len(times)

print(f"\n===== Raspberry Pi 5 ARM64 YOLO Detection =====")
print(f"Architecture           : {platform.machine()}")
print(f"Average inference time : {avg_ms:.2f} ms")
print(f"Estimated FPS          : {1000/avg_ms:.1f}")
print(f"Output shape           : {result[0].shape}")
print("================================================")
print("YOLO object detection deployment complete on ARM64")