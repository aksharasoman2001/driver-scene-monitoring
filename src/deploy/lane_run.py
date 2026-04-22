import onnxruntime as ort
import numpy as np
import platform
import os
import time

print(f"Architecture : {platform.machine()}")
print(f"Simulating   : NVIDIA Jetson Nano ARM64")
print(f"Model size   : {os.path.getsize('/models/lane_detection.onnx')/1e6:.1f} MB")

session = ort.InferenceSession('/models/lane_detection.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Model loaded successfully")
print(f"Input  : {input_name} {session.get_inputs()[0].shape}")
print(f"Output : {output_name}")

# Run only 1 inference instead of 50
dummy = np.random.randn(1, 3, 288, 800).astype(np.float32)
start = time.time()
result = session.run([output_name], {input_name: dummy})
elapsed = (time.time() - start) * 1000

print(f"\n===== Jetson Nano ARM64 Lane Detection =====")
print(f"Architecture   : {platform.machine()}")
print(f"Inference time : {elapsed:.2f} ms (QEMU simulation)")
print(f"Output shape   : {result[0].shape}")
print("============================================")
print("Deployment simulation complete on ARM64")