import onnxruntime as ort
import numpy as np
import time
import platform

print(f"Architecture : {platform.machine()}")
print(f"Simulating   : NVIDIA Jetson Nano ARM64")

session = ort.InferenceSession('/models/dms_eye.onnx')
print("Model loaded successfully\n")

# Print real input/output names
print("Input name  :", session.get_inputs()[0].name)
print("Output name :", session.get_outputs()[0].name)

dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Warmup
for _ in range(5):
    session.run([output_name], {input_name: dummy})

# Benchmark
times = []
for _ in range(50):
    start = time.time()
    result = session.run([output_name], {input_name: dummy})
    times.append((time.time() - start) * 1000)

avg_ms = sum(times) / len(times)
pred = np.argmax(result[0])

print("\n===== Jetson Nano ARM64 Simulation Results =====")
print(f"Architecture           : {platform.machine()}")
print(f"Average inference time : {avg_ms:.2f} ms")
print(f"Estimated FPS          : {1000/avg_ms:.1f}")
print(f"Prediction             : {'EYE OPEN' if pred == 0 else 'EYE CLOSED'}")
print("================================================")
print("Deployment simulation complete on ARM64")
