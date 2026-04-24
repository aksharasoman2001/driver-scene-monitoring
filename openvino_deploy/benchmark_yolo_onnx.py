import onnxruntime as ort
import numpy as np
import time

print("=" * 50)
print("YOLO ONNX Benchmark on Intel CPU")
print("=" * 50)

# Load ONNX model
session = ort.InferenceSession("yolo_detection.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Input name: {input_name}")
print(f"Device: CPU (Intel i3)")

# Create dummy input
dummy = np.random.randn(1, 3, 512, 512).astype(np.float32)

# Warmup
print("\nWarming up...")
for _ in range(3):
    session.run([output_name], {input_name: dummy})

# Benchmark
print("Running 50 inferences...")
times = []
for _ in range(50):
    start = time.perf_counter()
    session.run([output_name], {input_name: dummy})
    times.append((time.perf_counter() - start) * 1000)

avg_ms = sum(times) / len(times)
fps = 1000 / avg_ms

print("\n" + "=" * 50)
print(f"Average inference time : {avg_ms:.2f} ms")
print(f"FPS                    : {fps:.1f}")
print(f"Min time               : {min(times):.2f} ms")
print(f"Max time               : {max(times):.2f} ms")
print("=" * 50)
print("YOLO ONNX benchmark complete!")