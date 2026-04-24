import openvino as ov
import numpy as np
import time

print("=" * 50)
print("DMS OpenVINO Benchmark on Intel CPU")
print("=" * 50)

# Load OpenVINO model
core = ov.Core()
model = core.read_model("dms_eye.xml")
compiled = core.compile_model(model, "CPU")

# Get input details
input_layer = compiled.input(0)
print(f"Input shape: {input_layer.shape}")
print(f"Device: CPU (Intel i3)")

# Create dummy input
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Warmup
print("\nWarming up...")
for _ in range(5):
    compiled(dummy)

# Benchmark
print("Running 100 inferences...")
times = []
for _ in range(100):
    start = time.perf_counter()
    compiled(dummy)
    times.append((time.perf_counter() - start) * 1000)

avg_ms = sum(times) / len(times)
fps = 1000 / avg_ms

print("\n" + "=" * 50)
print(f"Average inference time : {avg_ms:.2f} ms")
print(f"FPS                    : {fps:.1f}")
print(f"Min time               : {min(times):.2f} ms")
print(f"Max time               : {max(times):.2f} ms")
print("=" * 50)
print("OpenVINO benchmark complete!")