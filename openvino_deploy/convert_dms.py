import openvino as ov
import time

print("Converting DMS ONNX to OpenVINO IR format...")

# Load ONNX model
core = ov.Core()
onnx_model = core.read_model("dms_eye.onnx")

# Convert and save
ov.save_model(onnx_model, "dms_eye.xml")

print("✓ Conversion complete!")
print("Files created:")
print("  - dms_eye.xml  (model architecture)")
print("  - dms_eye.bin  (model weights)")