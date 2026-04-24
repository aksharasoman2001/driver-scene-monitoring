import openvino as ov

print("Converting YOLO ONNX to OpenVINO IR format...")

# Load ONNX model
core = ov.Core()
onnx_model = core.read_model("yolo_detection.onnx")

# Convert and save
ov.save_model(onnx_model, "yolo_detection.xml")

print("✓ Conversion complete!")
print("Files created:")
print("  - yolo_detection.xml  (model architecture)")
print("  - yolo_detection.bin  (model weights)")