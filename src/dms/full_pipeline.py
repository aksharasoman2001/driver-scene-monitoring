# full_pipeline.py
# Week 7 — Unified DMS + Lane Detection + Object Detection Pipeline
# Runs on driving video — no GPU required!

import cv2
import torch
import numpy as np
import mediapipe as mp
import sys
import os

# ── Add Lane Detection repo to path ──────────
sys.path.insert(0, r"D:\personal_project\Ultra-Fast-Lane-Detection")

from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# ── Model Paths ───────────────────────────────
DMS_MODEL_PATH  = r"D:\personal_project\DMS\best_model.pt"
LANE_MODEL_PATH = r"D:\personal_project\lane_detection\best_lane_model.pth"
YOLO_MODEL_PATH = r"D:\personal_project\object_detctionn_training_batch2\best.pt"
VIDEO_PATH      = r"D:\personal_project\dashcamvideo.mp4"

# ── Device ────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ════════════════════════════════════════════════
# 1. LOAD YOLO MODEL
# ════════════════════════════════════════════════
print("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("YOLO ready! ✅")

# ════════════════════════════════════════════════
# 2. LOAD DMS MODEL (MobileNetV3)
# ════════════════════════════════════════════════
print("Loading DMS eye model...")
import torchvision.models as models

dms_model = models.mobilenet_v3_small(weights=None)
dms_model.classifier[3] = torch.nn.Linear(
    dms_model.classifier[3].in_features, 2
)
dms_model.load_state_dict(torch.load(DMS_MODEL_PATH, map_location=device))
dms_model.eval().to(device)
print("DMS model ready! ✅")

# ── DMS image transform ───────────────────────
dms_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ════════════════════════════════════════════════
# 3. LOAD LANE MODEL
# ════════════════════════════════════════════════
print("Loading Lane Detection model...")
from model.model import parsingNet

lane_model = parsingNet(
    pretrained=False,
    backbone='18',
    cls_dim=(101, 56, 4),
    use_aux=False
).to(device)

state = torch.load(LANE_MODEL_PATH, map_location=device)
compatible = {k: v for k, v in state['model'].items() if k in lane_model.state_dict()}
lane_model.load_state_dict(compatible, strict=False)
lane_model.eval()
print("Lane model ready! ✅")

# ── Lane detection settings ───────────────────
ROW_ANCHOR = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100,
              104, 108, 112, 116, 120, 124, 128, 132, 136,
              140, 144, 148, 152, 156, 160, 164, 168, 172,
              176, 180, 184, 188, 192, 196, 200, 204, 208,
              212, 216, 220, 224, 228, 232, 236, 240, 244,
              248, 252, 256, 260, 264, 268, 272, 276, 280, 287]

lane_transform = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

LANE_COLORS = [
    (0,   255, 255),   # yellow — left-outer
    (0,   255, 0),     # green  — left-inner
    (0,   0,   255),   # red    — right-inner
    (255, 0,   0),     # blue   — right-outer
]

# ════════════════════════════════════════════════
# 4. MEDIAPIPE FACE SETUP
# ════════════════════════════════════════════════
print("Loading MediaPipe...")
BaseOptions       = mp.tasks.BaseOptions
FaceLandmarker    = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOpt = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

import urllib.request
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading MediaPipe model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )

mp_options = FaceLandmarkerOpt(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    num_faces=1,
    running_mode=VisionRunningMode.IMAGE
)
print("MediaPipe ready! ✅")

# ── EAR/MAR landmark indices ──────────────────
LEFT_EYE   = [33,  160, 158, 133, 153, 144]
RIGHT_EYE  = [362, 385, 387, 263, 373, 380]
MOUTH_IDX  = [61, 291, 37, 0, 267, 84, 14, 314]
FACE_PTS   = [1, 152, 33, 263, 61, 291]
MODEL_3D   = np.array([
    (0.0,    0.0,    0.0),
    (0.0,   -63.6,  -12.5),
    (-43.3,  32.7,  -26.0),
    (43.3,   32.7,  -26.0),
    (-28.9, -28.9,  -24.1),
    (28.9,  -28.9,  -24.1),
], dtype=np.float64)

# ── EAR/MAR helpers ───────────────────────────
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_pt(lms, idx, w, h):
    return (lms[idx].x * w, lms[idx].y * h)

def calc_ear(lms, eye, w, h):
    pts = [get_pt(lms, i, w, h) for i in eye]
    p1,p2,p3,p4,p5,p6 = pts
    return (euclidean(p2,p6) + euclidean(p3,p5)) / (2.0 * euclidean(p1,p4))

def calc_mar(lms, w, h):
    left  = get_pt(lms, 61,  w, h)
    right = get_pt(lms, 291, w, h)
    top1  = get_pt(lms, 37,  w, h)
    top2  = get_pt(lms, 0,   w, h)
    top3  = get_pt(lms, 267, w, h)
    bot1  = get_pt(lms, 84,  w, h)
    bot2  = get_pt(lms, 14,  w, h)
    bot3  = get_pt(lms, 314, w, h)
    v1 = euclidean(top1, bot1)
    v2 = euclidean(top2, bot2)
    v3 = euclidean(top3, bot3)
    return (v1+v2+v3) / (3.0 * euclidean(left, right))

# ── Drowsiness state ──────────────────────────
ear_closed_frames = 0
yawn_frames       = 0
EAR_THRESH        = 0.25
MAR_THRESH        = 0.5
CLOSED_LIMIT      = 20
YAWN_LIMIT        = 15

# ════════════════════════════════════════════════
# 5. LANE DETECTION FUNCTION
# ════════════════════════════════════════════════
def detect_lanes(frame, model, device):
    h, w = frame.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor  = lane_transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)

    out = out[0].data.cpu().numpy()
    out = out[:, ::-1, :]
    prob = scipy_softmax(out, axis=0)
    # Use the actual shape of the probability map to define idx
    idx  = np.arange(prob.shape[0] - 1) 
    loc  = np.sum(prob[:-1] * idx.reshape(-1,1,1), axis=0)
    loc[prob[-1] > 0.8] = 0
    out  = loc

    col_sample     = np.linspace(0, 800-1, 101)
    col_sample_w   = col_sample[1] - col_sample[0]

    for i in range(4):
        if np.sum(out[:, i] != 0) > 2:
            for k in range(out.shape[0]-1, -1, -1):
                if out[k, i] > 0:
                    px = int(out[k, i] * col_sample_w * w / 800) - 1
                    py = int(h * (ROW_ANCHOR[k] / 288)) - 1
                    cv2.circle(frame, (px, py), 5, LANE_COLORS[i], -1)
    return frame

# scipy softmax helper
from scipy.special import softmax as scipy_softmax

# ════════════════════════════════════════════════
# 6. MAIN PIPELINE LOOP
# ════════════════════════════════════════════════
print("\nStarting pipeline... Press Q to quit\n")
cap = cv2.VideoCapture(VIDEO_PATH)

with FaceLandmarker.create_from_options(mp_options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        dist_coeffs   = np.zeros((4,1))
        focal_length  = w
        camera_matrix = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ], dtype=np.float64)

        alerts = []

        # ── A) YOLO Object Detection ──────────
        yolo_results = yolo_model(frame, verbose=False)[0]
        for box in yolo_results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = f"{yolo_results.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,165,0), 2)
            cv2.putText(frame, label, (x1, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,165,0), 1)

        # ── B) Lane Detection ─────────────────
        frame = detect_lanes(frame, lane_model, device)

        # ── C) DMS — MediaPipe + EAR/MAR/Pose ─
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        if result.face_landmarks:
            lms = result.face_landmarks[0]

            # EAR
            left_ear  = calc_ear(lms, LEFT_EYE,  w, h)
            right_ear = calc_ear(lms, RIGHT_EYE, w, h)
            avg_ear   = (left_ear + right_ear) / 2.0

            # MAR
            mar = calc_mar(lms, w, h)

            # Head pose
            pts2d = np.array([get_pt(lms, i, w, h) for i in FACE_PTS], dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(MODEL_3D, pts2d, camera_matrix, dist_coeffs)
            if ok:
                rmat, _ = cv2.Rodrigues(rvec)
                angles, *_ = cv2.RQDecomp3x3(rmat)
                pitch, yaw, roll = angles
                if pitch > 90:  pitch -= 180
                if pitch < -90: pitch += 180
                if yaw   > 90:  yaw   -= 180
                if yaw   < -90: yaw   += 180
            else:
                pitch = yaw = roll = 0

            # Drowsiness logic
            if avg_ear < EAR_THRESH:
                ear_closed_frames += 1
            else:
                ear_closed_frames = 0

            if mar > MAR_THRESH:
                yawn_frames += 1
            else:
                yawn_frames = 0

            if ear_closed_frames >= CLOSED_LIMIT:
                alerts.append("DROWSY — Eyes Closed!")
            if yawn_frames >= YAWN_LIMIT:
                alerts.append("YAWNING Detected!")
            if abs(yaw) > 30:
                alerts.append("DISTRACTION — Looking Away!")
            if abs(pitch) > 20:
                alerts.append("HEAD DOWN!")

            # ── DMS HUD ───────────────────────
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Yaw: {yaw:+.1f} Pitch: {pitch:+.1f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ── D) Alerts + Border ────────────────
        if alerts:
            cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 6)
            for i, alert in enumerate(alerts):
                cv2.putText(frame, f"!! {alert}", (10, 120 + i*35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            cv2.rectangle(frame, (0,0), (w,h), (0,255,0), 3)
            cv2.putText(frame, "All Systems Normal", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("DMS + Lane + Object Detection Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Pipeline complete! ✅")