# dms_pipeline.py
# Week 4 — Complete DMS Pipeline
# Combines EAR + MAR + Head Pose + MobileNetV3 Model

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from collections import deque
import os
import urllib.request

# ── CONFIG ────────────────────────────────────
MODEL_PATH = r"D:\personal_project\DMS\dms_eye_fp16.pt"
IMG_SIZE   = 224
DEVICE     = torch.device('cpu')

# ── MEDIAPIPE SETUP ───────────────────────────
BaseOptions        = mp.tasks.BaseOptions
FaceLandmarker     = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading MediaPipe face model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )
    print("Download complete! ✅")

# ── LOAD MOBILENETV3 MODEL ────────────────────
print("Loading MobileNetV3 model...")
dms_model = models.mobilenet_v3_small(weights=None)
dms_model.classifier[3] = nn.Linear(dms_model.classifier[3].in_features, 2)
dms_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
dms_model.eval()
print("✅ Model loaded!")

# ── TRANSFORMS ───────────────────────────────
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── LANDMARK INDICES ──────────────────────────
LEFT_EYE       = [33,  160, 158, 133, 153, 144]
RIGHT_EYE      = [362, 385, 387, 263, 373, 380]
MOUTH_LEFT     = 61
MOUTH_RIGHT    = 291
MOUTH_TOP_1, MOUTH_TOP_2, MOUTH_TOP_3 = 39, 0, 269
MOUTH_BOT_1, MOUTH_BOT_2, MOUTH_BOT_3 = 84, 14, 314
FACE_POINTS_IDX = [1, 152, 33, 263, 61, 291]

# ── 3D REFERENCE POINTS FOR HEAD POSE ─────────
MODEL_3D_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,   -63.6,  -12.5),
    (-43.3,  32.7,  -26.0),
    (43.3,   32.7,  -26.0),
    (-28.9, -28.9,  -24.1),
    (28.9,  -28.9,  -24.1),
], dtype=np.float64)

# ── THRESHOLDS ────────────────────────────────
EAR_THRESHOLD        = 0.25
MAR_THRESHOLD        = 0.5
CLOSED_FRAMES_LIMIT  = 20
YAWN_FRAMES_LIMIT    = 15
YAW_THRESHOLD        = 30
PITCH_THRESHOLD      = 20
ROLL_THRESHOLD       = 20

# ──────────────────────────────────────────────
# EAR FUNCTIONS (from your ear_calculator.py)
# ──────────────────────────────────────────────
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_indices, frame_w, frame_h):
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append((lm.x * frame_w, lm.y * frame_h))
    p1, p2, p3, p4, p5, p6 = points
    ear = (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4))
    return ear, points

def draw_eye_points(frame, points, color):
    for (x, y) in points:
        cv2.circle(frame, (int(x), int(y)), 2, color, -1)

# ──────────────────────────────────────────────
# MAR FUNCTIONS (from your mar_calculator.py)
# ──────────────────────────────────────────────
def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)

def calculate_mar(landmarks, frame_w, frame_h):
    left  = get_point(landmarks, MOUTH_LEFT,  frame_w, frame_h)
    right = get_point(landmarks, MOUTH_RIGHT, frame_w, frame_h)
    top1  = get_point(landmarks, MOUTH_TOP_1, frame_w, frame_h)
    top2  = get_point(landmarks, MOUTH_TOP_2, frame_w, frame_h)
    top3  = get_point(landmarks, MOUTH_TOP_3, frame_w, frame_h)
    bot1  = get_point(landmarks, MOUTH_BOT_1, frame_w, frame_h)
    bot2  = get_point(landmarks, MOUTH_BOT_2, frame_w, frame_h)
    bot3  = get_point(landmarks, MOUTH_BOT_3, frame_w, frame_h)
    vertical_1 = euclidean(top1, bot1)
    vertical_2 = euclidean(top2, bot2)
    vertical_3 = euclidean(top3, bot3)
    horizontal = euclidean(left, right)
    mar = (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)
    return mar

# ──────────────────────────────────────────────
# HEAD POSE FUNCTIONS (from your head_pose.py)
# ──────────────────────────────────────────────
def calculate_head_pose(landmarks, w, h):
    focal_length  = w
    camera_matrix = np.array([
        [focal_length, 0,            w / 2],
        [0,            focal_length, h / 2],
        [0,            0,            1    ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_2d_points = []
    for idx in FACE_POINTS_IDX:
        lm = landmarks[idx]
        image_2d_points.append((lm.x * w, lm.y * h))
    image_2d_points = np.array(image_2d_points, dtype=np.float64)

    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_3D_POINTS, image_2d_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0, None, None, None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    pitch, yaw, roll = angles[0], angles[1], angles[2]

    if pitch > 90:  pitch = pitch - 180
    elif pitch < -90: pitch = pitch + 180
    if yaw > 90:    yaw = yaw - 180
    elif yaw < -90: yaw = yaw + 180

    return yaw, pitch, roll, rotation_vec, translation_vec, camera_matrix

def draw_axes(frame, nose_point, rotation_vec, translation_vec, camera_matrix):
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    axis_points = np.float32([[50, 0, 0], [0, 50, 0], [0, 0, 50]])
    imgpts, _   = cv2.projectPoints(axis_points, rotation_vec, translation_vec, camera_matrix, dist_coeffs)
    nose = tuple(nose_point.astype(int))
    cv2.arrowedLine(frame, nose, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 2)
    cv2.arrowedLine(frame, nose, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 2)
    cv2.arrowedLine(frame, nose, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 2)

# ──────────────────────────────────────────────
# MODEL PREDICTION FUNCTION
# ──────────────────────────────────────────────
def get_eye_crop(frame, landmarks, eye_indices, w, h, padding=10):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    x1  = max(0, min(p[0] for p in pts) - padding)
    y1  = max(0, min(p[1] for p in pts) - padding)
    x2  = min(w, max(p[0] for p in pts) + padding)
    y2  = min(h, max(p[1] for p in pts) + padding)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None

def predict_eye(crop):
    if crop is None:
        return "open"
    try:
        img_tensor = transform(crop).unsqueeze(0)
        with torch.no_grad():
            output = dms_model(img_tensor)
            pred   = torch.argmax(torch.softmax(output, dim=1), dim=1).item()
        return "open" if pred == 0 else "closed"
    except:
        return "open"

# ──────────────────────────────────────────────
# STATE TRACKING
# ──────────────────────────────────────────────
closed_frame_count = 0
yawn_frame_count   = 0
yawn_alert_active  = False
yawn_counter       = 0
PERCLOS_WINDOW     = deque(maxlen=900)

# ── MEDIAPIPE OPTIONS ─────────────────────────
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    running_mode=VisionRunningMode.IMAGE
)

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("✅ Webcam started! Press Q to quit.\n")

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result    = landmarker.detect(mp_image)

        alert_level = 0
        status_text = "SAFE"

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # ── EAR ──────────────────────────────
            left_ear,  left_pts  = calculate_ear(landmarks, LEFT_EYE,  w, h)
            right_ear, right_pts = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            draw_eye_points(frame, left_pts,  (0, 255, 0))
            draw_eye_points(frame, right_pts, (255, 0, 0))

            # ── MAR ──────────────────────────────
            mar = calculate_mar(landmarks, w, h)

            # ── HEAD POSE ─────────────────────────
            yaw, pitch, roll, rot_vec, trans_vec, cam_mat = calculate_head_pose(landmarks, w, h)
            if rot_vec is not None:
                nose_2d = np.array([landmarks[1].x * w, landmarks[1].y * h])
                draw_axes(frame, nose_2d, rot_vec, trans_vec, cam_mat)

            # ── MODEL PREDICTION ──────────────────
            left_crop  = get_eye_crop(frame, landmarks, LEFT_EYE,  w, h)
            right_crop = get_eye_crop(frame, landmarks, RIGHT_EYE, w, h)
            left_pred  = predict_eye(left_crop)
            right_pred = predict_eye(right_crop)
            eye_state  = "CLOSED" if (left_pred == "closed" and right_pred == "closed") else "OPEN"

            # ── PERCLOS ───────────────────────────
            PERCLOS_WINDOW.append(1 if eye_state == "CLOSED" else 0)
            perclos = (sum(PERCLOS_WINDOW) / len(PERCLOS_WINDOW)) * 100

            # ── DROWSINESS LOGIC ──────────────────
            if avg_ear < EAR_THRESHOLD:
                closed_frame_count += 1
            else:
                closed_frame_count = 0

            print(f"closed_frame_count: {closed_frame_count}")

            if closed_frame_count >= CLOSED_FRAMES_LIMIT or perclos > 15:
                alert_level = 2
                status_text = "DROWSINESS DETECTED!"
            elif avg_ear < EAR_THRESHOLD:
                alert_level = 1
                status_text = "EYES CLOSING..."

            # ── YAWN LOGIC ────────────────────────
            if mar >= MAR_THRESHOLD:
                yawn_frame_count += 1
            else:
                if yawn_alert_active:
                    yawn_counter += 1
                yawn_frame_count  = 0
                yawn_alert_active = False

            if yawn_frame_count >= YAWN_FRAMES_LIMIT:
                yawn_alert_active = True
                alert_level       = max(alert_level, 1)
                status_text       = f"YAWNING! Total: {yawn_counter + 1}"

            # ── HEAD POSE ALERT ───────────────────
            if abs(yaw) > YAW_THRESHOLD:
                alert_level = max(alert_level, 1)
                status_text = "DISTRACTION! Looking Away"
            if abs(pitch) > PITCH_THRESHOLD:
                alert_level = max(alert_level, 1)
                status_text = "HEAD DOWN!"

            # ── HUD OVERLAY ───────────────────────
            cv2.putText(frame, f"EAR: {avg_ear:.3f}",      (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cv2.putText(frame, f"MAR: {mar:.3f}",          (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cv2.putText(frame, f"EYE: {eye_state}",        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cv2.putText(frame, f"YAWNS: {yawn_counter}",   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cv2.putText(frame, f"YAW: {yaw:+.1f}",         (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cv2.putText(frame, f"PITCH: {pitch:+.1f}",     (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        else:
            cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ── BORDER COLOR ──────────────────────────
        border_color = (0, 255, 0) if alert_level == 0 else \
                       (0, 255, 255) if alert_level == 1 else \
                       (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, 8)
        cv2.putText(frame, status_text, (w//2 - 150, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, border_color, 2)

        cv2.imshow("DMS - Driver Monitoring System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("DMS stopped. ✅")