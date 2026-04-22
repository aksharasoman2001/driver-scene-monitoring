# combined_pipeline.py
# DMS + Lane Detection + Object Detection — all running together
# With BDD100K class names, frame skipping for speed, improved lane detection

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from collections import deque
import os
import urllib.request

# ── CONFIG ────────────────────────────────────
DMS_ONNX_PATH   = r"D:\\dms_deploy\\dms_eye.onnx"
LANE_ONNX_PATH  = r"D:\\dms_deploy\\lane_detection.onnx"
YOLO_ONNX_PATH  = r"D:\\dms_deploy\\yolo_detection.onnx"
DRIVING_VIDEO   = r"D:\\dms_deploy\\LaneDetectionTestvideo.mp4"
OUTPUT_VIDEO    = r"D:\\dms_deploy\\demo_output.mp4"
IMG_SIZE        = 224

# ── BDD100K CLASSES ───────────────────────────
BDD_CLASSES = [
    'person', 'rider', 'car', 'bus', 'truck',
    'bike', 'motor', 'traffic light', 'traffic sign', 'train'
]

CLASS_COLORS = [
    (255, 0, 0),      # person
    (255, 100, 0),    # rider
    (0, 255, 0),      # car
    (0, 165, 255),    # bus
    (0, 200, 200),    # truck
    (255, 255, 0),    # bike
    (255, 0, 255),    # motor
    (0, 255, 255),    # traffic light
    (0, 0, 255),      # traffic sign
    (128, 0, 128),    # train
]

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
    print("Done")

# ── LOAD ONNX MODELS ──────────────────────────
print("Loading ONNX models...")
dms_session  = ort.InferenceSession(DMS_ONNX_PATH)
lane_session = ort.InferenceSession(LANE_ONNX_PATH)
yolo_session = ort.InferenceSession(YOLO_ONNX_PATH)

dms_input  = dms_session.get_inputs()[0].name
lane_input = lane_session.get_inputs()[0].name
yolo_input = yolo_session.get_inputs()[0].name
print("All 3 ONNX models loaded")

# ── LANDMARK INDICES ──────────────────────────
LEFT_EYE       = [33, 160, 158, 133, 153, 144]
RIGHT_EYE      = [362, 385, 387, 263, 373, 380]
MOUTH_LEFT     = 61
MOUTH_RIGHT    = 291
MOUTH_TOP_1, MOUTH_TOP_2, MOUTH_TOP_3 = 39, 0, 269
MOUTH_BOT_1, MOUTH_BOT_2, MOUTH_BOT_3 = 84, 14, 314
FACE_POINTS_IDX = [1, 152, 33, 263, 61, 291]

MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
], dtype=np.float64)

# ── THRESHOLDS ────────────────────────────────
EAR_THRESHOLD        = 0.25
MAR_THRESHOLD        = 0.5
CLOSED_FRAMES_LIMIT  = 20
YAWN_FRAMES_LIMIT    = 15
YAW_THRESHOLD        = 30
PITCH_THRESHOLD      = 20

# ── DMS HELPER FUNCTIONS ──────────────────────
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_indices, w, h):
    points = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = points
    ear = (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4))
    return ear, points

def calculate_mar(landmarks, w, h):
    def pt(idx): return (landmarks[idx].x * w, landmarks[idx].y * h)
    left, right = pt(MOUTH_LEFT), pt(MOUTH_RIGHT)
    v1 = euclidean(pt(MOUTH_TOP_1), pt(MOUTH_BOT_1))
    v2 = euclidean(pt(MOUTH_TOP_2), pt(MOUTH_BOT_2))
    v3 = euclidean(pt(MOUTH_TOP_3), pt(MOUTH_BOT_3))
    return (v1 + v2 + v3) / (3.0 * euclidean(left, right))

def calculate_head_pose(landmarks, w, h):
    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    image_2d = np.array([(landmarks[i].x * w, landmarks[i].y * h)
                         for i in FACE_POINTS_IDX], dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(
        MODEL_3D_POINTS, image_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success: return 0.0, 0.0, 0.0
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    if pitch > 90: pitch -= 180
    elif pitch < -90: pitch += 180
    if yaw > 90: yaw -= 180
    elif yaw < -90: yaw += 180
    return yaw, pitch, roll

def get_eye_crop(frame, landmarks, eye_indices, w, h, padding=10):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    x1 = max(0, min(p[0] for p in pts) - padding)
    y1 = max(0, min(p[1] for p in pts) - padding)
    x2 = min(w, max(p[0] for p in pts) + padding)
    y2 = min(h, max(p[1] for p in pts) + padding)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None

def predict_eye_onnx(crop):
    if crop is None: return "open"
    try:
        img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = img.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
        out = dms_session.run(None, {dms_input: img})[0]
        return "open" if np.argmax(out) == 0 else "closed"
    except:
        return "open"

# ── LANE DETECTION FUNCTION (improved for dashed lines) ──
def run_lane_detection(frame):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (800, 288))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = img.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
    out = lane_session.run(None, {lane_input: img})[0]

    griding_num = 100
    row_anchor = np.linspace(0.42, 1, 56)
    out = out[0]
    prob = np.exp(out[:-1]) / np.exp(out[:-1]).sum(axis=0, keepdims=True)
    idx = np.arange(griding_num).reshape(-1, 1, 1)
    loc = (prob * idx).sum(axis=0)

    # Lower threshold for catching dashed lines
    max_probs = prob.max(axis=0)
    loc[(out.argmax(axis=0) == griding_num) | (max_probs < 0.1)] = -1

    lane_colors = [(0,255,0), (255,255,0), (0,255,255), (255,0,255)]
    lanes_detected = 0
    for lane in range(4):
        points = []
        for row in range(56):
            if loc[row, lane] > 0:
                x = int(loc[row, lane] * w / griding_num)
                y = int(row_anchor[row] * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 4, lane_colors[lane], -1)
        if len(points) > 2:
            lanes_detected += 1
            for i in range(len(points)-1):
                cv2.line(frame, points[i], points[i+1], lane_colors[lane], 2)

    return frame, lanes_detected

# ── YOLO DETECTION FUNCTION (with BDD class names) ──
def run_yolo_detection(frame):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
    out = yolo_session.run(None, {yolo_input: img})[0]

    detections = out[0].T
    scores = detections[:, 4:].max(axis=1)
    classes = detections[:, 4:].argmax(axis=1)

    threshold = 0.3
    n_objects = 0
    for i, score in enumerate(scores):
        if score > threshold:
            cx, cy, bw, bh = detections[i, :4]
            cls_id = int(classes[i])

            x1 = int((cx - bw/2) * w / 512)
            y1 = int((cy - bh/2) * h / 512)
            x2 = int((cx + bw/2) * w / 512)
            y2 = int((cy + bh/2) * h / 512)

            class_name = BDD_CLASSES[cls_id] if cls_id < len(BDD_CLASSES) else 'unknown'
            color = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {score:.2f}"
            cv2.putText(frame, label, (x1, max(y1-5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            n_objects += 1

    return frame, n_objects

# ── STATE TRACKING ────────────────────────────
closed_frame_count = 0
yawn_frame_count = 0
yawn_alert_active = False
yawn_counter = 0
PERCLOS_WINDOW = deque(maxlen=300)

# Lane detection frame skip for speed boost
lane_frame_skip = 3
lane_frame_counter = 0
last_lane_frame = None
last_lanes_count = 0

# ── MEDIAPIPE OPTIONS ─────────────────────────
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_faces=1,
    running_mode=VisionRunningMode.IMAGE
)

# ── MAIN LOOP ─────────────────────────────────
print("Opening webcam and driving video...")
webcam = cv2.VideoCapture(0)
video = cv2.VideoCapture(DRIVING_VIDEO)

video_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_w = video_w * 2
out_h = video_h
writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'),
                         10, (out_w, out_h))

print("Processing — Press Q to stop\n")
frame_count = 0

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret_v, road_frame = video.read()
        ret_w, face_frame = webcam.read()
        if not ret_v or not ret_w: break

        face_frame = cv2.flip(face_frame, 1)
        fh, fw = face_frame.shape[:2]

        # ── DMS PROCESSING ──
        alert_level = 0
        status_text = "SAFE"
        rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            left_ear, _ = calculate_ear(lm, LEFT_EYE, fw, fh)
            right_ear, _ = calculate_ear(lm, RIGHT_EYE, fw, fh)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(lm, fw, fh)
            yaw, pitch, roll = calculate_head_pose(lm, fw, fh)

            left_crop = get_eye_crop(face_frame, lm, LEFT_EYE, fw, fh)
            right_crop = get_eye_crop(face_frame, lm, RIGHT_EYE, fw, fh)
            left_pred = predict_eye_onnx(left_crop)
            right_pred = predict_eye_onnx(right_crop)
            eye_state = "CLOSED" if (left_pred == "closed" and right_pred == "closed") else "OPEN"

            PERCLOS_WINDOW.append(1 if eye_state == "CLOSED" else 0)
            perclos = (sum(PERCLOS_WINDOW) / len(PERCLOS_WINDOW)) * 100

            if avg_ear < EAR_THRESHOLD:
                closed_frame_count += 1
            else:
                closed_frame_count = 0

            if closed_frame_count >= CLOSED_FRAMES_LIMIT or perclos > 15:
                alert_level = 2; status_text = "DROWSINESS!"
            elif avg_ear < EAR_THRESHOLD:
                alert_level = 1; status_text = "EYES CLOSING"

            if mar >= MAR_THRESHOLD:
                yawn_frame_count += 1
            else:
                if yawn_alert_active: yawn_counter += 1
                yawn_frame_count = 0
                yawn_alert_active = False

            if yawn_frame_count >= YAWN_FRAMES_LIMIT:
                yawn_alert_active = True
                alert_level = max(alert_level, 1)
                status_text = f"YAWNING x{yawn_counter+1}"

            if abs(yaw) > YAW_THRESHOLD:
                alert_level = max(alert_level, 1)
                status_text = "LOOKING AWAY"
            if abs(pitch) > PITCH_THRESHOLD:
                alert_level = max(alert_level, 1)
                status_text = "HEAD DOWN"

            cv2.putText(face_frame, f"EAR: {avg_ear:.2f}",  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(face_frame, f"MAR: {mar:.2f}",      (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(face_frame, f"EYE: {eye_state}",    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(face_frame, f"YAW: {yaw:+.0f}",     (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(face_frame, f"PERCLOS: {perclos:.0f}%", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            cv2.putText(face_frame, "NO FACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        border = (0,255,0) if alert_level == 0 else (0,255,255) if alert_level == 1 else (0,0,255)
        cv2.rectangle(face_frame, (0,0), (fw-1, fh-1), border, 6)
        cv2.putText(face_frame, status_text, (10, fh-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, border, 2)

        # ── LANE DETECTION (with frame skip for speed) ──
        lane_frame_counter += 1
        if lane_frame_counter % lane_frame_skip == 0 or last_lane_frame is None:
            road_processed, n_lanes = run_lane_detection(road_frame.copy())
            last_lane_frame = road_processed.copy()
            last_lanes_count = n_lanes
        else:
            road_processed = last_lane_frame.copy()
            n_lanes = last_lanes_count

        # ── YOLO DETECTION (every frame) ──
        road_processed, n_objects = run_yolo_detection(road_processed)

        cv2.putText(road_processed, f"Lanes: {n_lanes}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(road_processed, f"Objects: {n_objects}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # ── COMBINE FRAMES ──
        face_resized = cv2.resize(face_frame, (video_w, video_h))
        combined = cv2.hconcat([face_resized, road_processed])

        cv2.putText(combined, "DMS", (10, video_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(combined, "ADAS", (video_w + 10, video_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        writer.write(combined)
        cv2.imshow("Combined DMS + ADAS Pipeline", cv2.resize(combined, (1280, 480)))

        frame_count += 1
        if frame_count % 20 == 0:
            print(f"Processed {frame_count} frames")

        if cv2.waitKey(1) & 0xFF == ord('q'): break

print(f"\nDone — {frame_count} frames processed")
webcam.release()
video.release()
writer.release()
cv2.destroyAllWindows()
print(f"Demo saved: {OUTPUT_VIDEO}")