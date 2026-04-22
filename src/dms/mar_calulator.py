# mar_calculator.py
# Mouth Aspect Ratio (MAR) Calculator using MediaPipe Face Landmarks
# Day 6 of Week 1

import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

# ── MediaPipe Setup ──────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ── Download Model if Needed ─────────────────
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading MediaPipe face model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )
    print("Download complete! ✅")

# ── MAR Landmark Indices (MediaPipe 478-point model) ─
# Outer corners of mouth
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
# Upper lip points
MOUTH_TOP_1  = 39    # upper-outer
MOUTH_TOP_2  = 0     # upper-middle
MOUTH_TOP_3  = 269   # upper-inner
# Lower lip points
MOUTH_BOT_1  = 84   # lower-outer
MOUTH_BOT_2  = 14    # lower-middle
MOUTH_BOT_3  = 314   # lower-inner

MOUTH_INDICES = [
    MOUTH_LEFT, MOUTH_RIGHT,
    MOUTH_TOP_1, MOUTH_TOP_2, MOUTH_TOP_3,
    MOUTH_BOT_1, MOUTH_BOT_2, MOUTH_BOT_3
]

# MAR threshold — above this = mouth is OPEN (yawning)
MAR_THRESHOLD = 0.5
# Consecutive frames mouth must be open to trigger yawn alert
YAWN_FRAMES_LIMIT = 15

# ── Helper Functions ──────────────────────────
def euclidean(p1, p2):
    """Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_point(landmarks, idx, w, h):
    """Convert normalized landmark to pixel coordinates."""
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)

def calculate_mar(landmarks, frame_w, frame_h):
    """
    MAR = (||top1-bot1|| + ||top2-bot2|| + ||top3-bot3||) / (3 * ||left-right||)

    Three vertical distances across the mouth
    divided by horizontal mouth width × 3
    """
    left  = get_point(landmarks, MOUTH_LEFT,  frame_w, frame_h)
    right = get_point(landmarks, MOUTH_RIGHT, frame_w, frame_h)

    top1  = get_point(landmarks, MOUTH_TOP_1, frame_w, frame_h)
    top2  = get_point(landmarks, MOUTH_TOP_2, frame_w, frame_h)
    top3  = get_point(landmarks, MOUTH_TOP_3, frame_w, frame_h)

    bot1  = get_point(landmarks, MOUTH_BOT_1, frame_w, frame_h)
    bot2  = get_point(landmarks, MOUTH_BOT_2, frame_w, frame_h)
    bot3  = get_point(landmarks, MOUTH_BOT_3, frame_w, frame_h)

    # Three vertical distances
    vertical_1 = euclidean(top1, bot1)
    vertical_2 = euclidean(top2, bot2)
    vertical_3 = euclidean(top3, bot3)

    # Horizontal mouth width
    horizontal = euclidean(left, right)

    mar = (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)

    all_points = [left, right, top1, top2, top3, bot1, bot2, bot3]
    return mar, all_points

def draw_mouth_points(frame, points):
    colors = [
        (0, 255, 255),   # left corner  - yellow
        (0, 255, 255),   # right corner - yellow
        (0, 165, 255),   # top1 - orange
        (0, 165, 255),   # top2 - orange
        (0, 165, 255),   # top3 - orange
        (255, 0, 255),   # bot1 - magenta
        (255, 0, 255),   # bot2 - magenta
        (255, 0, 255),   # bot3 - magenta
    ]
    for (x, y), color in zip(points, colors):
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)

# ── Setup FaceLandmarker ─────────────────────
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    running_mode=VisionRunningMode.IMAGE
)

# ── State Tracking ────────────────────────────
yawn_frame_count = 0
yawn_alert_active = False
yawn_counter = 0       # total yawns detected this session

# ── Open Camera ──────────────────────────────
cap = cv2.VideoCapture(0)
print("Camera started! Press Q to quit")
print("Watch your MAR values — open your mouth wide to see them rise!\n")

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # ── Calculate MAR ─────────────────────────
            mar, mouth_pts = calculate_mar(landmarks, w, h)

            # ── Print to terminal every frame ─────────
            status = "YAWNING 🥱" if mar >= MAR_THRESHOLD else "NORMAL  😐"
            print(f"MAR: {mar:.3f} | Threshold: {MAR_THRESHOLD} | Status: {status} | Total Yawns: {yawn_counter}")

            # ── Yawn alert logic ──────────────────────
            if mar >= MAR_THRESHOLD:
                yawn_frame_count += 1
            else:
                # Yawn just ended — count it
                if yawn_alert_active:
                    yawn_counter += 1
                yawn_frame_count = 0
                yawn_alert_active = False

            if yawn_frame_count >= YAWN_FRAMES_LIMIT:
                yawn_alert_active = True

            # ── Draw mouth landmark points ─────────────
            draw_mouth_points(frame, mouth_pts)

            # ── HUD Overlay ───────────────────────────
            cv2.putText(frame, f"MAR: {mar:.3f}",             (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"Threshold: {MAR_THRESHOLD}", (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
            cv2.putText(frame, f"Yawns: {yawn_counter}",      (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Alert states
            if yawn_alert_active:
                cv2.putText(frame, "⚠ YAWN DETECTED!", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)
                cv2.rectangle(frame, (0, 0), (w, h), (0, 165, 255), 6)
            elif mar >= MAR_THRESHOLD:
                cv2.putText(frame, f"Mouth Open ({yawn_frame_count} frames)", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 255), 4)
            else:
                cv2.putText(frame, "Mouth Closed - Normal", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 3)

        else:
            cv2.putText(frame, "No Face Detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("DMS - MAR Calculator (Day 6)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"\nSession complete! Total yawns detected: {yawn_counter} ✅")