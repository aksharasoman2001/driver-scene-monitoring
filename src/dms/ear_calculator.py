# ear_calculator.py
# Eye Aspect Ratio (EAR) Calculator using MediaPipe Face Landmarks
# Day 4-5 of Week 1

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

# ── EAR Landmark Indices (MediaPipe 478-point model) ─
# Each eye needs 6 specific points: [outer, top1, top2, inner, bot1, bot2]
LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR threshold — below this = eye is CLOSED
EAR_THRESHOLD = 0.25
# Number of consecutive frames eye must be closed to trigger alert
CLOSED_FRAMES_LIMIT = 20

# ── EAR Formula ──────────────────────────────
def euclidean(p1, p2):
    """Calculate Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_indices, frame_w, frame_h):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    p1 = outer corner
    p2 = upper-outer
    p3 = upper-inner
    p4 = inner corner
    p5 = lower-inner
    p6 = lower-outer
    """
    # Extract the 6 points for this eye
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x = lm.x * frame_w
        y = lm.y * frame_h
        points.append((x, y))

    p1, p2, p3, p4, p5, p6 = points

    # Vertical distances
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)

    # Horizontal distance
    horizontal = euclidean(p1, p4)

    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear, points

# ── Draw Eye Points on Frame ─────────────────
def draw_eye_points(frame, points, color):
    for (x, y) in points:
        cv2.circle(frame, (int(x), int(y)), 2, color, -1)

# ── Setup FaceLandmarker ─────────────────────
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    running_mode=VisionRunningMode.IMAGE
)

# ── State Tracking ────────────────────────────
closed_frame_count = 0
alert_active = False

# ── Open Camera ──────────────────────────────
cap = cv2.VideoCapture(0)
print("Camera started! Press Q to quit")
print("Watch your EAR values in the terminal — blink to see them drop!\n")

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

            # ── Calculate EAR for both eyes ───────────
            left_ear,  left_pts  = calculate_ear(landmarks, LEFT_EYE,  w, h)
            right_ear, right_pts = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            # ── Print to terminal every frame ─────────
            status = "OPEN  👀" if avg_ear >= EAR_THRESHOLD else "CLOSED 😴"
            print(f"Left EAR: {left_ear:.3f} | Right EAR: {right_ear:.3f} | Avg EAR: {avg_ear:.3f} | Eye: {status}")

            # ── Drowsiness alert logic ─────────────────
            if avg_ear < EAR_THRESHOLD:
                closed_frame_count += 1
            else:
                closed_frame_count = 0
                alert_active = False

            if closed_frame_count >= CLOSED_FRAMES_LIMIT:
                alert_active = True

            # ── Draw eye landmark points ───────────────
            draw_eye_points(frame, left_pts,  (0, 255, 0))   # Green = left
            draw_eye_points(frame, right_pts, (255, 0, 0))   # Blue  = right

            # ── HUD Overlay ───────────────────────────
            # EAR values
            cv2.putText(frame, f"Left  EAR: {left_ear:.3f}",  (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, f"Right EAR: {right_ear:.3f}", (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
            cv2.putText(frame, f"Avg   EAR: {avg_ear:.3f}",   (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Threshold: {EAR_THRESHOLD}",  (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

            # Eye status
            if alert_active:
                # Red flashing alert
                cv2.putText(frame, "⚠ DROWSINESS ALERT!", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 6)
            elif avg_ear < EAR_THRESHOLD:
                cv2.putText(frame, f"EYES CLOSED ({closed_frame_count} frames)", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
                cv2.rectangle(frame, (0, 0), (w, h), (0, 165, 255), 4)
            else:
                cv2.putText(frame, "EYES OPEN - Safe", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 3)

        else:
            cv2.putText(frame, "No Face Detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("DMS - EAR Calculator (Day 4-5)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\nDone! ✅")