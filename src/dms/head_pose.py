# head_pose.py
# Head Pose Estimation using MediaPipe + cv2.solvePnP
# Day 7 of Week 1

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

# ── 6 Key Landmark Indices for solvePnP ──────
FACE_POINTS_IDX = [1, 152, 33, 263, 61, 291]

# ── 3D Reference Points (real world coordinates in mm) ─
MODEL_3D_POINTS = np.array([
    (0.0,    0.0,    0.0),     # Nose tip
    (0.0,   -63.6,  -12.5),   # Chin
    (-43.3,  32.7,  -26.0),   # Left eye corner
    (43.3,   32.7,  -26.0),   # Right eye corner
    (-28.9, -28.9,  -24.1),   # Left mouth corner
    (28.9,  -28.9,  -24.1),   # Right mouth corner
], dtype=np.float64)

# ── Alert Thresholds ──────────────────────────
YAW_THRESHOLD   = 30   # degrees — looking left/right
PITCH_THRESHOLD = 20   # degrees — looking up/down
ROLL_THRESHOLD  = 20   # degrees — head tilting

# ── Setup FaceLandmarker ─────────────────────
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    running_mode=VisionRunningMode.IMAGE
)

# ── Helper: Draw Angle Bar ────────────────────
def draw_angle_bar(frame, x, y, label, angle, threshold, max_angle=45):
    bar_width = 150
    filled = int(abs(angle) / max_angle * bar_width)
    filled = min(filled, bar_width)
    color = (0, 255, 0) if abs(angle) < threshold else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + bar_width, y + 15), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + 15), color, -1)
    cv2.putText(frame, f"{label}: {angle:+.1f} deg",
        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

# ── Helper: Draw 3D Axes on Face ──────────────
def draw_axes(frame, nose_point, rotation_vec, translation_vec, camera_matrix, dist_coeffs):
    axis_length = 50.0
    axis_points = np.float32([
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length],
    ])
    imgpts, _ = cv2.projectPoints(axis_points, rotation_vec, translation_vec, camera_matrix, dist_coeffs)
    nose = tuple(nose_point.astype(int))
    cv2.arrowedLine(frame, nose, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 2)
    cv2.arrowedLine(frame, nose, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 2)
    cv2.arrowedLine(frame, nose, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 2)

# ── Open Camera ──────────────────────────────
cap = cv2.VideoCapture(0)
print("Camera started! Press Q to quit")
print("Move your head left/right/up/down to see angles change!\n")

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ── Camera Matrix ─────────────────────────
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0,            center[0]],
            [0,            focal_length, center[1]],
            [0,            0,            1         ]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # ── Extract 6 key 2D points ───────────
            image_2d_points = []
            for idx in FACE_POINTS_IDX:
                lm = landmarks[idx]
                x_px = lm.x * w
                y_px = lm.y * h
                image_2d_points.append((x_px, y_px))
            image_2d_points = np.array(image_2d_points, dtype=np.float64)

            # ── solvePnP ──────────────────────────
            success, rotation_vec, translation_vec = cv2.solvePnP(
                MODEL_3D_POINTS,
                image_2d_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # ── Convert to angles ──────────────
                rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

                pitch = angles[0]
                yaw   = angles[1]
                roll  = angles[2]

                # ── Normalize angles ✅ (this is the fix!) ─
                if pitch > 90:
                    pitch = pitch - 180
                elif pitch < -90:
                    pitch = pitch + 180

                if yaw > 90:
                    yaw = yaw - 180
                elif yaw < -90:
                    yaw = yaw + 180

                # ── Print to terminal ──────────────
                print(f"Yaw: {yaw:+.1f}° | Pitch: {pitch:+.1f}° | Roll: {roll:+.1f}°")

                # ── Draw 3D axes on nose ───────────
                nose_2d = image_2d_points[0]
                draw_axes(frame, nose_2d, rotation_vec, translation_vec, camera_matrix, dist_coeffs)

                # ── Draw angle bars ────────────────
                draw_angle_bar(frame, 10, 40,  "Yaw  (L/R)", yaw,   YAW_THRESHOLD)
                draw_angle_bar(frame, 10, 80,  "Pitch(U/D)", pitch, PITCH_THRESHOLD)
                draw_angle_bar(frame, 10, 120, "Roll (tilt)", roll,  ROLL_THRESHOLD)

                # ── Alert status ───────────────────
                alerts = []
                if abs(yaw)   > YAW_THRESHOLD:   alerts.append("DISTRACTION! Looking Away")
                if abs(pitch) > PITCH_THRESHOLD: alerts.append("HEAD DOWN! Drowsy?")
                if abs(roll)  > ROLL_THRESHOLD:  alerts.append("HEAD TILT! Check Driver")

                if alerts:
                    for i, alert in enumerate(alerts):
                        cv2.putText(frame, f"!! {alert}", (10, 170 + i * 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 6)
                else:
                    cv2.putText(frame, "Head Pose: Normal", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 3)

        else:
            cv2.putText(frame, "No Face Detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("DMS - Head Pose Estimation (Day 7)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\nDone! ✅")