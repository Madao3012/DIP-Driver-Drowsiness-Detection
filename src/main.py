from picamera2 import Picamera2
import cv2
import mediapipe as mp
import time

from ear import LEFT_EYE, RIGHT_EYE, to_xy, ear_from_pts, EARState
from mar import mar_from_landmarks
from pose import euler_angles_from_landmarks

# ---------- Camera ----------
picam = Picamera2()
config = picam.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam.configure(config)
picam.start()
time.sleep(0.1)

# ---------- MediaPipe Face Mesh ----------
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# ---------- EAR state ----------
ear_state = EARState(thresh=0.21, ema_alpha=0.7, closed_needed=3)

print("Press 'q' to quit.")

while True:
    frame_rgb = picam.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame_rgb.shape[:2]

    res = face_mesh.process(frame_rgb)

    if res.multi_face_landmarks:
        fl = res.multi_face_landmarks[0]

        # EAR
        left_pts = to_xy(fl.landmark, w, h, LEFT_EYE)
        right_pts = to_xy(fl.landmark, w, h, RIGHT_EYE)
        left_ear = ear_from_pts(left_pts)
        right_ear = ear_from_pts(right_pts)
        ear_value = (left_ear + right_ear) / 2.0
        smoothed_ear, eyes_closed = ear_state.update(ear_value)

        # MAR
        mar_value = mar_from_landmarks(fl.landmark, w, h)

        # Head pose
        yaw, pitch, roll = euler_angles_from_landmarks(fl.landmark, w, h)

        # Overlay text
        status = "DROWSY" if eyes_closed else "ALERT"
        cv2.putText(frame_bgr, f"EAR: {smoothed_ear:.3f} ({status})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if eyes_closed else (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"MAR: {mar_value:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if yaw is not None:
            cv2.putText(frame_bgr, f"Pose (Y,P,R): {yaw:.1f}, {pitch:.1f}, {roll:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw landmarks
        mp_draw.draw_landmarks(frame_bgr, fl, mp_face_mesh.FACEMESH_CONTOURS)

    else:
        cv2.putText(frame_bgr, "NO FACE DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam.stop()
face_mesh.close()
cv2.destroyAllWindows()