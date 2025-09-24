# drowsiness_detection_save_video_webcam.py

import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import os
from datetime import datetime


class DrowsinessDetector:
    def __init__(self, max_idle_time=5, buffer_len=50):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.eye_directions = deque(maxlen=buffer_len)
        self.last_drowsy_time = None
        self.max_idle_time = max_idle_time

    def get_pupil_center(self, landmarks, iris_indices, w, h):
        points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in iris_indices])
        center = np.mean(points, axis=0)
        return tuple(center.astype(int))

    def get_eye_socket_center(self, landmarks, eye_socket_indices, w, h):
        points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_socket_indices])
        center = np.mean(points, axis=0)
        return tuple(center.astype(int))

    def get_eye_direction(self, pupil_center, eye_socket_center, scale=50):
        dx = (pupil_center[0] - eye_socket_center[0]) * scale / 50
        dy = (pupil_center[1] - eye_socket_center[1]) * scale / 50
        end_point = (int(eye_socket_center[0] + dx), int(eye_socket_center[1] + dy))
        return end_point

    def detect_drowsiness(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        drowsy = False
        eye_direction = (None, None)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                # Draw face rectangle
                x_min = int(min([lm.x for lm in face_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in face_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in face_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in face_landmarks.landmark]) * h)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, 'Driver', (x_min, y_min-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Iris and eye socket landmarks
                LEFT_IRIS = [474, 475, 476, 477]
                RIGHT_IRIS = [469, 470, 471, 472]
                LEFT_EYE = [33, 133, 159, 145]
                RIGHT_EYE = [362, 263, 386, 374]

                # Pupil centers
                left_pupil = self.get_pupil_center(face_landmarks.landmark, LEFT_IRIS, w, h)
                right_pupil = self.get_pupil_center(face_landmarks.landmark, RIGHT_IRIS, w, h)
                pupil_center = ((left_pupil[0]+right_pupil[0])//2, (left_pupil[1]+right_pupil[1])//2)

                # Eye socket centers
                left_eye_center = self.get_eye_socket_center(face_landmarks.landmark, LEFT_EYE, w, h)
                right_eye_center = self.get_eye_socket_center(face_landmarks.landmark, RIGHT_EYE, w, h)
                eye_socket_center = ((left_eye_center[0]+right_eye_center[0])//2, (left_eye_center[1]+right_eye_center[1])//2)

                # Draw direction line
                end_point = self.get_eye_direction(pupil_center, eye_socket_center, scale=50)
                cv2.arrowedLine(frame, eye_socket_center, end_point, (0, 0, 255), 2, tipLength=0.3)

                # Normalized eye direction for drowsiness detection
                norm_x_left = (left_pupil[0] - left_eye_center[0]) / (left_eye_center[0] * 0.5)
                norm_y_left = (left_pupil[1] - left_eye_center[1]) / (left_eye_center[1] * 0.5)
                norm_x_right = (right_pupil[0] - right_eye_center[0]) / (right_eye_center[0] * 0.5)
                norm_y_right = (right_pupil[1] - right_eye_center[1]) / (right_eye_center[1] * 0.5)
                avg_dir = ((norm_x_left+norm_x_right)/2, (norm_y_left+norm_y_right)/2)
                eye_direction = avg_dir
                self.eye_directions.append(avg_dir)

                print(f"Eye Direction (norm): {avg_dir[0]:.3f}, {avg_dir[1]:.3f}")

                # Drowsiness check
                if len(self.eye_directions) == self.eye_directions.maxlen:
                    dx = max(d[0] for d in self.eye_directions) - min(d[0] for d in self.eye_directions)
                    dy = max(d[1] for d in self.eye_directions) - min(d[1] for d in self.eye_directions)
                    if dx < 0.02 and dy < 0.02:
                        if self.last_drowsy_time is None:
                            self.last_drowsy_time = time.time()
                        elif time.time() - self.last_drowsy_time >= self.max_idle_time:
                            drowsy = True
                    else:
                        self.last_drowsy_time = None

        return frame, drowsy, eye_direction


if __name__ == "__main__":
    detector = DrowsinessDetector(max_idle_time=5)

    # ---------- Webcam Setup ----------
    cap = cv2.VideoCapture(0)  # Use default webcam (index 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20  # fallback if fps=0

    # Timestamped filename in same folder as script
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"driver_drowsiness_{timestamp}.mp4")

    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    print(f"Saving video to: {video_filename}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, is_drowsy, eye_dir = detector.detect_drowsiness(frame)

        if is_drowsy:
            cv2.putText(annotated_frame, 'DROWSINESS ALERT!', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if eye_dir[0] is not None:
            cv2.putText(annotated_frame, f"Eye Dir: ({eye_dir[0]:.2f}, {eye_dir[1]:.2f})",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Driver Drowsiness Detection", annotated_frame)

        # Write frame to file
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
