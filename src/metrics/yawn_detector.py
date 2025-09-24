# yawn_detector.py

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Mouth landmark indices for MAR calculation
# These indices are based on common representations for mouth landmarks in MediaPipe Face Mesh.
# You might need to adjust them based on the exact MediaPipe Face Mesh output and visual inspection.
# A good reference for MediaPipe Face Mesh landmarks can be found in their documentation or examples.

# Inner mouth landmarks for vertical distance
# These points represent the top and bottom of the inner mouth opening.
MOUTH_INNER_TOP = [13, 14]
MOUTH_INNER_BOTTOM = [16, 17]

# Mouth corners for horizontal distance
MOUTH_CORNERS = [61, 291]

# Yawn detection parameters
MAR_THRESHOLD = 0.6
YAWN_DURATION_THRESHOLD = 0.5 # seconds, how long MAR must be above threshold to count as a yawn
YAWN_COOLDOWN_PERIOD = 2 # seconds, to avoid counting multiple yawns from a single event
MAX_YAWNS_PER_PERIOD = 3
PERIOD_LENGTH_SECONDS = 120
MAR_SMOOTHING_WINDOW_SIZE = 5 # Number of frames to average MAR over for smoothing

# State variables
yawn_start_time = None
yawn_count = 0
yawn_timestamps = []
last_yawn_time = 0
mar_history = deque(maxlen=MAR_SMOOTHING_WINDOW_SIZE)

def calculate_mar(landmarks):
    # Extract coordinates for mouth landmarks
    # Assuming landmarks are normalized (0 to 1)

    # Calculate vertical distance (average of distances between top and bottom inner lip points)
    p1 = np.array([landmarks[MOUTH_INNER_TOP[0]].x, landmarks[MOUTH_INNER_TOP[0]].y])
    p2 = np.array([landmarks[MOUTH_INNER_BOTTOM[0]].x, landmarks[MOUTH_INNER_BOTTOM[0]].y])
    p3 = np.array([landmarks[MOUTH_INNER_TOP[1]].x, landmarks[MOUTH_INNER_TOP[1]].y])
    p4 = np.array([landmarks[MOUTH_INNER_BOTTOM[1]].x, landmarks[MOUTH_INNER_BOTTOM[1]].y])

    vertical_dist1 = np.linalg.norm(p1 - p2)
    vertical_dist2 = np.linalg.norm(p3 - p4)
    vertical_dist = (vertical_dist1 + vertical_dist2) / 2.0

    # Calculate horizontal distance (between mouth corners)
    p5 = np.array([landmarks[MOUTH_CORNERS[0]].x, landmarks[MOUTH_CORNERS[0]].y])
    p6 = np.array([landmarks[MOUTH_CORNERS[1]].x, landmarks[MOUTH_CORNERS[1]].y])
    horizontal_dist = np.linalg.norm(p5 - p6)

    if horizontal_dist == 0:
        return 0

    mar = vertical_dist / horizontal_dist
    return mar

# Main loop for video capture
cap = cv2.VideoCapture(0) # Use 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Flip for selfie-view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_time = time.time()
    smoothed_mar = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mar = calculate_mar(face_landmarks.landmark)
            mar_history.append(mar)
            smoothed_mar = np.mean(mar_history)

            # Yawn detection logic
            if smoothed_mar > MAR_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = current_time
                elif (current_time - yawn_start_time) > YAWN_DURATION_THRESHOLD:
                    if (current_time - last_yawn_time) > YAWN_COOLDOWN_PERIOD:
                        yawn_count += 1
                        yawn_timestamps.append(current_time)
                        last_yawn_time = current_time
                        print(f"Yawn detected! Total yawns: {yawn_count}")
                    yawn_start_time = None # Reset for next yawn
            else:
                yawn_start_time = None

            # Frequency tracking
            # Remove yawns older than PERIOD_LENGTH_SECONDS
            yawn_timestamps = [ts for ts in yawn_timestamps if (current_time - ts) < PERIOD_LENGTH_SECONDS]

            if len(yawn_timestamps) > MAX_YAWNS_PER_PERIOD:
                cv2.putText(frame, "DROWSINESS ALERT! Too many yawns!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, f"MAR: {smoothed_mar:.2f}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Yawns (120s): {len(yawn_timestamps)}", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Yawn Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


