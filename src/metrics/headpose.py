import numpy as np
import cv2
import mediapipe as mp
import time
import math
from collections import deque

# --- Mediapipe Face Mesh setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- OpenCV webcam ---
cap = cv2.VideoCapture(0)

# --- Thresholds and timers ---
DOWN_THRESHOLD = -15      # Pitch threshold for head down
TILT_THRESHOLD = 15       # Roll threshold in degrees
WARNING_DELAY = 2.0       # Seconds before showing warning
ROLL_DEADZONE = 5         # Degrees below which roll is ignored
GRACE_PERIOD = 1.0        # Short grace period to avoid flicker

# --- Nodding detection parameters ---
NOD_DOWN_THRESHOLD = -12
NOD_UP_THRESHOLD = -5
NOD_DETECTION_WINDOW = 15
MAX_NOD_DURATION = 3
MIN_NODS_FOR_DROWSY = 3
DROWSY_WARNING_DURATION = 10

# --- State variables ---
down_start_time = None
tilt_start_time = None
nod_state = "neutral"
nod_start_time = None
nod_cycles = deque(maxlen=10)
drowsy_warning_time = None
head_down_state = False
head_tilt_state = False

# --- Smoothing ---
SMOOTHING_WINDOW = 5
alpha = 0.2
pitch_history = deque(maxlen=SMOOTHING_WINDOW)
roll_history = deque(maxlen=SMOOTHING_WINDOW)

def calculate_roll_angle(landmarks, img_w, img_h):
    """Calculate roll angle using landmarks on the sides of the face"""
    try:
        left = landmarks[127]
        right = landmarks[356]
        dx = (right.x - left.x) * img_w
        dy = (right.y - left.y) * img_h
        if abs(dx) < 0.1:
            return 0
        angle = math.degrees(math.atan2(dy, dx))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        return angle
    except:
        return 0

def smooth_angle(angle, history):
    """Smooth angle using a moving average"""
    history.append(angle)
    if len(history) >= 3:
        return np.mean(list(history))
    return angle

def detect_nodding(pitch):
    """Detect nodding patterns indicating drowsiness"""
    global nod_state, nod_start_time, nod_cycles, drowsy_warning_time
    current_time = time.perf_counter()
    is_drowsy = False
    nod_info = ""

    # Nodding state machine
    if nod_state == "neutral" and pitch < NOD_DOWN_THRESHOLD:
        nod_state = "going_down"
        nod_start_time = current_time
        nod_info = "Head dropping..."
    elif nod_state == "going_down":
        if pitch > NOD_UP_THRESHOLD:
            duration = current_time - nod_start_time
            if duration <= MAX_NOD_DURATION:
                nod_cycles.append(current_time)
                nod_info = f"Nod detected! ({len(nod_cycles)} recent)"
            nod_state = "neutral"
            nod_start_time = None
        elif current_time - nod_start_time > MAX_NOD_DURATION:
            nod_state = "neutral"
            nod_start_time = None

    # Remove old nods
    while nod_cycles and current_time - nod_cycles[0] > NOD_DETECTION_WINDOW:
        nod_cycles.popleft()

    # Check drowsiness
    if len(nod_cycles) >= MIN_NODS_FOR_DROWSY:
        is_drowsy = True
        drowsy_warning_time = current_time
        nod_info = f"DROWSY! ({len(nod_cycles)} nods in {NOD_DETECTION_WINDOW}s)"

    # Keep drowsy warning visible for duration
    if drowsy_warning_time and current_time - drowsy_warning_time <= DROWSY_WARNING_DURATION:
        is_drowsy = True
    elif drowsy_warning_time and current_time - drowsy_warning_time > DROWSY_WARNING_DURATION:
        drowsy_warning_time = None

    return is_drowsy, nod_info

# --- Main loop ---
while cap.isOpened():
    start = time.time()
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    img_h, img_w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = False
    results = face_mesh.process(rgb_image)
    rgb_image.flags.writeable = True

    head_text = "Forward"
    pitch = 0
    roll = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = list(face_landmarks.landmark)

            # Roll calculation and smoothing
            raw_roll = calculate_roll_angle(landmarks, img_w, img_h)
            roll = smooth_angle(raw_roll, roll_history)
            if abs(roll) < ROLL_DEADZONE:
                roll = 0

            # SolvePnP for pitch calculation
            face_2d, face_3d = [], []
            for idx, lm in enumerate(landmarks):
                if idx in [33, 263, 1, 61, 291, 199]:
                    x_coord, y_coord = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x_coord, y_coord])
                    face_3d.append([x_coord, y_coord, lm.z])
            if face_2d:
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length,0,img_h/2],
                                       [0,focal_length,img_w/2],
                                       [0,0,1]], dtype=np.float64)
                dist_matrix = np.zeros((4,1), dtype=np.float64)
                success_pnp, rotation_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                if success_pnp:
                    rmat, _ = cv2.Rodrigues(rotation_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                    raw_pitch = angles[0]*360
                    pitch = smooth_angle(raw_pitch, pitch_history)

            # --- Nodding detection ---
            is_drowsy, nod_info = detect_nodding(pitch)

            # --- Head down / tilt state with hysteresis ---
            now = time.perf_counter()
            if not head_down_state and pitch < DOWN_THRESHOLD:
                if down_start_time is None:
                    down_start_time = now
                elif now - down_start_time >= WARNING_DELAY:
                    head_down_state = True
            elif head_down_state and pitch > DOWN_THRESHOLD + GRACE_PERIOD:
                head_down_state = False
                down_start_time = None

            if not head_tilt_state and abs(roll) > TILT_THRESHOLD:
                if tilt_start_time is None:
                    tilt_start_time = now
                elif now - tilt_start_time >= WARNING_DELAY:
                    head_tilt_state = True
            elif head_tilt_state and abs(roll) < TILT_THRESHOLD - GRACE_PERIOD:
                head_tilt_state = False
                tilt_start_time = None

            # --- Head direction text ---
            text_parts = []
            if roll < -TILT_THRESHOLD:
                text_parts.append("Tilted Left")
            elif roll > TILT_THRESHOLD:
                text_parts.append("Tilted Right")
            if pitch < -10:
                text_parts.append("Looking Down")
            elif pitch > 10:
                text_parts.append("Looking Up")
            head_text = " + ".join(text_parts) if text_parts else "Forward"

    # --- Dynamic text display ---
    y = 30
    line_gap = 30
    cv2.putText(image, f"Head: {head_text}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    y += line_gap
    cv2.putText(image, f"Pitch: {pitch:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
    y += line_gap
    cv2.putText(image, f"Roll: {roll:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
    y += line_gap
    cv2.putText(image, f"Nods: {len(nod_cycles)}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)
    y += line_gap
    cv2.putText(image, f"State: {nod_state}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)
    y += line_gap

    if is_drowsy:
        cv2.putText(image, "ALERT: DRIVER DROWSY!", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),3)
        y += int(line_gap*1.2)
    elif head_down_state:
        cv2.putText(image, "WARNING: Head too low!", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,100,255),3)
        y += int(line_gap*1.2)
    elif head_tilt_state:
        tilt_dir = "left" if roll < 0 else "right"
        cv2.putText(image, f"WARNING: Head tilted {tilt_dir}!", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,100,255),3)
        y += int(line_gap*1.2)

    if nod_info:
        cv2.putText(image, nod_info, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0),2)

    # --- FPS ---
    fps = 1.0 / max(time.time() - start, 1e-6)
    cv2.putText(image, f"FPS: {fps:.1f}", (20, img_h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

    cv2.imshow("Drowsiness Detection System", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
