import cv2
import mediapipe as mp
from math import hypot
import time

# --- EAR Utilities ---
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def _dist(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])

def ear_from_pts(pts):
    if len(pts) != 6:
        return 0.0
    p1, p2, p3, p4, p5, p6 = pts
    h = _dist(p1, p4)
    if h == 0:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * h)

def to_xy(landmarks, frame_w, frame_h, idxs):
    out = []
    for i in idxs:
        lm = landmarks[i]
        x = min(max(lm.x, 0.0), 1.0)
        y = min(max(lm.y, 0.0), 1.0)
        out.append((int(x * frame_w), int(y * frame_h)))
    return out

class EARState:
    def __init__(self, thresh=0.21, alpha=0.7, closed_needed=3):
        self.thresh = thresh
        self.alpha = alpha
        self.closed_needed = closed_needed
        self.ema = None
        self.closed_run = 0

    def reset(self):
        self.ema = None
        self.closed_run = 0

    def update(self, ear_value):
        if ear_value is None:
            return self.ema, self.closed_run >= self.closed_needed
        if self.ema is None:
            self.ema = ear_value
        else:
            self.ema = self.alpha * self.ema + (1 - self.alpha) * ear_value
        if self.ema < self.thresh:
            self.closed_run += 1
        else:
            self.closed_run = 0
        return self.ema, self.closed_run >= self.closed_needed

# --- Webcam + MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)
ear_state = EARState()

def calibrate_ear(duration=3):
    """Calibration routine: measures open-eye EAR for `duration` seconds."""
    print("Calibration started: Keep your eyes OPEN...")
    ears = []
    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_pts  = to_xy(landmarks, frame.shape[1], frame.shape[0], LEFT_EYE)
            right_pts = to_xy(landmarks, frame.shape[1], frame.shape[0], RIGHT_EYE)
            ear = (ear_from_pts(left_pts) + ear_from_pts(right_pts)) / 2.0
            ears.append(ear)
            cv2.putText(frame, "Calibrating OPEN...", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Blink Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to abort
            break
    if ears:
        avg_ear = sum(ears)/len(ears)
        ear_state.thresh = avg_ear * 0.7
        print(f"Calibration complete. EAR threshold set to {ear_state.thresh:.2f}")

# --- Main Loop ---
print("Press 'C' to calibrate. Press ESC to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_pts  = to_xy(landmarks, frame.shape[1], frame.shape[0], LEFT_EYE)
        right_pts = to_xy(landmarks, frame.shape[1], frame.shape[0], RIGHT_EYE)
        ear = (ear_from_pts(left_pts) + ear_from_pts(right_pts)) / 2.0
        smoothed_ear, closed = ear_state.update(ear)
        status = "CLOSED" if closed else "OPEN"
        cv2.putText(frame, f"EAR: {smoothed_ear:.2f} {status}", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if closed else (0,255,0),2)
        for pt in left_pts + right_pts:
            cv2.circle(frame, pt, 2, (255,0,0), -1)

    cv2.imshow("Blink Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('c') or key == ord('C'):
        calibrate_ear()

cap.release()
cv2.destroyAllWindows()
