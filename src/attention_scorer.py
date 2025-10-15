# attention_scorer.py — unified temporal scoring (EAR, gaze, pose; PERCLOS; nod detection)
import time
import numpy as np
from collections import deque

class AttentionScorer:
    def __init__(
        self,
        t_now: float,
        ear_thresh: float,
        gaze_thresh: float,
        perclos_thresh: float,
        roll_thresh: float,
        pitch_thresh: float,
        yaw_thresh: float,
        ear_time_thresh: float,
        gaze_time_thresh: float,
        pose_time_thresh: float,
        decay_factor: float = 0.9,
        perclos_window: float = 30.0,
        # Nod detection
        nod_down_thresh: float = -12.0,
        nod_up_thresh: float = -5.0,
        nod_window: int = 15,
        nods_for_drowsy: int = 3,
        nod_window_seconds: float = 10.0,
        verbose: bool = False,
    ):
        self.verbose = verbose
        # thresholds
        self.ear_thresh = ear_thresh
        self.gaze_thresh = gaze_thresh
        self.perclos_thresh = perclos_thresh
        self.roll_thresh = roll_thresh
        self.pitch_thresh = pitch_thresh
        self.yaw_thresh = yaw_thresh
        self.ear_time_thresh = ear_time_thresh
        self.gaze_time_thresh = gaze_time_thresh
        self.pose_time_thresh = pose_time_thresh
        self.decay_factor = decay_factor
        # timers
        self.last_time = t_now
        self.ear_low_acc = 0.0
        self.gaze_high_acc = 0.0
        self.pose_bad_acc = 0.0
        # perclos rolling window of boolean 'closed' flags with their timestamps
        self.perclos_window = perclos_window
        self.closed_history = deque()  # (timestamp, closed_bool)
        # nod detection
        self.nod_down_thresh = nod_down_thresh
        self.nod_up_thresh = nod_up_thresh
        self.nod_window = max(1, nod_window)
        self.pitch_history = deque(maxlen=self.nod_window)
        self.nod_state = "idle"  # idle|down|up_wait
        self.nod_times = deque()  # timestamps of nod completions
        self.nods_for_drowsy = nods_for_drowsy
        self.nod_window_seconds = nod_window_seconds

    def _update_timer(self, t_now: float):
        dt = max(0.0, t_now - self.last_time)
        self.last_time = t_now
        return dt

    def get_rolling_PERCLOS(self, t_now: float, ear_value: float):
        # update history
        closed = (ear_value is not None) and (ear_value < self.ear_thresh)
        self.closed_history.append((t_now, closed))

        # purge old entries
        while self.closed_history and (t_now - self.closed_history[0][0] > self.perclos_window):
            self.closed_history.popleft()

        # compute percentage closed
        if not self.closed_history:
            return False, 0.0

        times = list(self.closed_history)
        closed_time = 0.0

        # accumulate time eyes were closed
        for i in range(1, len(times)):
            dt = times[i][0] - times[i-1][0]
            if times[i-1][1]:
                closed_time += dt

        # ensure we have a valid window span
        window_span = max(self.perclos_window, t_now - times[0][0])

        # calculate perclos
        perclos = closed_time / window_span

        # determine if user is tired
        tired = perclos >= self.perclos_thresh
        return tired, perclos


    def _update_nod(self, t_now: float, pitch: float | None):
        if pitch is None:
            return False
        self.pitch_history.append(pitch)
        smoothed_pitch = float(np.mean(self.pitch_history))
        nod_completed = False
        if self.nod_state == "idle":
            if smoothed_pitch <= self.nod_down_thresh:
                self.nod_state = "down"
        elif self.nod_state == "down":
            if smoothed_pitch >= self.nod_up_thresh:
                self.nod_state = "idle"
                self.nod_times.append(t_now)
                nod_completed = True
        # purge old nods
        while self.nod_times and (t_now - self.nod_times[0] > self.nod_window_seconds):
            self.nod_times.popleft()
        return nod_completed

    def eval_scores(self, t_now, ear_score, gaze_score, head_roll, head_pitch, head_yaw):
        dt = self._update_timer(t_now)
        # EAR
        asleep = False
        if ear_score is not None and ear_score < self.ear_thresh:
            self.ear_low_acc += dt
            if self.ear_low_acc >= self.ear_time_thresh:
                asleep = True
        else:
            self.ear_low_acc = max(0.0, self.ear_low_acc * self.decay_factor)
        # Gaze
        looking_away = False
        if gaze_score is not None and gaze_score > self.gaze_thresh:
            self.gaze_high_acc += dt
            if self.gaze_high_acc >= self.gaze_time_thresh:
                looking_away = True
        else:
            self.gaze_high_acc = max(0.0, self.gaze_high_acc * self.decay_factor)
        # Pose
        distracted = False
        pose_bad = False
        for v, th in zip([head_roll, head_pitch, head_yaw], [self.roll_thresh, self.pitch_thresh, self.yaw_thresh]):
            if v is not None and abs(float(v)) > th:
                pose_bad = True
                break
        if pose_bad:
            self.pose_bad_acc += dt
            if self.pose_bad_acc >= self.pose_time_thresh:
                distracted = True
        else:
            self.pose_bad_acc = max(0.0, self.pose_bad_acc * self.decay_factor)
        # Nod detection (from pitch)
        nod_completed = self._update_nod(t_now, float(head_pitch) if head_pitch is not None else None)
        nods_recent = len(self.nod_times)
        drowsy_by_nods = nods_recent >= self.nods_for_drowsy
        return asleep, looking_away, distracted, nod_completed, drowsy_by_nods, nods_recent
