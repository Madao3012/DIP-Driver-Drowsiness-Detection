# yawn_detector.py — class-based MAR (mouth aspect ratio) with temporal logic
import numpy as np
from collections import deque

# Landmarks (MediaPipe Face Mesh indices)
# Using inner-lip vertical pair and mouth corners for width
MOUTH_INNER_TOP = [13, 14]
MOUTH_INNER_BOTTOM = [16, 17]
MOUTH_CORNERS = [61, 291]

class YawnDetector:
    def __init__(self, mar_thresh: float = 0.60, min_duration: float = 0.5):
        self.mar_thresh = mar_thresh
        self.min_duration = min_duration
        self.over_thresh_time = 0.0

    @staticmethod
    def _euclid(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def compute_mar(self, lms_norm_xy, frame_size):
        """Compute MAR from normalized landmarks (N x 2) mapped to pixel coords."""
        w, h = frame_size
        def to_xy(idx):
            x = float(np.clip(lms_norm_xy[idx,0], 0.0, 1.0) * w)
            y = float(np.clip(lms_norm_xy[idx,1], 0.0, 1.0) * h)
            return np.array([x, y], dtype=np.float32)
        # vertical distance: average of two inner pairs
        vt = self._euclid(to_xy(MOUTH_INNER_TOP[0]), to_xy(MOUTH_INNER_BOTTOM[0]))
        vb = self._euclid(to_xy(MOUTH_INNER_TOP[1]), to_xy(MOUTH_INNER_BOTTOM[1]))
        v = 0.5 * (vt + vb)
        # width
        wmouth = self._euclid(to_xy(MOUTH_CORNERS[0]), to_xy(MOUTH_CORNERS[1])) + 1e-6
        return v / wmouth

    def update(self, dt: float, mar_value: float):
        """Update temporal logic; returns (is_yawning_now, yawn_event_completed)."""
        is_open = mar_value >= self.mar_thresh
        yawn_event = False
        if is_open:
            self.over_thresh_time += dt
        else:
            if self.over_thresh_time >= self.min_duration:
                yawn_event = True
            self.over_thresh_time = 0.0
        return is_open, yawn_event
