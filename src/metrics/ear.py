# ear.py
# Lightweight EAR utilities optimized for Raspberry Pi + MediaPipe Face Mesh.

from math import hypot
from typing import Iterable, List, Sequence, Tuple, Optional

# MediaPipe Face Mesh indices for a 6-point EAR (with refine_landmarks=True)
# Ordering p1..p6: [outer_corner, upper1, upper2, inner_corner, lower2, lower1]
LEFT_EYE:  List[int] = [33, 160, 158, 133, 153, 144]
RIGHT_EYE: List[int] = [263, 387, 385, 362, 380, 373]

Point = Tuple[int, int]

def _dist(a: Point, b: Point) -> float:
    """Fast Euclidean distance using math.hypot on integer pixel coords."""
    return hypot(a[0] - b[0], a[1] - b[1])

def ear_from_pts(pts: Sequence[Point]) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for one eye from 6 ordered points in pixel coords.
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    pts: [(x1,y1),...,(x6,y6)] ordered p1..p6
    Returns: EAR as float (typical open ~0.25–0.30; closed ~0.05–0.10).
    """
    if len(pts) != 6:
        return 0.0  # defensive
    p1, p2, p3, p4, p5, p6 = pts
    v1 = _dist(p2, p6)
    v2 = _dist(p3, p5)
    h  = _dist(p1, p4)
    return 0.0 if h == 0.0 else (v1 + v2) / (2.0 * h)

def to_xy(landmarks, frame_w: int, frame_h: int, idxs: Iterable[int]) -> List[Point]:
    """
    Convert MediaPipe normalized landmarks (0..1) to integer pixel coords for given indices.
    Keeps arithmetic minimal for Raspberry Pi performance.
    """
    w = frame_w
    h = frame_h
    out: List[Point] = []
    append = out.append  # local binding micro-opt
    for i in idxs:
        li = landmarks[i]
        # clamp just in case landmarks are slightly outside [0,1]
        x = li.x
        y = li.y
        if x < 0.0: x = 0.0
        elif x > 1.0: x = 1.0
        if y < 0.0: y = 0.0
        elif y > 1.0: y = 1.0
        append((int(x * w), int(y * h)))
    return out

class EARState:
    """
    Maintains smoothed EAR (EMA) and a consecutive-frames debounce for CLOSED state.
    Designed to be lightweight for Raspberry Pi:
      - __slots__ to reduce per-instance memory
      - Optional None handling in update() for frames where EAR couldn't be computed
    """
    __slots__ = ("thresh", "alpha", "closed_needed", "_ema", "_closed_run")

    def __init__(self, thresh: float = 0.21, ema_alpha: float = 0.7, closed_needed: int = 3) -> None:
        self.thresh = float(thresh)
        self.alpha = float(ema_alpha)
        self.closed_needed = int(closed_needed)
        self._ema: Optional[float] = None
        self._closed_run: int = 0

    def reset(self) -> None:
        """Reset smoothing and debounce counters (e.g., after long tracking loss)."""
        self._ema = None
        self._closed_run = 0

    def update(self, ear_value: Optional[float]) -> Tuple[Optional[float], bool]:
        """
        Update state with a new EAR sample.
        - If ear_value is None, keep previous EMA and do not advance the closed counter.
        Returns: (ema_ear or None if uninitialized, closed_bool)
        """
        # Handle missing measurement gracefully
        if ear_value is None:
            return self._ema, (self._closed_run >= self.closed_needed)

        # EMA smoothing
        ema = self._ema
        a = self.alpha
        if ema is None:
            ema = ear_value
        else:
            ema = a * ema + (1.0 - a) * ear_value
        self._ema = ema

        # Debounce below threshold
        if ema < self.thresh:
            self._closed_run += 1
        else:
            self._closed_run = 0

        return ema, (self._closed_run >= self.closed_needed)
