# metrics/yawn.py
import time
from collections import deque

class YawnDetector:
    """
    Detects yawning patterns for drowsiness detection
    - Tracks mouth open duration using FSM
    - Counts yawns per time period
    - Uses MAR (Mouth Aspect Ratio) for detection
    """
    
    def __init__(self, config=None):
        """
        Initialize Yawn Detector with configuration
        
        Args:
            config: Configuration dictionary with yawn parameters
        """
        self.config = config or {}
        
        # Get configuration values with defaults
        mouth_config = self.config.get('mouth', {})
        
        self.mar_threshold = mouth_config.get('mar_thresh', 0.7)
        self.yawn_duration = mouth_config.get('yawn_min_duration_s', 2.0)
        self.yawns_per_minute = 2  # Could be added to config if needed
        
        # FSM states
        self.STATE_CLOSED = 0
        self.STATE_OPEN = 1
        self.current_state = self.STATE_CLOSED
        
        # Tracking
        self.yawn_start_time = None
        self.yawn_history = deque()  # (end_time, duration)
        self.current_yawn_duration = 0
        
    def update(self, mar_value, timestamp=None):
        """
        Update with new MAR value and detect yawning
        
        Args:
            mar_value: Current Mouth Aspect Ratio value
            timestamp: Current timestamp (optional)
            
        Returns:
            dict: Yawn detection results and metrics
        """
        if timestamp is None:
            timestamp = time.time()
            
        is_mouth_open = mar_value > self.mar_threshold
        
        # FSM logic for yawn detection
        if self.current_state == self.STATE_CLOSED:
            if is_mouth_open:
                # Transition to mouth open
                self.current_state = self.STATE_OPEN
                self.yawn_start_time = timestamp
                
        elif self.current_state == self.STATE_OPEN:
            if not is_mouth_open:
                # Transition to closed - potential yawn completed
                self.current_state = self.STATE_CLOSED
                yawn_duration = timestamp - self.yawn_start_time
                
                if yawn_duration >= self.yawn_duration:
                    # Valid yawn detected
                    self.yawn_history.append((timestamp, yawn_duration))
                    self._clean_old_yawns(timestamp)
                
            else:
                # Still mouth open - update duration
                self.current_yawn_duration = timestamp - self.yawn_start_time
        
        # Clean old yawns
        self._clean_old_yawns(timestamp)
        
        # Calculate yawns per minute
        yawns_per_min = self._calculate_yawns_per_minute(timestamp)
        is_drowsy = yawns_per_min >= self.yawns_per_minute
        
        return {
            'current_state': 'OPEN' if self.current_state == self.STATE_OPEN else 'CLOSED',
            'current_yawn_duration': self.current_yawn_duration if self.current_state == self.STATE_OPEN else 0,
            'yawns_last_minute': yawns_per_min,
            'total_yawns_detected': len(self.yawn_history),
            'is_drowsy': is_drowsy,
            'is_mouth_open': is_mouth_open,
            'mar_threshold': self.mar_threshold,
            'yawn_duration_threshold': self.yawn_duration
        }
    
    def _clean_old_yawns(self, current_time):
        """
        Remove yawns older than 1 minute for rate calculation
        
        Args:
            current_time: Current timestamp for age calculation
        """
        one_minute_ago = current_time - 60
        while self.yawn_history and self.yawn_history[0][0] < one_minute_ago:
            self.yawn_history.popleft()
    
    def _calculate_yawns_per_minute(self, current_time):
        """
        Calculate yawns per minute in the last 60 seconds
        
        Args:
            current_time: Current timestamp for rate calculation
            
        Returns:
            int: Number of yawns in the last minute
        """
        if not self.yawn_history:
            return 0
            
        # Count yawns in last 60 seconds
        one_minute_ago = current_time - 60
        recent_yawns = sum(1 for end_time, _ in self.yawn_history if end_time >= one_minute_ago)
        
        return recent_yawns
    
    def reset(self):
        """Reset the detector state and clear history"""
        self.current_state = self.STATE_CLOSED
        self.yawn_start_time = None
        self.yawn_history.clear()
        self.current_yawn_duration = 0
        
    def update_config(self, new_config):
        """
        Update configuration parameters
        
        Args:
            new_config: New configuration dictionary
        """
        if 'mouth' in new_config:
            mouth_config = new_config['mouth']
            self.mar_threshold = mouth_config.get('mar_thresh', self.mar_threshold)
            self.yawn_duration = mouth_config.get('yawn_min_duration_s', self.yawn_duration)