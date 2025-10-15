# metrics/nods.py
import time
from collections import deque

class NodDetector:
    """
    Detects head nodding (looking down) patterns indicative of drowsiness
    Uses a Finite State Machine (FSM) to track pitch movements
    """
    
    def __init__(self, config=None):
        """
        Initialize Nod Detector with configuration
        
        Args:
            config: Configuration dictionary with nod parameters
        """
        self.config = config or {}
        
        # Get configuration values with defaults
        nods_config = self.config.get('nods', {})
        pose_config = self.config.get('pose', {})
        
        self.pitch_threshold = -nods_config.get('down_thresh_deg', 15)  # Negative for looking down
        self.nod_duration = pose_config.get('hold_time_s', 2.0)
        self.nods_per_minute = nods_config.get('count_for_drowsy', 3)
        
        # FSM states
        self.STATE_FORWARD = 0
        self.STATE_LOOKING_DOWN = 1
        self.current_state = self.STATE_FORWARD
        
        # Tracking
        self.nod_start_time = None
        self.nod_history = deque()  # (end_time, duration)
        self.current_nod_duration = 0
        
    def update(self, pitch_value, timestamp=None):
        """
        Update with new pitch value and detect nodding
        
        Args:
            pitch_value: Current head pitch angle in degrees
            timestamp: Current timestamp (optional)
            
        Returns:
            dict: Nod detection results and metrics
        """
        if timestamp is None:
            timestamp = time.time()
            
        is_looking_down = pitch_value < self.pitch_threshold
        
        # FSM logic for nod detection
        if self.current_state == self.STATE_FORWARD:
            if is_looking_down:
                # Transition to looking down
                self.current_state = self.STATE_LOOKING_DOWN
                self.nod_start_time = timestamp
                
        elif self.current_state == self.STATE_LOOKING_DOWN:
            if not is_looking_down:
                # Transition back to forward - nod completed
                self.current_state = self.STATE_FORWARD
                nod_duration = timestamp - self.nod_start_time
                
                if nod_duration >= self.nod_duration:
                    # Valid nod detected
                    self.nod_history.append((timestamp, nod_duration))
                    self._clean_old_nods(timestamp)
                
            else:
                # Still looking down - update duration
                self.current_nod_duration = timestamp - self.nod_start_time
        
        # Clean old nods
        self._clean_old_nods(timestamp)
        
        # Calculate nods per minute
        nods_per_min = self._calculate_nods_per_minute(timestamp)
        is_drowsy = nods_per_min >= self.nods_per_minute
        
        return {
            'current_state': 'LOOKING_DOWN' if self.current_state == self.STATE_LOOKING_DOWN else 'FORWARD',
            'current_nod_duration': self.current_nod_duration if self.current_state == self.STATE_LOOKING_DOWN else 0,
            'nods_last_minute': nods_per_min,
            'total_nods_detected': len(self.nod_history),
            'is_drowsy': is_drowsy,
            'is_looking_down': is_looking_down,
            'pitch_threshold': self.pitch_threshold,
            'nod_duration_threshold': self.nod_duration
        }
    
    def _clean_old_nods(self, current_time):
        """
        Remove nods older than 1 minute for rate calculation
        
        Args:
            current_time: Current timestamp for age calculation
        """
        one_minute_ago = current_time - 60
        while self.nod_history and self.nod_history[0][0] < one_minute_ago:
            self.nod_history.popleft()
    
    def _calculate_nods_per_minute(self, current_time):
        """
        Calculate nods per minute in the last 60 seconds
        
        Args:
            current_time: Current timestamp for rate calculation
            
        Returns:
            int: Number of nods in the last minute
        """
        if not self.nod_history:
            return 0
            
        # Count nods in last 60 seconds
        one_minute_ago = current_time - 60
        recent_nods = sum(1 for end_time, _ in self.nod_history if end_time >= one_minute_ago)
        
        return recent_nods
    
    def reset(self):
        """Reset the detector state and clear history"""
        self.current_state = self.STATE_FORWARD
        self.nod_start_time = None
        self.nod_history.clear()
        self.current_nod_duration = 0
        
    def update_config(self, new_config):
        """
        Update configuration parameters
        
        Args:
            new_config: New configuration dictionary
        """
        if 'nods' in new_config:
            nods_config = new_config['nods']
            pose_config = new_config.get('pose', {})
            
            self.pitch_threshold = -nods_config.get('down_thresh_deg', -self.pitch_threshold)
            self.nod_duration = pose_config.get('hold_time_s', self.nod_duration)
            self.nods_per_minute = nods_config.get('count_for_drowsy', self.nods_per_minute)