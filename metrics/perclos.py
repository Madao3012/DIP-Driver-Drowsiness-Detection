# metrics/perclos.py
import time
from collections import deque

class PERCLOSCalculator:
    """
    PERCLOS (Percentage of Eye Closure) calculator
    Measures the percentage of time eyes are closed over a rolling window
    Industry-standard metric for drowsiness detection
    """
    
    def __init__(self, config=None):
        """
        Initialize PERCLOS calculator with configuration
        
        Args:
            config: Configuration dictionary with PERCLOS parameters
        """
        self.config = config or {}
        
        # Get configuration values with defaults
        perclos_config = self.config.get('perclos', {})
        self.window_seconds = perclos_config.get('window_s', 30)
        self.ear_threshold = perclos_config.get('closed_ear_thresh', 0.2)
        self.closure_threshold = perclos_config.get('tired_thresh', 0.7)
        
        # State tracking
        self.eye_states = deque()  # (timestamp, is_closed)
        self.frame_count = 0
        self.last_update_time = time.time()
        
    def update(self, ear_value, timestamp=None):
        """
        Update with new EAR value and calculate PERCLOS
        
        Args:
            ear_value: Current Eye Aspect Ratio value
            timestamp: Current timestamp (optional)
            
        Returns:
            dict: PERCLOS results and drowsiness status
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Determine if eyes are closed based on EAR threshold
        is_closed = ear_value < self.ear_threshold
        
        # Add new state
        self.eye_states.append((timestamp, is_closed))
        self.frame_count += 1
        
        # Remove old states outside window
        self._clean_old_states(timestamp)
        
        # Calculate PERCLOS
        perclos = self._calculate_perclos()
        is_drowsy = perclos > self.closure_threshold
        
        return {
            'perclos': perclos,
            'is_drowsy': is_drowsy,
            'window_closure_ratio': perclos,
            'frames_in_window': len(self.eye_states),
            'closed_frames': sum(1 for _, closed in self.eye_states if closed),
            'ear_threshold': self.ear_threshold,
            'closure_threshold': self.closure_threshold
        }
    
    def _clean_old_states(self, current_time):
        """
        Remove states older than the configured window
        
        Args:
            current_time: Current timestamp for age calculation
        """
        cutoff_time = current_time - self.window_seconds
        while self.eye_states and self.eye_states[0][0] < cutoff_time:
            self.eye_states.popleft()
    
    def _calculate_perclos(self):
        """
        Calculate PERCLOS as ratio of closed frames to total frames in window
        
        Returns:
            float: PERCLOS value between 0.0 and 1.0
        """
        if not self.eye_states:
            return 0.0
            
        closed_count = sum(1 for _, closed in self.eye_states if closed)
        total_count = len(self.eye_states)
        
        return closed_count / total_count if total_count > 0 else 0.0
    
    def reset(self):
        """Reset the calculator state and clear history"""
        self.eye_states.clear()
        self.frame_count = 0
        self.last_update_time = time.time()
        
    def update_config(self, new_config):
        """
        Update configuration parameters
        
        Args:
            new_config: New configuration dictionary
        """
        if 'perclos' in new_config:
            perclos_config = new_config['perclos']
            self.window_seconds = perclos_config.get('window_s', self.window_seconds)
            self.ear_threshold = perclos_config.get('closed_ear_thresh', self.ear_threshold)
            self.closure_threshold = perclos_config.get('tired_thresh', self.closure_threshold)