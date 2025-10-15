# metrics/blinks.py
import time
from collections import deque

class BlinkAnalyzer:
    """
    Analyzes blink patterns for drowsiness detection
    - Detects individual blinks using FSM
    - Calculates blink rate over time
    - Detects prolonged eye closure indicative of drowsiness
    """
    
    def __init__(self, config=None):
        """
        Initialize Blink Analyzer with configuration
        
        Args:
            config: Configuration dictionary with blink parameters
        """
        self.config = config or {}
        
        # Get configuration values with defaults
        eyes_config = self.config.get('eyes', {})
        blinks_config = self.config.get('blinks', {})
        
        self.ear_threshold = eyes_config.get('ear_thresh', 0.2)
        self.blink_duration = blinks_config.get('max_duration_s', 0.3)
        self.no_blink_threshold = eyes_config.get('ear_time_thresh_s', 5.0)
        self.min_separation = blinks_config.get('min_separation_s', 0.2)
        
        # FSM states
        self.STATE_OPEN = 0
        self.STATE_CLOSED = 1
        self.current_state = self.STATE_OPEN
        
        # Tracking
        self.closure_start_time = None
        self.last_blink_time = time.time()  # Initialize with current time
        self.blink_history = deque()  # (timestamp, duration)
        self.last_open_time = time.time()
        
    def update(self, ear_value, timestamp=None):
        """
        Update with new EAR value and analyze blink patterns
        
        Args:
            ear_value: Current Eye Aspect Ratio value
            timestamp: Current timestamp (optional)
            
        Returns:
            dict: Blink analysis results and metrics
        """
        if timestamp is None:
            timestamp = time.time()
            
        is_closed = ear_value < self.ear_threshold
        
        # FSM logic for blink detection
        if self.current_state == self.STATE_OPEN:
            if is_closed:
                # Transition to closed - blink start
                self.current_state = self.STATE_CLOSED
                self.closure_start_time = timestamp
                
        elif self.current_state == self.STATE_CLOSED:
            if not is_closed:
                # Transition to open - blink end
                self.current_state = self.STATE_OPEN
                blink_duration = timestamp - self.closure_start_time
                
                # Valid blink detected (reasonable duration and sufficient separation)
                if (0.1 <= blink_duration <= 1.0 and 
                    timestamp - self.last_blink_time >= self.min_separation):
                    self.blink_history.append((timestamp, blink_duration))
                    self.last_blink_time = timestamp
                    self._clean_old_blinks(timestamp)
        
        # Track time since last blink - cap at a reasonable maximum
        time_since_last_blink = timestamp - self.last_blink_time
        max_reasonable_time = 60.0  # 1 minute maximum
        time_since_last_blink = min(time_since_last_blink, max_reasonable_time)
        
        # Update last open time
        if not is_closed:
            self.last_open_time = timestamp
        
        # Calculate metrics
        blink_rate = self._calculate_blink_rate(timestamp)
        no_blink_too_long = time_since_last_blink > self.no_blink_threshold
        
        return {
            'current_state': 'CLOSED' if self.current_state == self.STATE_CLOSED else 'OPEN',
            'is_eyes_closed': is_closed,
            'time_since_last_blink': time_since_last_blink,
            'blink_rate_per_min': blink_rate,
            'total_blinks_detected': len(self.blink_history),
            'no_blink_too_long': no_blink_too_long,
            'is_drowsy': no_blink_too_long,  # Prolonged no blinking indicates drowsiness
            'ear_threshold': self.ear_threshold,
            'no_blink_threshold': self.no_blink_threshold
        }
    
    def _clean_old_blinks(self, current_time):
        """
        Remove blinks older than 2 minutes for rate calculation
        
        Args:
            current_time: Current timestamp for age calculation
        """
        two_minutes_ago = current_time - 120
        while self.blink_history and self.blink_history[0][0] < two_minutes_ago:
            self.blink_history.popleft()
    
    def _calculate_blink_rate(self, current_time):
        """
        Calculate blink rate per minute in last 2 minutes
        
        Args:
            current_time: Current timestamp for rate calculation
            
        Returns:
            float: Blink rate in blinks per minute
        """
        if not self.blink_history:
            return 0
            
        # Count blinks in last 2 minutes
        two_minutes_ago = current_time - 120
        recent_blinks = sum(1 for timestamp, _ in self.blink_history if timestamp >= two_minutes_ago)
        
        # Convert to per minute rate
        return (recent_blinks / 2.0) if recent_blinks > 0 else 0
    
    def reset(self):
        """Reset the analyzer state and clear history"""
        self.current_state = self.STATE_OPEN
        self.closure_start_time = None
        self.last_blink_time = time.time()  # Reset to current time
        self.blink_history.clear()
        self.last_open_time = time.time()
        
    def update_config(self, new_config):
        """
        Update configuration parameters
        
        Args:
            new_config: New configuration dictionary
        """
        if 'eyes' in new_config:
            eyes_config = new_config['eyes']
            blinks_config = new_config.get('blinks', {})
            
            self.ear_threshold = eyes_config.get('ear_thresh', self.ear_threshold)
            self.no_blink_threshold = eyes_config.get('ear_time_thresh_s', self.no_blink_threshold)
            self.blink_duration = blinks_config.get('max_duration_s', self.blink_duration)
            self.min_separation = blinks_config.get('min_separation_s', self.min_separation)