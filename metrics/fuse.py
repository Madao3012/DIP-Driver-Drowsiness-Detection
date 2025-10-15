# metrics/fuse.py
import time

class AttentionScorer:
    """
    Fuses multiple drowsiness metrics with weighted scoring
    Calculates composite risk score and overall attention level
    Provides configurable alert thresholds and cooldown mechanism
    """
    
    def __init__(self, config=None):
        """
        Initialize Attention Scorer with configuration
        
        Args:
            config: Configuration dictionary with fusion parameters
        """
        self.config = config or {}
        
        # Get configuration values with defaults
        fusion_config = self.config.get('fusion', {})
        alerts_config = self.config.get('alerts', {})
        
        # Weight configuration - higher weight = more important for drowsiness
        self.weights = fusion_config.get('weights', {
            'perclos': 0.35,      # PERCLOS is strong indicator
            'nods': 0.25,         # Head nodding
            'blinks': 0.15,       # Blink patterns
            'yawns': 0.25         # Yawning frequency
        })
        
        # Alert thresholds from config
        self.critical_threshold = alerts_config.get('critical_threshold', 70)
        self.high_threshold = alerts_config.get('high_threshold', 60)
        self.medium_threshold = alerts_config.get('medium_threshold', 40)
        self.low_threshold = alerts_config.get('low_threshold', 20)
        
        # Risk scoring
        self.risk_scores = {
            'perclos': 0,
            'nods': 0, 
            'blinks': 0,
            'yawns': 0
        }
        
        self.overall_risk = 0
        self.attention_level = 100  # 100% = fully attentive
        
        # Alert tracking
        self.last_alert_time = 0
        self.alert_cooldown = alerts_config.get('cooldown_s', 10)  # seconds between alerts
        
    def update(self, perclos_data, nod_data, blink_data, yawn_data, timestamp=None):
        """
        Update with all metric data and calculate composite scores
        
        Args:
            perclos_data: PERCLOS metric data
            nod_data: Head nod metric data
            blink_data: Blink pattern metric data
            yawn_data: Yawn metric data
            timestamp: Current timestamp (optional)
            
        Returns:
            dict: Comprehensive attention and risk assessment
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Calculate individual risk scores (0-100)
        self.risk_scores['perclos'] = self._calculate_perclos_risk(perclos_data)
        self.risk_scores['nods'] = self._calculate_nod_risk(nod_data)
        self.risk_scores['blinks'] = self._calculate_blink_risk(blink_data) 
        self.risk_scores['yawns'] = self._calculate_yawn_risk(yawn_data)
        
        # Calculate weighted overall risk
        self.overall_risk = self._calculate_weighted_risk()
        
        # Convert to attention level (inverse of risk)
        self.attention_level = max(0, 100 - self.overall_risk)
        
        # Determine if alert should be triggered
        should_alert = self._should_trigger_alert(timestamp)
        
        return {
            'overall_risk': self.overall_risk,
            'attention_level': self.attention_level,
            'risk_breakdown': self.risk_scores.copy(),
            'weighted_components': self._get_weighted_components(),
            'should_alert': should_alert,
            'alert_level': self._get_alert_level(),
            'is_critical': self.overall_risk >= self.critical_threshold
        }
    
    def _calculate_perclos_risk(self, perclos_data):
        """
        Calculate risk from PERCLOS (0-100)
        
        Args:
            perclos_data: PERCLOS metric data
            
        Returns:
            int: PERCLOS risk score (0-100)
        """
        perclos_value = perclos_data.get('perclos', 0)
        
        if perclos_value <= 0:
            return 0
        elif perclos_value >= 0.7:  # 70% eye closure is critical
            return 100
        else:
            return int((perclos_value / 0.7) * 100)
    
    def _calculate_nod_risk(self, nod_data):
        """
        Calculate risk from nodding (0-100)
        
        Args:
            nod_data: Head nod metric data
            
        Returns:
            int: Nod risk score (0-100)
        """
        nods_per_min = nod_data.get('nods_last_minute', 0)
        
        if nods_per_min <= 0:
            return 0
        elif nods_per_min >= 3:  # 3+ nods per minute is critical
            return 100
        else:
            return int((nods_per_min / 3.0) * 100)
    
    def _calculate_blink_risk(self, blink_data):
        """
        Calculate risk from blink patterns (0-100)
        
        Args:
            blink_data: Blink pattern metric data
            
        Returns:
            int: Blink risk score (0-100)
        """
        no_blink_too_long = blink_data.get('no_blink_too_long', False)
        time_since_blink = blink_data.get('time_since_last_blink', 0)
        
        # Handle cases where time_since_blink might be very large
        if time_since_blink > 1000:  # Sanity check for unreasonable values
            time_since_blink = 6  # Treat as just above threshold
        
        if no_blink_too_long:
            # Scale risk based on how long since last blink
            excess_time = max(0, time_since_blink - 5.0)  # 5 seconds is threshold
            max_excess = 10  # Maximum excess time to consider (10 seconds beyond threshold)
            risk_value = min(100, int((excess_time / max_excess) * 100))
            return risk_value
        else:
            return 0
    
    def _calculate_yawn_risk(self, yawn_data):
        """
        Calculate risk from yawning (0-100)
        
        Args:
            yawn_data: Yawn metric data
            
        Returns:
            int: Yawn risk score (0-100)
        """
        yawns_per_min = yawn_data.get('yawns_last_minute', 0)
        
        if yawns_per_min <= 0:
            return 0
        elif yawns_per_min >= 2:  # 2+ yawns per minute is critical
            return 100
        else:
            return int((yawns_per_min / 2.0) * 100)
    
    def _calculate_weighted_risk(self):
        """
        Calculate weighted overall risk score
        
        Returns:
            float: Weighted overall risk score
        """
        weighted_sum = 0
        total_weight = 0
        
        for metric, risk in self.risk_scores.items():
            weight = self.weights.get(metric, 0)
            weighted_sum += risk * weight
            total_weight += weight
        
        return weighted_sum if total_weight > 0 else 0
    
    def _get_weighted_components(self):
        """
        Get individual weighted contributions to overall risk
        
        Returns:
            dict: Weighted risk contributions per metric
        """
        components = {}
        for metric, risk in self.risk_scores.items():
            components[metric] = risk * self.weights.get(metric, 0)
        return components
    
    def _should_trigger_alert(self, current_time):
        """
        Determine if alert should be triggered based on cooldown and risk level
        
        Args:
            current_time: Current timestamp for cooldown check
            
        Returns:
            bool: True if alert should be triggered
        """
        if self.overall_risk < self.high_threshold:  # Only alert for high risk
            return False
            
        if current_time - self.last_alert_time >= self.alert_cooldown:
            self.last_alert_time = current_time
            return True
            
        return False
    
    def _get_alert_level(self):
        """
        Get alert level based on overall risk
        
        Returns:
            str: Alert level description
        """
        if self.overall_risk >= self.critical_threshold:
            return "CRITICAL"
        elif self.overall_risk >= self.high_threshold:
            return "HIGH" 
        elif self.overall_risk >= self.medium_threshold:
            return "MEDIUM"
        elif self.overall_risk >= self.low_threshold:
            return "LOW"
        else:
            return "NORMAL"
    
    def update_weights(self, new_weights):
        """
        Update the weight configuration
        
        Args:
            new_weights: New weight dictionary
        """
        self.weights.update(new_weights)
    
    def update_config(self, new_config):
        """
        Update configuration parameters
        
        Args:
            new_config: New configuration dictionary
        """
        if 'fusion' in new_config:
            fusion_config = new_config['fusion']
            self.weights = fusion_config.get('weights', self.weights)
            
        if 'alerts' in new_config:
            alerts_config = new_config['alerts']
            self.critical_threshold = alerts_config.get('critical_threshold', self.critical_threshold)
            self.high_threshold = alerts_config.get('high_threshold', self.high_threshold)
            self.medium_threshold = alerts_config.get('medium_threshold', self.medium_threshold)
            self.low_threshold = alerts_config.get('low_threshold', self.low_threshold)
            self.alert_cooldown = alerts_config.get('cooldown_s', self.alert_cooldown)
    
    def reset(self):
        """Reset all scores and state"""
        for metric in self.risk_scores:
            self.risk_scores[metric] = 0
        self.overall_risk = 0
        self.attention_level = 100
        self.last_alert_time = 0