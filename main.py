# main.py - Complete Drowsiness Detection System
import cv2
import numpy as np
import time
import yaml
import os
import threading
import pygame
from vision.eyes import EyeDetector
from vision.mouth import MouthDetector
from vision.pose import PoseDetector
from metrics.perclos import PERCLOSCalculator
from metrics.nods import NodDetector
from metrics.blinks import BlinkAnalyzer
from metrics.yawn import YawnDetector
from metrics.fuse import AttentionScorer
from data_logger import DataLogger


file_path = r"C:\Users\Gagana PC\Desktop\DIP\DIP-Driver-Drowsiness-Detection\siren-alert-96052.mp3"
 

class DrowsinessDetectionSystem:
    """
    Complete drowsiness detection system integrating all vision and metric components
    Provides real-time monitoring with configurable parameters and professional dashboard
    """
    
    def __init__(self, config_path="configs/default.yaml"):
        """
        Initialize the complete drowsiness detection system
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize vision detectors with config
        self.eye_detector = EyeDetector(self.config)
        self.mouth_detector = MouthDetector(self.config)
        self.pose_detector = PoseDetector(self.config)
        
        # Initialize metrics calculators with config
        self.perclos = PERCLOSCalculator(self.config)
        self.nods = NodDetector(self.config)
        self.blinks = BlinkAnalyzer(self.config)
        self.yawns = YawnDetector(self.config)
        self.fuser = AttentionScorer(self.config)
        
        # System state
        self.frame_count = 0
        self.last_alert_time = 0
        self.alert_cooldown = self.config['alerts']['cooldown_s']
        
    def load_config(self, config_path):
        """
        Load configuration from YAML file with fallback defaults
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using defaults.")
            return self.get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return self.get_default_config()
    
    def get_default_config(self):
        """
        Provide default configuration if file loading fails
        
        Returns:
            dict: Default configuration values
        """
        return {
            'perclos': {'window_s': 30, 'closed_ear_thresh': 0.2, 'tired_thresh': 0.7},
            'nods': {'down_thresh_deg': 15, 'count_for_drowsy': 3},
            'eyes': {'ear_thresh': 0.2, 'ear_time_thresh_s': 5.0},
            'mouth': {'mar_thresh': 0.7, 'yawn_min_duration_s': 2.0},
            'pose': {'hold_time_s': 2.0},
            'blinks': {'max_duration_s': 0.4},
            'fusion': {'weights': {'perclos': 0.35, 'nods': 0.25, 'blinks': 0.15, 'yawns': 0.25}},
            'alerts': {'cooldown_s': 10, 'critical_threshold': 70, 'high_threshold': 60},
            'camera': {'width': 1280, 'height': 720, 'mirror_effect': True},
            'display': {
                'dashboard_width': 400, 
                'show_landmarks': False, 
                'show_values': True, 
                'show_fps': True,
                'colors': {
                    'background': [45, 45, 45],
                    'text_primary': [255, 255, 255],
                    'text_secondary': [200, 200, 200],
                    'attention_good': [0, 200, 0],
                    'attention_warning': [0, 200, 200],
                    'attention_critical': [0, 0, 255],
                    'metric_good': [100, 255, 100],
                    'metric_warning': [255, 255, 100],
                    'metric_critical': [255, 100, 100],
                    'separator': [100, 100, 100]
                }
            }
        }
    
    def process_frame(self, frame):
        """
        Process a single frame through the complete system pipeline
        
        Args:
            frame: Input camera frame
            
        Returns:
            tuple: (results_dict, display_frame)
        """
        timestamp = time.time()
        self.frame_count += 1
        
        # Apply mirror effect if configured
        if self.config['camera']['mirror_effect']:
            display_frame = cv2.flip(frame, 1)
        else:
            display_frame = frame.copy()
        
        # Get raw detection data from vision modules
        eye_data = self.eye_detector.process_frame(display_frame)
        mouth_data = self.mouth_detector.process_frame(display_frame)
        pose_data = self.pose_detector.process_frame(display_frame)
        
        # Update metrics with current frame data
        perclos_result = self.perclos.update(
            eye_data['avg_ear'] if eye_data['eyes_detected'] else 0.3,
            timestamp
        )
        
        nod_result = self.nods.update(
            pose_data['pitch'] if pose_data['pose_detected'] else 0,
            timestamp
        )
        
        blink_result = self.blinks.update(
            eye_data['avg_ear'] if eye_data['eyes_detected'] else 0.3,
            timestamp
        )
        
        yawn_result = self.yawns.update(
            mouth_data['mar'] if (mouth_data['mouth_detected'] and mouth_data['mouth_visible']) else 0.3,
            timestamp
        )
        
        # Fuse all metrics for overall attention score
        attention_result = self.fuser.update(
            perclos_result, nod_result, blink_result, yawn_result, timestamp
        )
        
        # Prepare comprehensive results
        results = {
            'vision_data': {
                'eyes': eye_data,
                'mouth': mouth_data,
                'pose': pose_data
            },
            'metrics_data': {
                'perclos': perclos_result,
                'nods': nod_result,
                'blinks': blink_result,
                'yawns': yawn_result
            },
            'attention_result': attention_result,
            'timestamp': timestamp,
            'frame_count': self.frame_count
        }
        
        return results, display_frame
    
    def draw_dashboard(self, frame, results):
        """
        Draw professional dashboard with all metrics and status information
        
        Args:
            frame: Frame to use for dimensions
            results: Processing results from system
            
        Returns:
            numpy.ndarray: Dashboard image
        """
        height, width = frame.shape[:2]
        dashboard_width = self.config['display']['dashboard_width']
        colors = self.config['display']['colors']
        
        # Create dashboard with professional background
        dashboard = np.zeros((height, dashboard_width, 3), dtype=np.uint8)
        dashboard[:] = colors['background']
        
        y_position = 40
        
        # Title Section
        cv2.putText(dashboard, "DROWSINESS DETECTION", (20, y_position),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, colors['text_primary'], 2)
        y_position += 50
        
        # Attention Level (Main Indicator)
        attention_level = results['attention_result']['attention_level']
        risk_level = results['attention_result']['overall_risk']
        alert_level = results['attention_result']['alert_level']
        
        # Color based on attention level
        if attention_level >= 70:
            color = colors['attention_good']
            status_text = "ATTENTIVE"
        elif attention_level >= 40:
            color = colors['attention_warning']
            status_text = "CAUTION"
        else:
            color = colors['attention_critical']
            status_text = "DROWSY"
        
        cv2.putText(dashboard, f"ATTENTION: {attention_level:.0f}%", 
                   (20, y_position), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        y_position += 30
        
        cv2.putText(dashboard, f"STATUS: {status_text}", 
                   (20, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y_position += 30
        
        cv2.putText(dashboard, f"RISK LEVEL: {alert_level}", 
                   (20, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text_secondary'], 1)
        y_position += 40
        
        # Separator
        cv2.line(dashboard, (10, y_position), (dashboard_width-10, y_position), colors['separator'], 2)
        y_position += 30
        
        # Direction and Pose Section
        cv2.putText(dashboard, "DIRECTION & POSE:", (20, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text_primary'], 1)
        y_position += 25
        
        vision_data = results['vision_data']
        
        # Gaze Direction
        gaze_direction = vision_data['eyes']['overall_gaze']
        gaze_color = colors['metric_good'] if gaze_direction == "CENTER" else colors['metric_warning']
        cv2.putText(dashboard, f"GAZE: {gaze_direction}", (20, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 1)
        y_position += 20
        
        # Head Orientation
        head_orientation = vision_data['pose']['head_orientation']
        head_color = colors['metric_good'] if head_orientation == "FORWARD" else colors['metric_warning'] if head_orientation == "LOOKING DOWN" else colors['metric_critical']
        cv2.putText(dashboard, f"HEAD: {head_orientation}", (20, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, head_color, 1)
        y_position += 30
        
        # Metrics Breakdown Section
        cv2.putText(dashboard, "METRICS BREAKDOWN:", (20, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text_primary'], 1)
        y_position += 30
        
        metrics = results['metrics_data']
        
        # PERCLOS
        perclos_pct = metrics['perclos']['perclos'] * 100
        perclos_color = colors['metric_good'] if perclos_pct < 50 else colors['metric_warning'] if perclos_pct < 70 else colors['metric_critical']
        cv2.putText(dashboard, f"PERCLOS: {perclos_pct:.1f}%", (20, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, perclos_color, 1)
        y_position += 25
        
        # Nods
        nods_min = metrics['nods']['nods_last_minute']
        nods_color = colors['metric_good'] if nods_min < 1 else colors['metric_warning'] if nods_min < 2 else colors['metric_critical']
        cv2.putText(dashboard, f"NODS/MIN: {nods_min}", (20, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, nods_color, 1)
        y_position += 25
        
        # Blinks
        time_since_blink = metrics['blinks']['time_since_last_blink']
        blink_color = colors['metric_good'] if time_since_blink < 3 else colors['metric_warning'] if time_since_blink < 5 else colors['metric_critical']
        cv2.putText(dashboard, f"LAST BLINK: {time_since_blink:.1f}s", (20, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_color, 1)
        y_position += 25
        
        # Yawns
        yawns_min = metrics['yawns']['yawns_last_minute']
        yawn_color = colors['metric_good'] if yawns_min < 1 else colors['metric_warning'] if yawns_min < 2 else colors['metric_critical']
        cv2.putText(dashboard, f"YAWNS/MIN: {yawns_min}", (20, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, yawn_color, 1)
        y_position += 40
        
        # Detection Status Section
        cv2.putText(dashboard, "DETECTION STATUS:", (20, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text_primary'], 1)
        y_position += 25
        
        status_eyes = "DETECTED" if vision_data['eyes']['eyes_detected'] else "NOT DETECTED"
        status_mouth = "VISIBLE" if (vision_data['mouth']['mouth_detected'] and vision_data['mouth']['mouth_visible']) else "NOT VISIBLE"
        status_pose = "DETECTED" if vision_data['pose']['pose_detected'] else "NOT DETECTED"
        
        eyes_color = colors['metric_good'] if vision_data['eyes']['eyes_detected'] else colors['metric_critical']
        mouth_color = colors['metric_good'] if (vision_data['mouth']['mouth_detected'] and vision_data['mouth']['mouth_visible']) else colors['metric_critical']
        pose_color = colors['metric_good'] if vision_data['pose']['pose_detected'] else colors['metric_critical']
        
        cv2.putText(dashboard, f"Eyes: {status_eyes}", (20, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, eyes_color, 1)
        y_position += 20
        
        cv2.putText(dashboard, f"Mouth: {status_mouth}", (20, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mouth_color, 1)
        y_position += 20
        
        cv2.putText(dashboard, f"Pose: {status_pose}", (20, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 1)
        y_position += 40
        
        # Current Values Section (always show MAR and other values)
        cv2.putText(dashboard, "CURRENT VALUES:", (20, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text_primary'], 1)
        y_position += 25
        
        # Always show EAR if eyes detected
        if vision_data['eyes']['eyes_detected']:
            cv2.putText(dashboard, f"EAR: {vision_data['eyes']['avg_ear']:.3f}", (20, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text_secondary'], 1)
            y_position += 20
        
        # Always show MAR if mouth detected (regardless of visibility)
        if vision_data['mouth']['mouth_detected']:
            cv2.putText(dashboard, f"MAR: {vision_data['mouth']['mar']:.3f}", (20, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text_secondary'], 1)
            y_position += 20
        
        # Show pitch if pose detected
        if vision_data['pose']['pose_detected']:
            cv2.putText(dashboard, f"PITCH: {vision_data['pose']['pitch']:.1f}°", (20, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text_secondary'], 1)
            y_position += 20
        
        # Alert Section
        if results['attention_result']['should_alert']:
            alert_height = 50
            cv2.rectangle(dashboard, (10, y_position), (dashboard_width-10, y_position+alert_height), colors['attention_critical'], -1)
            cv2.putText(dashboard, "DROWSINESS ALERT!", (dashboard_width//2 - 100, y_position+30),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, colors['text_primary'], 2)
        
        return dashboard

def play_alert_sound(file_path):

    def _play():
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Error playing alert sound: {e}")

    threading.Thread(target=_play, daemon=True).start()        


def main():
    """
    Main function to run the drowsiness detection system
    """
    # Initialize the complete system
    system = DrowsinessDetectionSystem("configs/default.yaml")
    
    # Initialize camera with config values
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, system.config['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, system.config['camera']['height'])
    cap.set(cv2.CAP_PROP_FPS, system.config['camera']['fps'])
    
    # Initialize data logger
    logger = DataLogger()


    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("Starting Complete Drowsiness Detection System")
    print(f"Camera: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
    print(f"Configuration: configs/default.yaml")
    print("\nControls:")
    print("  'q' - Quit application")
    print("  'r' - Reset all metrics")
    print("  's' - Show configuration values")
    print("\nDetection Tips:")
    print("- Look down for configured duration to register a nod")
    print("- System uses PERCLOS, nods, blinks, and yawns for detection")
    print("- Adjust thresholds in configs/default.yaml for sensitivity")
    
    alert_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera")
                break
            
            # Process frame through complete system
            results, display_frame = system.process_frame(frame)
            
            # --- Event Logging ---
            metrics = results['metrics_data']
            vision_data = results['vision_data']
            attention = results['attention_result']

            # Initialize persistent variable for eye closure tracking
            if not hasattr(system, "_eyes_closed_start"):
                system._eyes_closed_start = None

            # --- Yawn Event ---
            if metrics['yawns'].get('new_yawn_detected', False):
                logger.log_event("Yawn Detected", f"MAR={vision_data['mouth']['mar']:.2f}")

            # --- Head Nod Event ---
            if metrics['nods'].get('new_nod_detected', False):
                logger.log_event("Head Nod Detected", f"Pitch={vision_data['pose']['pitch']:.1f}°")

            # --- Head Tilt Event ---
            if vision_data['pose']['head_orientation'] in ["LOOKING LEFT", "LOOKING RIGHT"]:
                logger.log_event("Head Tilt Detected", f"Direction={vision_data['pose']['head_orientation']}")

            # --- Eyes Closed >2s Event ---
            ear_value = vision_data['eyes']['avg_ear']
            ear_thresh = system.config['eyes']['ear_thresh']
            if ear_value < ear_thresh:
                if system._eyes_closed_start is None:
                    system._eyes_closed_start = time.time()
                else:
                    duration = time.time() - system._eyes_closed_start
                    if duration > 2.0:
                        logger.log_event("Eyes Closed >2s", f"Duration={duration:.1f}s")
                        system._eyes_closed_start = None  # reset timer after logging
            else:
                system._eyes_closed_start = None

            # --- Drowsiness Detected Event ---
            if attention['should_alert']:
                logger.log_event("Drowsiness Detected", f"Attention Level={attention['attention_level']:.0f}%")


            # Draw professional dashboard
            dashboard = system.draw_dashboard(display_frame, results)
            
            # Combine the display frame and dashboard
            frame_resized = cv2.resize(display_frame, (actual_width, actual_height))
            combined = np.hstack([frame_resized, dashboard])
            
            # Display frame info and FPS if enabled
            if system.config['display']['show_fps']:
                elapsed_time = time.time() - start_time
                fps = system.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                cv2.putText(combined, f"Frame: {results['frame_count']} | FPS: {fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Handle alerts
            if results['attention_result']['should_alert']:
                alert_count += 1
                current_time = time.strftime("%H:%M:%S")
                attention_level = results['attention_result']['attention_level']
                print(f"ALERT [{current_time}] #{alert_count}: Attention level {attention_level:.0f}%")
                
                # Visual alert on screen
                cv2.putText(combined, "DROWSINESS DETECTED!", 
                           (actual_width//2 - 150, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)
                
                # Play MP3/WAV alert sound
                play_alert_sound(file_path)
            
            # Show the combined output
            cv2.imshow('Drowsiness Detection System', combined)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset all metrics
                system.perclos.reset()
                system.nods.reset()
                system.blinks.reset()
                system.yawns.reset()
                system.fuser.reset()
                alert_count = 0
                start_time = time.time()
                system.frame_count = 0
                print("All metrics reset!")
            elif key == ord('s'):
                # Show current configuration
                print("\nCurrent Configuration:")
                print(f"  PERCLOS: window={system.config['perclos']['window_s']}s, threshold={system.config['perclos']['tired_thresh']}")
                print(f"  Nods: threshold={system.config['nods']['down_thresh_deg']}°, count_for_drowsy={system.config['nods']['count_for_drowsy']}")
                print(f"  Eyes: EAR threshold={system.config['eyes']['ear_thresh']}, time threshold={system.config['eyes']['ear_time_thresh_s']}s")
                print(f"  Fusion weights: {system.config['fusion']['weights']}")
                
    except KeyboardInterrupt:
        print("System interrupted by user")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        system.eye_detector.cleanup()
        system.mouth_detector.cleanup()
        system.pose_detector.cleanup()
        
        total_time = time.time() - start_time
        print(f"\nSystem Statistics:")
        print(f"  Total runtime: {total_time:.1f} seconds")
        print(f"  Frames processed: {system.frame_count}")
        print(f"  Average FPS: {system.frame_count/total_time:.1f}")
        print(f"  Total alerts: {alert_count}")
        print("System shutdown complete")

if __name__ == "__main__":
    main()