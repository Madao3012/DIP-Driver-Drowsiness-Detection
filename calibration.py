# calibration.py
import cv2
import numpy as np
import yaml
import time
from collections import defaultdict
import json
import os
from datetime import datetime

class SystemCalibrator:
    """
    Interactive calibration system for drowsiness detection parameters
    Collects user-specific data and outputs optimized parameters in YAML format
    """
    
    def __init__(self, config_path="configs/default.yaml"):
        self.config_path = config_path
        self.load_config()
        self.setup_data_structures()
        
    def load_config(self):
        """Load current configuration"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            print(f"Loaded configuration from {self.config_path}")
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found")
            self.config = {}
    
    def setup_data_structures(self):
        """Initialize data collection structures"""
        self.ear_data = defaultdict(list)
        self.mar_data = defaultdict(list)
        self.pose_data = defaultdict(lambda: defaultdict(list))
        self.calibration_results = {}
        self.phase_data = {}
        
    def run_interactive_calibration(self):
        """
        Run complete interactive calibration process
        Returns optimized parameters in YAML format
        """
        print("\n" + "="*60)
        print("        DRIVER DROWSINESS DETECTION CALIBRATION")
        print("="*60)
        print("This process will help optimize parameters for your specific use case.")
        print("Please ensure:")
        print("  - Good lighting conditions")
        print("  - Normal sitting position")
        print("  - Camera at eye level")
        print("  - Follow on-screen instructions carefully")
        print("\nPress 'q' at any time to exit calibration")
        print("="*60)
        
        input("\nPress ENTER to begin calibration...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return None
        
        try:
            # Phase 1: Eye calibration
            print("\nPHASE 1: EYE CALIBRATION")
            print("-" * 30)
            self.calibrate_eyes(cap)
            
            # Phase 2: Mouth calibration  
            print("\nPHASE 2: MOUTH CALIBRATION")
            print("-" * 30)
            self.calibrate_mouth(cap)
            
            # Phase 3: Head pose calibration
            print("\nPHASE 3: HEAD POSE CALIBRATION")
            print("-" * 30)
            self.calibrate_head_pose(cap)
            
        except KeyboardInterrupt:
            print("\nCalibration interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Calculate optimal parameters
        optimized_params = self.calculate_optimal_parameters()
        
        # Display results
        self.display_calibration_summary(optimized_params)
        
        return optimized_params
    
    def calibrate_eyes(self, cap):
        """Calibrate eye parameters with detailed instructions"""
        from vision.eyes import EyeDetector
        
        eye_detector = EyeDetector(self.config)
        
        # Sub-phase 1: Eyes open
        print("Keep your eyes OPEN and look straight ahead")
        print("Collecting data for 8 seconds...")
        self.run_calibration_phase(cap, eye_detector, "eyes_open", "EYES OPEN\nLook straight ahead", 8)
        
        # Sub-phase 2: Eyes closed
        print("Keep your eyes CLOSED gently")
        print("Collecting data for 8 seconds...")  
        self.run_calibration_phase(cap, eye_detector, "eyes_closed", "EYES CLOSED\nKeep eyes gently closed", 8)
        
        # Sub-phase 3: Normal blinking
        print("Blink NORMALLY (as you would while driving)")
        print("Collecting data for 10 seconds...")
        self.run_calibration_phase(cap, eye_detector, "eyes_blinking", "NORMAL BLINKING\nBlink naturally", 10)
        
        eye_detector.cleanup()
        print("Eye calibration completed")
    
    def calibrate_mouth(self, cap):
        """Calibrate mouth parameters"""
        from vision.mouth import MouthDetector
        
        mouth_detector = MouthDetector(self.config)
        
        # Sub-phase 1: Mouth closed
        print("Keep your mouth CLOSED normally")
        print("Collecting data for 8 seconds...")
        self.run_calibration_phase(cap, mouth_detector, "mouth_closed", "MOUTH CLOSED\nKeep mouth normally closed", 8)
        
        # Sub-phase 2: Mouth open (talking)
        print("Open your mouth as if TALKING")
        print("Collecting data for 8 seconds...")
        self.run_calibration_phase(cap, mouth_detector, "mouth_open", "MOUTH OPEN\nAs if talking", 8)
        
        # Sub-phase 3: Yawning motion
        print("Perform a YAWNING motion")
        print("Collecting data for 10 seconds...")
        self.run_calibration_phase(cap, mouth_detector, "mouth_yawning", "YAWNING MOTION\nOpen mouth wide", 10)
        
        mouth_detector.cleanup()
        print("Mouth calibration completed")
    
    def calibrate_head_pose(self, cap):
        """Calibrate head pose parameters"""
        from vision.pose import PoseDetector
        
        pose_detector = PoseDetector(self.config)
        
        # Sub-phase 1: Forward position
        print("Look FORWARD (normal driving position)")
        print("Collecting data for 8 seconds...")
        self.run_calibration_phase(cap, pose_detector, "head_forward", "LOOK FORWARD\nNormal driving position", 8)
        
        # Sub-phase 2: Looking down
        print("Look DOWN (as if checking dashboard)")
        print("Collecting data for 8 seconds...")
        self.run_calibration_phase(cap, pose_detector, "head_down", "LOOK DOWN\nAs if checking dashboard", 8)
        
        # Sub-phase 3: Looking side
        print("Look LEFT and RIGHT (shoulder check)")
        print("Collecting data for 10 seconds...")
        self.run_calibration_phase(cap, pose_detector, "head_side", "LOOK SIDE\nLeft and right shoulder check", 10)
        
        pose_detector.cleanup()
        print("Head pose calibration completed")
    
    def run_calibration_phase(self, cap, detector, phase_name, instruction_text, duration):
        """Run a single calibration phase"""
        start_time = time.time()
        data_points = []
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Apply mirror effect (flip horizontally) for natural viewing
            display_frame = cv2.flip(frame, 1)
            
            # Process frame with appropriate detector (use original frame for processing)
            results = detector.process_frame(frame)
            
            # Collect relevant data based on detector type
            if hasattr(detector, '__class__'):
                detector_type = detector.__class__.__name__
                if 'Eye' in detector_type and results['eyes_detected']:
                    data_points.append(results['avg_ear'])
                elif 'Mouth' in detector_type and results['mouth_detected']:
                    data_points.append(results['mar'])
                elif 'Pose' in detector_type and results['pose_detected']:
                    data_points.append({
                        'pitch': results['pitch'],
                        'yaw': results['yaw'], 
                        'roll': results['roll']
                    })
            
            # Display progress with centered instructions
            elapsed = time.time() - start_time
            progress = min(elapsed / duration, 1.0)
            self.display_calibration_frame(display_frame, instruction_text, progress, elapsed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt("User interrupted")
        
        # Store phase data
        self.phase_data[phase_name] = data_points
        print(f"Collected {len(data_points)} data points")
    
    def display_calibration_frame(self, frame, instruction_text, progress, elapsed):
        """Display calibration frame with centered instructions and progress information"""
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Split instruction text into lines
        instruction_lines = instruction_text.split('\n')
        
        # Calculate text positions for center of screen
        text_y_start = height // 2 - (len(instruction_lines) * 40) // 2
        
        # Draw semi-transparent background for better text readability
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, text_y_start - 50), (width, text_y_start + len(instruction_lines) * 50 + 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # Add centered instruction text using main.py font styles
        for i, line in enumerate(instruction_lines):
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = text_y_start + i * 40
            
            cv2.putText(display_frame, line, (text_x, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        
        # Add progress bar at bottom
        bar_width = width - 100
        bar_height = 20
        bar_x, bar_y = 50, height - 80
        
        # Background bar
        cv2.rectangle(display_frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progress bar
        progress_width = int(bar_width * progress)
        cv2.rectangle(display_frame, (bar_x, bar_y), 
                     (bar_x + progress_width, bar_y + bar_height), (0, 200, 0), -1)
        
        # Progress text (using main.py font style)
        progress_text = f"Progress: {progress*100:.0f}% ({elapsed:.1f}s)"
        text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        progress_x = (width - text_size[0]) // 2
        cv2.putText(display_frame, progress_text, (progress_x, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions at bottom (using main.py font style)
        quit_text = "Press 'q' to quit calibration"
        quit_text_size = cv2.getTextSize(quit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        quit_x = (width - quit_text_size[0]) // 2
        cv2.putText(display_frame, quit_text, (quit_x, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Calibration', display_frame)
    
    def calculate_optimal_parameters(self):
        """Calculate optimal parameters from collected data"""
        optimized_params = {}
        
        # Calculate EAR threshold
        if 'eyes_open' in self.phase_data and 'eyes_closed' in self.phase_data:
            open_ears = np.array(self.phase_data['eyes_open'])
            closed_ears = np.array(self.phase_data['eyes_closed'])
            
            if len(open_ears) > 0 and len(closed_ears) > 0:
                # Use statistical separation with safety margin
                open_mean, open_std = np.mean(open_ears), np.std(open_ears)
                closed_mean, closed_std = np.mean(closed_ears), np.std(closed_ears)
                
                # Optimal threshold with 1.5 sigma safety margin
                ear_threshold = closed_mean + 1.5 * closed_std
                ear_threshold = max(ear_threshold, open_mean - 1.5 * open_std)
                ear_threshold = np.clip(ear_threshold, 0.15, 0.35)
                
                optimized_params['eyes'] = {
                    'ear_thresh': float(ear_threshold),
                    'ear_time_thresh_s': 1.8
                }
                optimized_params['perclos'] = {
                    'closed_ear_thresh': float(ear_threshold)
                }
        
        # Calculate MAR threshold
        if 'mouth_closed' in self.phase_data and 'mouth_open' in self.phase_data:
            closed_mars = np.array(self.phase_data['mouth_closed'])
            open_mars = np.array(self.phase_data['mouth_open'])
            
            if len(closed_mars) > 0 and len(open_mars) > 0:
                closed_mean, closed_std = np.mean(closed_mars), np.std(closed_mars)
                open_mean, open_std = np.mean(open_mars), np.std(open_mars)
                
                # Find separation point
                mar_threshold = closed_mean + 2.0 * closed_std
                mar_threshold = np.clip(mar_threshold, 0.4, 0.8)
                
                optimized_params['mouth'] = {
                    'mar_thresh': float(mar_threshold),
                    'yawn_min_duration_s': 0.6
                }
        
        # Calculate pose thresholds
        if 'head_forward' in self.phase_data and 'head_down' in self.phase_data:
            forward_data = self.phase_data['head_forward']
            down_data = self.phase_data['head_down']
            
            if len(forward_data) > 0 and len(down_data) > 0:
                # Extract pitch values
                forward_pitch = [d['pitch'] for d in forward_data if isinstance(d, dict)]
                down_pitch = [d['pitch'] for d in down_data if isinstance(d, dict)]
                
                if len(forward_pitch) > 0 and len(down_pitch) > 0:
                    # Calculate pitch threshold (looking down)
                    pitch_threshold = np.percentile(forward_pitch, 90)
                    optimized_params['pose'] = {
                        'pitch_thresh_deg': float(abs(pitch_threshold)),
                        'roll_thresh_deg': 15.0,
                        'yaw_thresh_deg': 20.0
                    }
                    optimized_params['nods'] = {
                        'down_thresh_deg': float(abs(pitch_threshold))
                    }
        
        # Set default values for missing parameters
        defaults = {
            'blinks': {
                'min_separation_s': 0.2,
                'max_duration_s': 0.4
            },
            'perclos': optimized_params.get('perclos', {'closed_ear_thresh': 0.21, 'tired_thresh': 0.25}),
            'fusion': {
                'weights': {
                    'perclos': 0.35,
                    'nods': 0.25, 
                    'blinks': 0.15,
                    'yawns': 0.25
                }
            }
        }
        
        # Merge with defaults
        for key, value in defaults.items():
            if key not in optimized_params:
                optimized_params[key] = value
        
        return optimized_params
    
    def display_calibration_summary(self, optimized_params):
        """Display calibration results summary"""
        print("\n" + "="*60)
        print("            CALIBRATION RESULTS SUMMARY")
        print("="*60)
        
        if not optimized_params:
            print("No calibration data collected")
            return
        
        print("\nOPTIMIZED PARAMETERS:")
        print("-" * 40)
        
        if 'eyes' in optimized_params:
            print(f"EAR Threshold: {optimized_params['eyes']['ear_thresh']:.3f}")
            current_ear = self.config.get('eyes', {}).get('ear_thresh', 'N/A')
            print(f"Current: {current_ear}")
        
        if 'mouth' in optimized_params:
            print(f"MAR Threshold: {optimized_params['mouth']['mar_thresh']:.3f}")
            current_mar = self.config.get('mouth', {}).get('mar_thresh', 'N/A') 
            print(f"Current: {current_mar}")
        
        if 'pose' in optimized_params:
            print(f"Pitch Threshold: {optimized_params['pose']['pitch_thresh_deg']:.1f} deg")
            current_pitch = self.config.get('pose', {}).get('pitch_thresh_deg', 'N/A')
            print(f"Current: {current_pitch}")
        
        print("\nNext steps:")
        print("1. Review the parameters above")
        print("2. Copy the YAML output below to your config file")
        print("3. Test the system with the new parameters")
        print("4. Fine-tune if necessary")
    
    def generate_yaml_output(self, optimized_params):
        """Generate YAML format output for the optimized parameters"""
        if not optimized_params:
            return "# No calibration data available"
        
        yaml_output = "# Optimized Parameters from Calibration\n"
        yaml_output += f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        yaml_output += "# Replace the corresponding sections in configs/default.yaml\n\n"
        
        for category, params in optimized_params.items():
            yaml_output += f"{category}:\n"
            for key, value in params.items():
                if isinstance(value, dict):
                    yaml_output += f"  {key}:\n"
                    for sub_key, sub_value in value.items():
                        yaml_output += f"    {sub_key}: {sub_value}\n"
                else:
                    yaml_output += f"  {key}: {value}\n"
            yaml_output += "\n"
        
        return yaml_output
    
    def save_calibration_report(self, optimized_params, filename="calibration_report.txt"):
        """Save detailed calibration report"""
        report = f"Driver Drowsiness Detection Calibration Report\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 50 + "\n\n"
        
        report += "COLLECTED DATA SUMMARY:\n"
        report += "-" * 30 + "\n"
        for phase, data in self.phase_data.items():
            report += f"{phase}: {len(data)} samples\n"
        
        report += "\nOPTIMIZED PARAMETERS:\n"
        report += "-" * 30 + "\n"
        report += self.generate_yaml_output(optimized_params)
        
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"Detailed report saved to: {filename}")

def main():
    """Main calibration function"""
    calibrator = SystemCalibrator("configs/default.yaml")
    
    print("Driver Drowsiness Detection System Calibration")
    print("This will generate optimized parameters in YAML format")
    
    response = input("Start calibration? (y/n): ").lower().strip()
    if response != 'y':
        print("Calibration cancelled.")
        return
    
    # Run calibration
    optimized_params = calibrator.run_interactive_calibration()
    
    if optimized_params:
        # Generate YAML output
        yaml_output = calibrator.generate_yaml_output(optimized_params)
        
        print("\n" + "="*60)
        print("              YAML OUTPUT")
        print("="*60)
        print("\nCopy the following YAML to update your configuration:\n")
        print(yaml_output)
        
        # Save detailed report
        calibrator.save_calibration_report(optimized_params)
        
        # Offer to save YAML to file
        save_yaml = input("\nSave YAML to file? (y/n): ").lower().strip()
        if save_yaml == 'y':
            with open("calibrated_parameters.yaml", "w") as f:
                f.write(yaml_output)
            print("YAML saved to 'calibrated_parameters.yaml'")
    
    print("\nCalibration process completed!")

if __name__ == "__main__":
    main()