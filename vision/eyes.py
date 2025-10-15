# vision/eyes.py
import math
import cv2
import numpy as np
import mediapipe as mp

class EyeDetector:
    """
    Eye detection and analysis using MediaPipe Face Mesh
    - Calculates Eye Aspect Ratio (EAR)
    - Detects gaze direction
    - Provides eye landmarks for visualization
    """
    
    def __init__(self, config=None):
        """
        Initialize Eye Detector with configuration
        
        Args:
            config: Configuration dictionary with eye parameters
        """
        self.config = config or {}
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe face mesh indices for eyes
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Specific indices for EAR calculation (6 points per eye)
        self.LEFT_EAR_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EAR_INDICES = [362, 385, 387, 263, 373, 380]
        
        # Enhanced gaze detection indices
        self.LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
        
        # Eye corner indices for more accurate gaze detection
        self.LEFT_EYE_CORNERS = [33, 133]   # Left and right corners of left eye
        self.RIGHT_EYE_CORNERS = [362, 263] # Left and right corners of right eye

    def calculate_ear(self, landmarks, eye_indices):
        """
        Calculate Eye Aspect Ratio for a single eye
        
        Args:
            landmarks: MediaPipe face landmarks
            eye_indices: Indices for the 6 EAR points
            
        Returns:
            float: Eye Aspect Ratio value
        """
        try:
            # Extract the 6 key points for EAR calculation
            points = []
            for idx in eye_indices:
                point = landmarks[idx]
                points.append((point.x, point.y))
            
            # Calculate vertical distances
            p2_p6 = self._euclidean_distance(points[1], points[5])
            p3_p5 = self._euclidean_distance(points[2], points[4])
            
            # Calculate horizontal distance
            p1_p4 = self._euclidean_distance(points[0], points[3])
            
            # EAR formula
            ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
            return ear
            
        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return 0.0

    def _euclidean_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            float: Euclidean distance
        """
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def detect_gaze(self, landmarks, frame_shape):
        """
        Enhanced gaze direction detection for both eyes
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_shape: Frame dimensions (height, width, channels)
            
        Returns:
            dict: Gaze information for both eyes
        """
        try:
            height, width = frame_shape[:2]
            
            # Get iris centers
            left_iris = landmarks[self.LEFT_IRIS_INDICES[0]]
            right_iris = landmarks[self.RIGHT_IRIS_INDICES[0]]
            
            # Convert to pixel coordinates
            left_iris_x = int(left_iris.x * width)
            left_iris_y = int(left_iris.y * height)
            right_iris_x = int(right_iris.x * width)
            right_iris_y = int(right_iris.y * height)
            
            # Enhanced gaze estimation
            gaze_left = self._estimate_gaze_enhanced(landmarks, self.LEFT_EYE_INDICES, 
                                                   self.LEFT_EYE_CORNERS, left_iris_x, width, height)
            gaze_right = self._estimate_gaze_enhanced(landmarks, self.RIGHT_EYE_INDICES, 
                                                    self.RIGHT_EYE_CORNERS, right_iris_x, width, height)
            
            # Determine overall gaze direction
            overall_gaze = self._determine_overall_gaze(gaze_left, gaze_right)
            
            return {
                'left_gaze': gaze_left,
                'right_gaze': gaze_right,
                'overall_gaze': overall_gaze,
                'left_iris_pos': (left_iris_x, left_iris_y),
                'right_iris_pos': (right_iris_x, right_iris_y)
            }
            
        except Exception as e:
            print(f"Error detecting gaze: {e}")
            return {'left_gaze': 'unknown', 'right_gaze': 'unknown', 'overall_gaze': 'unknown',
                    'left_iris_pos': (0, 0), 'right_iris_pos': (0, 0)}

    def _estimate_gaze_enhanced(self, landmarks, eye_indices, eye_corners, iris_x, width, height):
        """
        Enhanced gaze estimation using eye corners as reference
        
        Args:
            landmarks: MediaPipe face landmarks
            eye_indices: Eye landmark indices
            eye_corners: Eye corner indices
            iris_x: Iris x-coordinate
            width: Frame width
            height: Frame height
            
        Returns:
            str: Gaze direction ('LEFT', 'RIGHT', 'CENTER', 'unknown')
        """
        try:
            # Get eye corner positions
            left_corner = landmarks[eye_corners[0]]
            right_corner = landmarks[eye_corners[1]]
            
            left_corner_x = int(left_corner.x * width)
            right_corner_x = int(right_corner.x * width)
            
            # Calculate eye width
            eye_width = abs(right_corner_x - left_corner_x)
            
            if eye_width == 0:  # Avoid division by zero
                return "unknown"
            
            # Calculate normalized iris position relative to eye corners
            iris_position = (iris_x - left_corner_x) / eye_width
            
            # Determine gaze direction with adjusted thresholds
            if iris_position < 0.4:
                return "LEFT"
            elif iris_position > 0.6:
                return "RIGHT"
            else:
                return "CENTER"
                
        except Exception as e:
            print(f"Error in enhanced gaze estimation: {e}")
            return "unknown"

    def _determine_overall_gaze(self, gaze_left, gaze_right):
        """
        Determine overall gaze direction based on both eyes
        
        Args:
            gaze_left: Left eye gaze direction
            gaze_right: Right eye gaze direction
            
        Returns:
            str: Overall gaze direction
        """
        if gaze_left == "unknown" or gaze_right == "unknown":
            return "UNKNOWN"
        
        # If both eyes agree, use that direction
        if gaze_left == gaze_right:
            return gaze_left
        
        # If eyes disagree but one is center, use the other
        if gaze_left == "CENTER":
            return gaze_right
        if gaze_right == "CENTER":
            return gaze_left
        
        # Default to center if eyes are looking in different directions
        return "CENTER"

    def process_frame(self, frame):
        """
        Process a single frame and return eye metrics
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            dict: Eye detection results and metrics
        """
        results = {
            'left_ear': 0.0,
            'right_ear': 0.0,
            'avg_ear': 0.0,
            'gaze_left': 'unknown',
            'gaze_right': 'unknown',
            'overall_gaze': 'unknown',
            'left_iris_pos': (0, 0),
            'right_iris_pos': (0, 0),
            'eyes_detected': False
        }
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            mesh_results = self.face_mesh.process(rgb_frame)
            
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                
                # Calculate EAR for both eyes
                left_ear = self.calculate_ear(landmarks, self.LEFT_EAR_INDICES)
                right_ear = self.calculate_ear(landmarks, self.RIGHT_EAR_INDICES)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Detect gaze
                gaze_data = self.detect_gaze(landmarks, frame.shape)
                
                results.update({
                    'left_ear': left_ear,
                    'right_ear': right_ear,
                    'avg_ear': avg_ear,
                    'gaze_left': gaze_data['left_gaze'],
                    'gaze_right': gaze_data['right_gaze'],
                    'overall_gaze': gaze_data['overall_gaze'],
                    'left_iris_pos': gaze_data['left_iris_pos'],
                    'right_iris_pos': gaze_data['right_iris_pos'],
                    'eyes_detected': True
                })
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            
        return results

    def draw_eye_landmarks(self, frame, landmarks):
        """
        Draw eye landmarks on the frame for visualization
        
        Args:
            frame: Frame to draw on
            landmarks: MediaPipe face landmarks
        """
        try:
            height, width = frame.shape[:2]
            
            # Draw left eye landmarks
            for idx in self.LEFT_EYE_INDICES:
                point = landmarks[idx]
                x = int(point.x * width)
                y = int(point.y * height)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Draw right eye landmarks
            for idx in self.RIGHT_EYE_INDICES:
                point = landmarks[idx]
                x = int(point.x * width)
                y = int(point.y * height)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
            # Draw iris centers
            left_iris = landmarks[self.LEFT_IRIS_INDICES[0]]
            right_iris = landmarks[self.RIGHT_IRIS_INDICES[0]]
            left_x = int(left_iris.x * width)
            left_y = int(left_iris.y * height)
            right_x = int(right_iris.x * width)
            right_y = int(right_iris.y * height)
            
            cv2.circle(frame, (left_x, left_y), 3, (0, 0, 255), -1)
            cv2.circle(frame, (right_x, right_y), 3, (0, 0, 255), -1)
            
        except Exception as e:
            print(f"Error drawing landmarks: {e}")

    def cleanup(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()