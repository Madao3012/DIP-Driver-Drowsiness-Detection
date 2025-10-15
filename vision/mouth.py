# vision/mouth.py
import math
import cv2
import numpy as np
import mediapipe as mp

class MouthDetector:
    """
    Mouth detection and analysis using MediaPipe Face Mesh
    - Calculates Mouth Aspect Ratio (MAR)
    - Detects mouth open/closed states
    - Provides mouth landmarks for visualization
    """
    
    def __init__(self, config=None):
        """
        Initialize Mouth Detector with configuration
        
        Args:
            config: Configuration dictionary with mouth parameters
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
        
        # Correct MediaPipe mouth landmarks
        self.MOUTH_OUTER_INDICES = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146
        ]
        
        self.MOUTH_INNER_INDICES = [
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
            308, 415, 310, 311, 312, 13, 82, 81, 80, 191
        ]
        
        # Key points for MAR calculation
        self.MAR_INDICES = [13, 14, 78, 308]

    def calculate_mar(self, landmarks):
        """
        Calculate Mouth Aspect Ratio
        
        Args:
            landmarks: MediaPipe face landmarks
            
        Returns:
            float: Mouth Aspect Ratio value
        """
        try:
            points = {}
            for idx in self.MAR_INDICES:
                point = landmarks[idx]
                points[idx] = (point.x, point.y)
            
            vertical_dist = self._euclidean_distance(points[13], points[14])
            horizontal_dist = self._euclidean_distance(points[78], points[308])
            
            if horizontal_dist > 0:
                mar = vertical_dist / horizontal_dist
            else:
                mar = 0.0
                
            return mar
            
        except Exception as e:
            print(f"Error calculating MAR: {e}")
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

    def detect_mouth_state(self, mar):
        """
        Detect if mouth is open or closed based on MAR threshold
        
        Args:
            mar: Mouth Aspect Ratio value
            
        Returns:
            str: Mouth state ('OPEN' or 'CLOSED')
        """
        mar_threshold = self.config.get('mouth', {}).get('mar_thresh', 0.7)
        return "OPEN" if mar > mar_threshold else "CLOSED"

    def is_mouth_visible(self, landmarks, frame_shape):
        """
        Check if mouth is actually visible (not covered)
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_shape: Frame dimensions (height, width, channels)
            
        Returns:
            bool: True if mouth is visible, False otherwise
        """
        try:
            height, width = frame_shape[:2]
            
            # Get key mouth points
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            left_corner = landmarks[78]
            right_corner = landmarks[308]
            
            # Convert to pixel coordinates
            upper_y = int(upper_lip.y * height)
            lower_y = int(lower_lip.y * height)
            left_x = int(left_corner.x * width)
            right_x = int(right_corner.x * width)
            
            # Calculate mouth dimensions
            mouth_width = abs(right_x - left_x)
            mouth_height = abs(lower_y - upper_y)
            
            # Check if mouth dimensions are reasonable
            # Mouth should be at least 5% of frame width and height
            min_width = width * 0.05
            min_height = height * 0.02
            
            # Mouth shouldn't be too large either (max 40% of frame width)
            max_width = width * 0.4
            
            if (mouth_width < min_width or mouth_width > max_width or 
                mouth_height < min_height):
                return False
                
            return True
            
        except Exception as e:
            print(f"Error checking mouth visibility: {e}")
            return False

    def get_mouth_landmarks(self, landmarks, frame_shape):
        """
        Get mouth landmark positions for visualization
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_shape: Frame dimensions (height, width, channels)
            
        Returns:
            dict: Mouth landmark points for visualization
        """
        try:
            height, width = frame_shape[:2]
            
            # Get outer mouth points
            outer_points = []
            for idx in self.MOUTH_OUTER_INDICES:
                point = landmarks[idx]
                x = int(point.x * width)
                y = int(point.y * height)
                outer_points.append((x, y))
            
            # Get inner mouth points
            inner_points = []
            for idx in self.MOUTH_INNER_INDICES:
                point = landmarks[idx]
                x = int(point.x * width)
                y = int(point.y * height)
                inner_points.append((x, y))
            
            # Get MAR points
            mar_points = []
            for idx in self.MAR_INDICES:
                point = landmarks[idx]
                x = int(point.x * width)
                y = int(point.y * height)
                mar_points.append((x, y))
            
            return {
                'outer_points': outer_points,
                'inner_points': inner_points,
                'mar_points': mar_points
            }
            
        except Exception as e:
            print(f"Error getting mouth landmarks: {e}")
            return {'outer_points': [], 'inner_points': [], 'mar_points': []}

    def process_frame(self, frame):
        """
        Process a single frame and return mouth metrics
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            dict: Mouth detection results and metrics
        """
        results = {
            'mar': 0.0,
            'mouth_state': 'CLOSED',
            'mouth_detected': False,
            'mouth_visible': False,
            'landmarks': {'outer_points': [], 'inner_points': [], 'mar_points': []}
        }
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            mesh_results = self.face_mesh.process(rgb_frame)
            
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                
                # Always calculate MAR regardless of visibility
                mar = self.calculate_mar(landmarks)
                mouth_state = self.detect_mouth_state(mar)
                
                # Check if mouth is actually visible (not covered)
                mouth_visible = self.is_mouth_visible(landmarks, frame.shape)
                
                if mouth_visible:
                    mouth_landmarks = self.get_mouth_landmarks(landmarks, frame.shape)
                    
                    results.update({
                        'mar': mar,
                        'mouth_state': mouth_state,
                        'mouth_detected': True,
                        'mouth_visible': True,
                        'landmarks': mouth_landmarks
                    })
                else:
                    # Mouth is detected but not visible (covered)
                    results.update({
                        'mar': mar,  # Still include MAR value
                        'mouth_state': mouth_state,
                        'mouth_detected': True,
                        'mouth_visible': False,
                        'landmarks': {'outer_points': [], 'inner_points': [], 'mar_points': []}
                    })
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            
        return results

    def draw_mouth_landmarks(self, frame, landmarks):
        """
        Draw mouth landmarks on the frame using professional colors
        
        Args:
            frame: Frame to draw on
            landmarks: Mouth landmark points
        """
        try:
            # Use professional color scheme
            outline_color = (255, 255, 255)  # White
            measurement_color = (255, 0, 0)   # Red
            
            # Draw outer mouth points and outline in white
            if len(landmarks['outer_points']) > 0:
                for point in landmarks['outer_points']:
                    cv2.circle(frame, point, 2, outline_color, -1)
                
                # Draw outer mouth outline
                points = np.array(landmarks['outer_points'], np.int32)
                cv2.polylines(frame, [points], True, outline_color, 1)
            
            # Draw inner mouth points in white
            if len(landmarks['inner_points']) > 0:
                for point in landmarks['inner_points']:
                    cv2.circle(frame, point, 1, outline_color, -1)
                
            # Draw MAR measurement lines in red
            if len(landmarks['mar_points']) >= 4:
                # Draw vertical line (mouth opening)
                cv2.line(frame, landmarks['mar_points'][0], landmarks['mar_points'][1], 
                        measurement_color, 2)
                
                # Draw horizontal line (mouth width)
                cv2.line(frame, landmarks['mar_points'][2], landmarks['mar_points'][3], 
                        measurement_color, 2)
                
                # Draw MAR points in white (slightly larger)
                for point in landmarks['mar_points']:
                    cv2.circle(frame, point, 3, outline_color, -1)
                
        except Exception as e:
            print(f"Error drawing mouth landmarks: {e}")

    def cleanup(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()