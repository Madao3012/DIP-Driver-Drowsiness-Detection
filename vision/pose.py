# vision/pose.py
import cv2
import numpy as np
import mediapipe as mp

class PoseDetector:
    """
    Head pose estimation using MediaPipe Face Mesh
    - Estimates head rotation (roll, pitch, yaw)
    - Detects head orientation for drowsiness detection
    - Provides pose landmarks for visualization
    """
    
    def __init__(self, config=None):
        """
        Initialize Pose Detector with configuration
        
        Args:
            config: Configuration dictionary with pose parameters
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
        
        # 3D model points for head pose estimation
        self.face_3d_model = np.array([
            [0.0, 0.0, 0.0],           # Nose tip - 1
            [0.0, -330.0, -65.0],      # Chin - 152
            [-225.0, 170.0, -135.0],   # Left eye left corner - 33
            [225.0, 170.0, -135.0],    # Right eye right corner - 263
            [-150.0, -150.0, -125.0],  # Left mouth corner - 61
            [150.0, -150.0, -125.0]    # Right mouth corner - 291
        ], dtype=np.float64)
        
        # MediaPipe indices corresponding to 3D model points
        self.pose_indices = [1, 152, 33, 263, 61, 291]
        
        # Additional points for visualization
        self.face_outline_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                                   397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                                   172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    def solve_pose(self, landmarks, frame_shape):
        """
        Estimate head pose using SolvePnP
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_shape: Frame dimensions (height, width, channels)
            
        Returns:
            dict: Pose estimation results including rotation and translation vectors
        """
        try:
            height, width = frame_shape[:2]
            camera_center = (width / 2, height / 2)
            focal_length = width
            camera_matrix = np.array([
                [focal_length, 0, camera_center[0]],
                [0, focal_length, camera_center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Assume no lens distortion
            dist_coeffs = np.zeros((4, 1))
            
            # Get 2D image points from landmarks
            image_points = []
            for idx in self.pose_indices:
                point = landmarks[idx]
                x = int(point.x * width)
                y = int(point.y * height)
                image_points.append([x, y])
            
            image_points = np.array(image_points, dtype=np.float64)
            
            # Solve PnP to get rotation and translation vectors
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.face_3d_model, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                
                # Calculate Euler angles
                pose_angles = self._calculate_head_angles(rotation_mat)
                
                return {
                    'success': True,
                    'rotation_vec': rotation_vec,
                    'translation_vec': translation_vec,
                    'rotation_mat': rotation_mat,
                    'roll': pose_angles[0],   # Head tilt
                    'pitch': pose_angles[1],  # Nodding (up/down)
                    'yaw': pose_angles[2],    # Turning (left/right)
                    'image_points': image_points
                }
            else:
                return {'success': False}
                
        except Exception as e:
            print(f"Error solving pose: {e}")
            return {'success': False}

    def _calculate_head_angles(self, rotation_matrix):
        """
        Calculate head pose angles using a more reliable method
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            tuple: (roll, pitch, yaw) angles in degrees
        """
        try:
            # Extract the forward direction vector from rotation matrix
            forward = rotation_matrix[:, 2]
            
            # Calculate pitch (up/down)
            pitch = -np.arcsin(forward[1])
            
            # Calculate yaw (left/right)
            yaw = np.arctan2(forward[0], forward[2])
            
            # Calculate roll (head tilt)
            right = rotation_matrix[:, 0]
            roll = np.arctan2(right[1], right[0])
            
            # Convert to degrees
            pitch_deg = np.degrees(pitch)
            yaw_deg = np.degrees(yaw)
            roll_deg = np.degrees(roll)
            
            return roll_deg, pitch_deg, yaw_deg
            
        except Exception as e:
            print(f"Error calculating head angles: {e}")
            return 0.0, 0.0, 0.0

    def get_pose_landmarks(self, landmarks, frame_shape):
        """
        Get pose landmark positions for visualization
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_shape: Frame dimensions (height, width, channels)
            
        Returns:
            dict: Pose landmark points for visualization
        """
        try:
            height, width = frame_shape[:2]
            
            # Get pose points
            pose_points = []
            for idx in self.pose_indices:
                point = landmarks[idx]
                x = int(point.x * width)
                y = int(point.y * height)
                pose_points.append((x, y))
            
            # Get face outline points
            outline_points = []
            for idx in self.face_outline_indices:
                point = landmarks[idx]
                x = int(point.x * width)
                y = int(point.y * height)
                outline_points.append((x, y))
            
            return {
                'pose_points': pose_points,
                'outline_points': outline_points
            }
            
        except Exception as e:
            print(f"Error getting pose landmarks: {e}")
            return {'pose_points': [], 'outline_points': []}

    def detect_head_orientation(self, pitch, yaw, roll):
        """
        Determine head orientation for drowsiness detection
        
        Args:
            pitch: Head pitch angle in degrees
            yaw: Head yaw angle in degrees
            roll: Head roll angle in degrees
            
        Returns:
            str: Head orientation description
        """
        # Get thresholds from config with defaults
        pose_config = self.config.get('pose', {})
        pitch_threshold = pose_config.get('pitch_thresh_deg', 15)
        
        # For drowsiness, we only care about looking down (head falling forward)
        # Looking down = negative pitch
        if pitch < -pitch_threshold:
            return "LOOKING DOWN"
        elif pitch > pitch_threshold:
            return "LOOKING UP"
        else:
            return "FORWARD"

    def process_frame(self, frame):
        """
        Process a single frame and return pose metrics
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            dict: Pose detection results and metrics
        """
        results = {
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'head_orientation': 'UNKNOWN',
            'pose_detected': False,
            'landmarks': {'pose_points': [], 'outline_points': []}
        }
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            mesh_results = self.face_mesh.process(rgb_frame)
            
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                
                # Solve pose
                pose_data = self.solve_pose(landmarks, frame.shape)
                
                if pose_data['success']:
                    # Get head orientation
                    head_orientation = self.detect_head_orientation(
                        pose_data['pitch'], pose_data['yaw'], pose_data['roll']
                    )
                    
                    # Get landmarks for visualization
                    pose_landmarks = self.get_pose_landmarks(landmarks, frame.shape)
                    
                    results.update({
                        'roll': pose_data['roll'],
                        'pitch': pose_data['pitch'],
                        'yaw': pose_data['yaw'],
                        'head_orientation': head_orientation,
                        'pose_detected': True,
                        'landmarks': pose_landmarks,
                        'pose_data': pose_data
                    })
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            
        return results

    def draw_pose_landmarks(self, frame, landmarks, pose_data=None):
        """
        Draw pose landmarks and axes on the frame using professional colors
        
        Args:
            frame: Frame to draw on
            landmarks: Pose landmark points
            pose_data: Additional pose data for drawing axes
        """
        try:
            # Use professional color scheme
            landmark_color = (255, 255, 255)  # White
            axis_color = (255, 0, 0)          # Red
            
            # Draw face outline points in white
            for point in landmarks['outline_points']:
                cv2.circle(frame, point, 1, landmark_color, -1)
            
            # Draw pose points in white (larger)
            for point in landmarks['pose_points']:
                cv2.circle(frame, point, 3, landmark_color, -1)
            
            # Draw coordinate axes if pose data is available
            if pose_data and pose_data['success']:
                height, width = frame.shape[:2]
                focal_length = width
                camera_center = (width / 2, height / 2)
                camera_matrix = np.array([
                    [focal_length, 0, camera_center[0]],
                    [0, focal_length, camera_center[1]],
                    [0, 0, 1]
                ], dtype=np.float64)
                
                dist_coeffs = np.zeros((4, 1))
                
                # Axis points in 3D space
                axis_points = np.float32([
                    [0, 0, 0],        # Nose tip
                    [50, 0, 0],       # X axis (right)
                    [0, -50, 0],      # Y axis (up)
                    [0, 0, 50]        # Z axis (forward)
                ])
                
                # Project 3D points to 2D image plane
                image_points, _ = cv2.projectPoints(
                    axis_points, pose_data['rotation_vec'], 
                    pose_data['translation_vec'], camera_matrix, dist_coeffs
                )
                
                image_points = np.int32(image_points).reshape(-1, 2)
                
                # Draw axes
                nose_point = tuple(image_points[0])
                x_point = tuple(image_points[1])
                y_point = tuple(image_points[2]) 
                z_point = tuple(image_points[3])
                
                # X-axis (right - red)
                cv2.line(frame, nose_point, x_point, axis_color, 2)
                # Y-axis (up - red) 
                cv2.line(frame, nose_point, y_point, axis_color, 2)
                # Z-axis (forward - red)
                cv2.line(frame, nose_point, z_point, axis_color, 2)
                
        except Exception as e:
            print(f"Error drawing pose landmarks: {e}")

    def cleanup(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()