# pose_estimation.py
# Minimal head-pose estimator for MediaPipe FaceMesh landmarks using SolvePnP.
# Uses utilities in utils.py for drawing and angle conversion.

import numpy as np
import cv2
from utils import rot_mat_to_euler, draw_pose_info

class HeadPoseEstimator:
    def __init__(self, show_axis=True, camera_matrix=None, dist_coeffs=None):
        self.show_axis = show_axis
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    # SolvePnP based on a small subset of stable face landmarks (MediaPipe indices)
    #  - 33: left eye outer corner
    #  - 263: right eye outer corner
    #  - 1: nose tip-ish
    #  - 61: mouth left
    #  - 291: mouth right
    #  - 199: chin-ish
    SEL = [33, 263, 1, 61, 291, 199]

    # Rough 3D model points (arbitrary units) to match the above 2D points
    MODEL_3D = np.array([
        (-30.0,  0.0,  0.0),   # left eye outer
        ( 30.0,  0.0,  0.0),   # right eye outer
        (  0.0,-25.0, 15.0),   # nose bridge / tip-ish
        (-18.0,-40.0,  5.0),   # mouth left
        ( 18.0,-40.0,  5.0),   # mouth right
        (  0.0,-60.0,  0.0),   # chin-ish
    ], dtype=np.float64)

    def _camera_params(self, frame_size):
        h, w = frame_size[1], frame_size[0]
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            return self.camera_matrix, self.dist_coeffs

        # Reasonable intrinsics guess for 640x480; works well enough without calibration
        f = w
        cam_mat = np.array([[f, 0, w/2],
                            [0, f, h/2],
                            [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)
        return cam_mat, dist

    def get_pose(self, frame, landmarks, frame_size):
        w, h = frame_size
        # Build 2D points from landmarks (normalized -> pixel coords)
        pts_2d = []
        for idx in self.SEL:
            x = float(np.clip(landmarks[idx, 0], 0.0, 1.0) * w)
            y = float(np.clip(landmarks[idx, 1], 0.0, 1.0) * h)
            pts_2d.append([x, y])
        pts_2d = np.array(pts_2d, dtype=np.float64)

        cam_mat, dist = self._camera_params((w, h))

        ok, rvec, tvec = cv2.solvePnP(self.MODEL_3D, pts_2d, cam_mat, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return frame, None, None, None

        rmat, _ = cv2.Rodrigues(rvec)
        # Angles in degrees: [roll, pitch, yaw] from utils.rot_mat_to_euler
        angles = rot_mat_to_euler(rmat)  # returns np.array([x, y, z]) deg
        roll, pitch, yaw = angles[0:1], angles[1:2], angles[2:3]

        if self.show_axis:
            # Project simple axis
            axis_len = 100.0
            axis_3d = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
            nose = pts_2d[2].reshape(1,2)  # using index 1 as origin (nose-ish), adjust if desired
            proj, _ = cv2.projectPoints(axis_3d, rvec, tvec, cam_mat, dist)
            draw_pose_info(frame, (int(nose[0,0]), int(nose[0,1])), proj, roll=roll, pitch=pitch, yaw=yaw)

        return frame, roll, pitch, yaw
