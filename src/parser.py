import argparse

def get_args():
    p = argparse.ArgumentParser(description="Driver State Monitoring (RPi4 + Picamera2)")
    # Runtime
    p.add_argument("--show_fps", type=bool, default=True)
    p.add_argument("--show_proc_time", type=bool, default=True)
    p.add_argument("--show_eye_proc", type=bool, default=False)
    p.add_argument("--show_axis", type=bool, default=True)
    p.add_argument("--verbose", type=bool, default=False)
    # EAR (eyes)
    p.add_argument("--ear_thresh", type=float, default=0.20)
    p.add_argument("--ear_time_thresh", type=float, default=2.0)
    p.add_argument("--ear_ema_alpha", type=float, default=0.5, help="EMA smoothing for EAR (from EARv2)")
    # Gaze
    p.add_argument("--gaze_thresh", type=float, default=0.015)
    p.add_argument("--gaze_time_thresh", type=float, default=2.0)
    p.add_argument("--gaze_ema_alpha", type=float, default=0.5, help="EMA smoothing for gaze score")
    # Pose
    p.add_argument("--roll_thresh", type=float, default=20.0)
    p.add_argument("--pitch_thresh", type=float, default=20.0)
    p.add_argument("--yaw_thresh", type=float, default=20.0)
    p.add_argument("--pose_time_thresh", type=float, default=2.5)
    p.add_argument("--pose_deadzone", type=float, default=5.0, help="Deadzone to reduce flicker (from headpose_2.0)")
    # PERCLOS
    p.add_argument("--perclos_window", type=float, default=30.0)
    p.add_argument("--perclos_thresh", type=float, default=0.2)
    # Yawn
    p.add_argument("--mar_thresh", type=float, default=0.60)
    p.add_argument("--yawn_min_duration", type=float, default=0.5)
    # Nod detection
    p.add_argument("--nod_down_thresh", type=float, default=-12.0)
    p.add_argument("--nod_up_thresh", type=float, default=-5.0)
    p.add_argument("--nod_window", type=int, default=15)
    p.add_argument("--nods_for_drowsy", type=int, default=3)
    p.add_argument("--nod_window_seconds", type=float, default=10.0)
    return p.parse_args()
