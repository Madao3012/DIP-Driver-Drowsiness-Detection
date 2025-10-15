# main.py — Raspberry Pi 4 + Picamera2 unified pipeline
import time
import numpy as np
import cv2
import mediapipe as mp
from picamera2 import Picamera2

from parser import get_args
from utils import get_landmarks, load_camera_parameters, draw_pose_info, rot_mat_to_euler
from eye_detector import EyeDetector
from attention_scorer import AttentionScorer
from yawn_detector import YawnDetector
from pose_estimation import HeadPoseEstimator

def init_picam(width=640, height=480, fmt="RGB888"):
    picam = Picamera2()
    config = picam.create_preview_configuration(main={"format": fmt, "size": (width, height)})
    picam.configure(config)
    picam.start()
    time.sleep(0.2)
    return picam

def main():
    args = get_args()
    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)
        except Exception:
            pass

    # MediaPipe FaceMesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    eye_det = EyeDetector(show_processing=args.show_eye_proc)
    head_est = HeadPoseEstimator(show_axis=args.show_axis, camera_matrix=None, dist_coeffs=None)
    yawn = YawnDetector(mar_thresh=args.mar_thresh, min_duration=args.yawn_min_duration)

    prev_time = time.perf_counter()
    scorer = AttentionScorer(
        t_now=prev_time,
        ear_thresh=args.ear_thresh,
        gaze_thresh=args.gaze_thresh,
        perclos_thresh=args.perclos_thresh,
        roll_thresh=args.roll_thresh,
        pitch_thresh=args.pitch_thresh,
        yaw_thresh=args.yaw_thresh,
        ear_time_thresh=args.ear_time_thresh,
        gaze_time_thresh=args.gaze_time_thresh,
        pose_time_thresh=args.pose_time_thresh,
        perclos_window=args.perclos_window,
        nod_down_thresh=args.nod_down_thresh,
        nod_up_thresh=args.nod_up_thresh,
        nod_window=args.nod_window,
        nods_for_drowsy=args.nods_for_drowsy,
        nod_window_seconds=args.nod_window_seconds,
        decay_factor=0.9,
        verbose=args.verbose,
    )

    picam = init_picam(640, 480, "RGB888")

    try:
        while True:
            t_now = time.perf_counter()
            dt = t_now - prev_time
            prev_time = t_now
            fps = 1.0 / dt if dt > 0 else 0.0

            frame = picam.capture_array()  # RGB
            h, w = frame.shape[:2]
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            e1 = cv2.getTickCount()
            res = face_mesh.process(frame)  # expects RGB
            ear = gaze = None
            roll = pitch = yaw = None
            perclos_val = 0.0
            mar = None
            yawning_now = False
            yawn_event = False

            if res.multi_face_landmarks:
                lmk = get_landmarks(res.multi_face_landmarks)  # normalized (N x 3)
                # Eye metrics
                eye_det.show_eye_keypoints(bgr, lmk, (w, h))
                ear = eye_det.get_EAR(lmk)
                tired, perclos_val = scorer.get_rolling_PERCLOS(t_now, ear)
                gaze = eye_det.get_Gaze_Score(bgr, lmk, (w, h))
                # Head pose
                bgr, roll, pitch, yaw = head_est.get_pose(bgr, lmk, (w, h))
                # Nod / attention logic
                asleep, looking_away, distracted, nod_completed, drowsy_by_nods, nods_recent = scorer.eval_scores(
                    t_now=t_now,
                    ear_score=ear,
                    gaze_score=gaze,
                    head_roll=roll if roll is not None else None,
                    head_pitch=pitch if pitch is not None else None,
                    head_yaw=yaw if yaw is not None else None,
                )
                # Yawn (MAR)
                mar = yawn.compute_mar(lmk[:, :2], (w, h))
                yawning_now, yawn_event = yawn.update(dt, mar)

                # HUD
                y = 30
                cv2.putText(bgr, f"FPS: {int(fps)}", (10, y), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1); y+=25
                if ear is not None: cv2.putText(bgr, f"EAR: {ear:.3f}", (10, y), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1); y+=25
                if gaze is not None: cv2.putText(bgr, f"Gaze: {gaze:.3f}", (10, y), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1); y+=25
                cv2.putText(bgr, f"PERCLOS: {perclos_val:.3f}", (10, y), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1); y+=25
                if mar is not None: cv2.putText(bgr, f"MAR: {mar:.3f}", (10, y), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1); y+=25

                # Pose
                if roll is not None: cv2.putText(bgr, f"roll: {np.round(roll,1)[0]}", (450, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 1)
                if pitch is not None: cv2.putText(bgr, f"pitch: {np.round(pitch,1)[0]}", (450, 70), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 1)
                if yaw is not None: cv2.putText(bgr, f"yaw: {np.round(yaw,1)[0]}", (450, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 1)

                # Alerts
                ay = h - 120
                if tired: cv2.putText(bgr, "TIRED (PERCLOS)!", (10, ay), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2); ay+=25
                if asleep: cv2.putText(bgr, "ASLEEP!", (10, ay), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2); ay+=25
                if looking_away: cv2.putText(bgr, "LOOKING AWAY!", (10, ay), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2); ay+=25
                if distracted: cv2.putText(bgr, "DISTRACTED (POSE)!", (10, ay), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2); ay+=25
                if drowsy_by_nods: cv2.putText(bgr, "DROWSY (NODS)!", (10, ay), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2); ay+=25
                if yawning_now: cv2.putText(bgr, "YAWNING...", (10, ay), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2); ay+=25
                if yawn_event: cv2.putText(bgr, "YAWN DETECTED", (10, ay), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2); ay+=25

            e2 = cv2.getTickCount()
            if args.show_proc_time:
                proc_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000.0
                cv2.putText(bgr, f"PROC: {proc_ms:.1f} ms", (10, h-10), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1)

            cv2.imshow("Driver Monitoring (q/ESC to quit)", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    finally:
        cv2.destroyAllWindows()
        try:
            picam.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()
