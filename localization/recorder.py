"""
Run during Phase 1 manual drive.
Usage: python -m localization.recorder --output recordings/run_001
"""
import csv
import cv2
import time
import threading
import pathlib
import argparse
from hardware.imu_sensor import IMUSensor
from hardware.serial_handler import SerialHandler
from perception.camera import Camera
from localization.config import IMU_HZ

def _write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def record(output_dir: str, duration_s: float = None):
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame_dir = out / 'frames'
    frame_dir.mkdir(exist_ok=True)

    imu     = IMUSensor()
    serial  = SerialHandler()
    cam     = Camera()

    imu_rows    = []
    rc_rows     = []
    frame_idx   = 0
    t_start     = time.monotonic()

    print(f"Recording to {out}. Press Ctrl-C to stop.")
    try:
        while True:
            t = time.monotonic() - t_start
            if duration_s and t > duration_s:
                break

            # IMU @ 50 Hz (called in tight loop, imu_sensor handles rate)
            try:
                yaw = float(imu.get_yaw())
                pitch = float(imu.get_pitch())
            except Exception:
                yaw = 0.0
                pitch = 0.0
            
            imu_rows.append([t, yaw, pitch, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # RC commands
            try:
                speed_mms = float(getattr(serial.status, 'speed_mm_s', 0.0))
                steer_deg = float(getattr(serial.status, 'steering_angle', 0.0))
            except Exception:
                speed_mms = 0.0
                steer_deg = 0.0
                
            rc_rows.append([t, steer_deg, speed_mms])

            # Camera @ 15 fps (save every 3rd IMU tick ≈ 50/3 ≈ 15 fps)
            if frame_idx % 3 == 0:
                frame = cam.read_frame()
                if frame is not None:
                    cv2.imwrite(str(frame_dir / f'frame_{frame_idx//3:06d}.jpg'), frame)

            frame_idx += 1
            time.sleep(1.0 / IMU_HZ)

    except KeyboardInterrupt:
        pass

    # Save CSVs
    _write_csv(out / 'imu.csv',
               ['t','yaw','pitch','roll','ax','ay','az','gx','gy','gz'],
               imu_rows)
    _write_csv(out / 'rc.csv',
               ['t','steering_pwm','throttle_pwm'],
               rc_rows)
    print(f"Saved {len(imu_rows)} IMU rows, {frame_idx//3} frames.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--duration', type=float, default=None)
    args = parser.parse_args()
    record(args.output, args.duration)
