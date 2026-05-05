"""
Global Real-Time Localizer.
Runs as a daemon thread in main.py.
Publishes to pose_queue at LOCALIZER_HZ.
"""
import queue
import time
import json
import pathlib
import numpy as np

from localization.config import LOCALIZER_HZ, N_PARTICLES, RESAMPLE_THRESH
from localization.graph_utils import load_graph, compute_edge_curvatures
from localization.particle_filter import (
    init_particles_uniform, init_particles_warm,
    predict, update_weights, effective_n, resample, map_estimate
)
from localization.optical_flow import estimate_velocity
from localization.appearance_map import load_appearance_map, extract_descriptor
from perception.perspective_transform import PerspectiveTransformer
import math

try:
    from perception.lane_detector import LaneDetector
except ImportError:
    LaneDetector = None

LAST_POSE_PATH = pathlib.Path('localization/last_pose.json')
APP_MAP_PATH   = pathlib.Path('localization/appearance_map.pkl')

class RealtimeLocalizer:
    def __init__(self, pose_queue: queue.Queue, imu, cam, serial):
        self.q       = pose_queue
        self.imu     = imu
        self.cam     = cam
        self.serial  = serial
        self.G       = load_graph()
        compute_edge_curvatures(self.G)

        self.app_map = (load_appearance_map(APP_MAP_PATH)
                        if APP_MAP_PATH.exists() else {})

        # Initialise particles
        if LAST_POSE_PATH.exists():
            try:
                prior = json.loads(LAST_POSE_PATH.read_text())
                self.particles = init_particles_warm(self.G, prior)
            except Exception:
                self.particles = init_particles_uniform(self.G)
        else:
            self.particles = init_particles_uniform(self.G)

        self.prev_roi  = None
        self.prev_time = time.monotonic()
        self.transformer = PerspectiveTransformer()
        self.latest_lateral_err_m = 0.0

    def update_lane_error(self, err_m: float):
        self.latest_lateral_err_m = err_m

    def run(self):
        dt_target = 1.0 / LOCALIZER_HZ
        while True:
            t0 = time.monotonic()

            # --- Read sensors ---
            try:
                frame       = self.cam.read_frame()
                psi_imu     = float(self.imu.get_yaw())
                
                # Directly grab speed in mm/s and steering in degrees
                speed_mms   = float(getattr(self.serial.status, 'speed_mm_s', 0.0))
                steer_deg   = float(getattr(self.serial.status, 'steering_angle', 0.0))
                delta       = math.radians(steer_deg)
                
            except Exception as e:
                time.sleep(dt_target)
                continue

            dt          = t0 - self.prev_time
            v_est = 0.0

            app_desc = None
            lateral_err = self.latest_lateral_err_m
            
            if frame is not None:
                bird        = self.transformer.warp(frame)
                roi         = bird[-int(bird.shape[0]*0.35):, :]

                if self.prev_roi is not None and dt > 0:
                    v_est = estimate_velocity(self.prev_roi, roi, dt)

                self.prev_roi  = roi

                # Velocity fallback logic against throttle
                if v_est == 0 and speed_mms > 200: # Assume positive speed commanded
                    v_est = 0.3
                elif v_est > 0 and abs(speed_mms) < 20: 
                    v_est = 0.0

                # Appearance descriptor for observation
                app_desc = extract_descriptor(frame)

            self.prev_time = t0

            # --- Particle filter cycle ---
            predict(self.particles, v_est, delta, dt, self.G, psi_imu)
            update_weights(self.particles, psi_imu, lateral_err,
                           app_desc, self.G, self.app_map, delta)

            if effective_n(self.particles) / N_PARTICLES < RESAMPLE_THRESH:
                self.particles = resample(self.particles)

            pose = map_estimate(self.particles, self.G)

            # --- Publish (drop stale, always latest) ---
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put_nowait(pose)

            # --- Sleep remainder of cycle ---
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, dt_target - elapsed))

    def shutdown(self):
        """Call on Ctrl-C to persist pose for warm start."""
        if self.particles:
            pose = map_estimate(self.particles, self.G)
            LAST_POSE_PATH.write_text(json.dumps(pose, indent=2))
