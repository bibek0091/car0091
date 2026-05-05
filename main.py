import cv2
import tkinter as tk
from tkinter import messagebox
import time
import math
import numpy as np

import csv
import json
import os
import sys
import argparse
from PIL import Image, ImageTk

# ── IMPORTS FROM PACKAGES ──────────────────────────────────
from config import *
from dashboard.dashboard_ui import DashboardUI
from dashboard.map_engine import MapEngine
from dashboard.adas_vision_utils import annotate_bev, JunctionDetector, RoundaboutNavigator
import queue
import threading
from localization.global_localizer import RealtimeLocalizer

try:
    from hardware.serial_handler import STM32_SerialHandler
except ImportError:
    class STM32_SerialHandler:
        def __init__(self): self.running = False
        def connect(self): return True
        def disconnect(self): pass
        def set_speed(self, s): pass
        def set_steering(self, s): pass
        def set_light_state(self, state, on): pass

try:
    from hardware.imu_sensor import IMUSensor
except ImportError:
    class IMUSensor:
        def __init__(self): self.is_calibrated = True
        def start(self): pass
        def stop(self): pass
        def get_yaw(self): return 0.0
        def get_pitch(self): return 0.0
        def get_has_hardware(self): return False
        def get_fused_velocity(self): return 0.0
        def update_optical_velocity(self, v): pass

try:
    from v2x.v2x_client import V2XClient
except ImportError:
    class V2XClient:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def update_state(self, *args, **kwargs): pass

# ── AUTONOMOUS STACK IMPORTS ───────────────────────────────
try:
    from perception.camera import Camera
    from perception.lane_detector import LaneDetector
    from control.controller import Controller
    _AUTO_DRIVE_AVAILABLE = True
except ImportError:
    _AUTO_DRIVE_AVAILABLE = False

try:
    from traffic.traffic_module import TrafficDecisionEngine, ThreadedYOLODetector
    from traffic.behavior_controller import BehaviorController
    from hardware.telemetry_logger import TelemetryLogger
    _AI_AVAILABLE = True
except ImportError:
    _AI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  Mini Telemetry Dashboard (Tela)
# ─────────────────────────────────────────────────────────────
class MiniTelemetryDashboard:
    """
    Lightweight telemetry overlay window showing real-time BFMC stats:
    - Lane status & type detection (solid, dashed, missing)
    - Curve detection (left/right/straight)
    - Bus lane alerts
    - Speed / Steer / IMU Yaw
    - AI detection status
    - Loop frequency & latency
    """
    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("TELEMETRY TELA")
        self.win.geometry("420x380")
        self.win.configure(bg="#0a0a0a")
        self.win.resizable(True, True)

        # Title bar
        tk.Label(self.win, text="BFMC TELEMETRY", bg="#0a0a0a", fg="#00ff88",
                 font=("Courier", 14, "bold")).pack(pady=4)

        # Lane status frame
        lane_frm = tk.LabelFrame(self.win, text="LANE STATUS", bg="#111", fg="#0ff",
                                  font=("Courier", 10, "bold"))
        lane_frm.pack(fill=tk.X, padx=8, pady=4)
        self.lbl_lane_type = tk.Label(lane_frm, text="TYPE: ---", bg="#111", fg="#fff",
                                       font=("Courier", 10))
        self.lbl_lane_type.pack(anchor="w", padx=6)
        self.lbl_lane_anchor = tk.Label(lane_frm, text="ANCHOR: ---", bg="#111", fg="#ff0",
                                         font=("Courier", 10))
        self.lbl_lane_anchor.pack(anchor="w", padx=6)
        self.lbl_lane_conf = tk.Label(lane_frm, text="CONF: 0.00", bg="#111", fg="#0f0",
                                       font=("Courier", 10))
        self.lbl_lane_conf.pack(anchor="w", padx=6)
        self.lbl_lane_curve = tk.Label(lane_frm, text="CURVE: STRAIGHT", bg="#111", fg="#f80",
                                        font=("Courier", 10))
        self.lbl_lane_curve.pack(anchor="w", padx=6)

        # Bus lane alert
        self.lbl_bus_lane = tk.Label(lane_frm, text="BUS LANE: CLEAR", bg="#111", fg="#0f0",
                                      font=("Courier", 10, "bold"))
        self.lbl_bus_lane.pack(anchor="w", padx=6)

        # Drive state frame
        drv_frm = tk.LabelFrame(self.win, text="DRIVE STATE", bg="#111", fg="#f0f",
                                 font=("Courier", 10, "bold"))
        drv_frm.pack(fill=tk.X, padx=8, pady=4)
        self.lbl_speed = tk.Label(drv_frm, text="SPEED: 0 PWM", bg="#111", fg="#ff4444",
                                   font=("Courier", 11, "bold"))
        self.lbl_speed.pack(anchor="w", padx=6)
        self.lbl_steer = tk.Label(drv_frm, text="STEER: 0.0 deg", bg="#111", fg="#44aaff",
                                   font=("Courier", 11, "bold"))
        self.lbl_steer.pack(anchor="w", padx=6)
        self.lbl_yaw = tk.Label(drv_frm, text="YAW: 0.0 deg", bg="#111", fg="#ffaa00",
                                 font=("Courier", 10))
        self.lbl_yaw.pack(anchor="w", padx=6)
        self.lbl_mode = tk.Label(drv_frm, text="MODE: MANUAL", bg="#111", fg="#ccc",
                                  font=("Courier", 10))
        self.lbl_mode.pack(anchor="w", padx=6)

        # Performance frame
        perf_frm = tk.LabelFrame(self.win, text="PERFORMANCE", bg="#111", fg="#ff0",
                                  font=("Courier", 10, "bold"))
        perf_frm.pack(fill=tk.X, padx=8, pady=4)
        self.lbl_hz = tk.Label(perf_frm, text="FREQ: 0.0 Hz", bg="#111", fg="#0ff",
                                font=("Courier", 10))
        self.lbl_hz.pack(anchor="w", padx=6)
        self.lbl_latency = tk.Label(perf_frm, text="LATENCY: 0 ms", bg="#111", fg="#f80",
                                     font=("Courier", 10))
        self.lbl_latency.pack(anchor="w", padx=6)
        self.lbl_ai = tk.Label(perf_frm, text="AI DETS: 0", bg="#111", fg="#8f8",
                                font=("Courier", 10))
        self.lbl_ai.pack(anchor="w", padx=6)

        # Root cause
        self.lbl_root = tk.Label(self.win, text="ROOT CAUSE: NORMAL", bg="#0a0a0a",
                                  fg="#888", font=("Courier", 9))
        self.lbl_root.pack(pady=4)

        self.win.protocol("WM_DELETE_WINDOW", self.win.withdraw)

    def update(self, lane_type, anchor, conf, curvature, bus_lane_active,
               speed, steer, yaw, mode, hz, latency_ms, ai_count, root_cause):
        # Lane
        color_type = {"SOLID": "#0f0", "DASHED": "#ff0", "SINGLE_EDGE": "#f80",
                      "MISSING": "#f00", "UNKNOWN": "#888"}.get(lane_type, "#888")
        self.lbl_lane_type.config(text=f"TYPE: {lane_type}", fg=color_type)
        self.lbl_lane_anchor.config(text=f"ANCHOR: {anchor}")
        self.lbl_lane_conf.config(text=f"CONF: {conf:.2f}",
                                   fg="#0f0" if conf > 0.5 else "#f80" if conf > 0.2 else "#f00")

        # Curve
        if curvature < 0.0005:
            curve_text = "STRAIGHT"
            curve_color = "#0f0"
        elif curvature < 0.002:
            curve_text = "GENTLE"
            curve_color = "#ff0"
        else:
            curve_text = "SHARP"
            curve_color = "#f00"
        self.lbl_lane_curve.config(text=f"CURVE: {curve_text} ({curvature:.5f})",
                                    fg=curve_color)

        # Bus lane
        if bus_lane_active:
            self.lbl_bus_lane.config(text="BUS LANE: AVOID!", fg="#f00", bg="#300")
        else:
            self.lbl_bus_lane.config(text="BUS LANE: CLEAR", fg="#0f0", bg="#111")

        # Drive
        self.lbl_speed.config(text=f"SPEED: {int(speed)} PWM")
        self.lbl_steer.config(text=f"STEER: {steer:.1f} deg")
        self.lbl_yaw.config(text=f"YAW: {yaw:.1f} deg")
        self.lbl_mode.config(text=f"MODE: {mode}")

        # Performance
        self.lbl_hz.config(text=f"FREQ: {hz:.1f} Hz",
                            fg="#0ff" if hz > 15 else "#f80" if hz > 8 else "#f00")
        self.lbl_latency.config(text=f"LATENCY: {latency_ms:.1f} ms",
                                 fg="#0f0" if latency_ms < 50 else "#f80" if latency_ms < 80 else "#f00")
        self.lbl_ai.config(text=f"AI DETS: {ai_count}")

        self.lbl_root.config(text=f"ROOT CAUSE: {root_cause}")

    def show(self):
        self.win.deiconify()

    def hide(self):
        self.win.withdraw()


# ─────────────────────────────────────────────────────────────
class BFMC_App:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.headless = args.headless

        if not self.headless:
            self.root.title("TEAM OPTINX BFMC 2026")
            self.root.geometry("1400x850")
            self.root.minsize(1200, 700)
            self.root.configure(bg=THEME["bg"])
            self.ui = DashboardUI(self.root, self)

        self.map_engine = MapEngine()

        # Hardware setup
        self.handler = STM32_SerialHandler()
        self.is_connected = False
        self.toggle_connection()  # Auto-connect on startup

        self.imu = IMUSensor()
        self.imu.start()

        # V2X Client (Daemon thread)
        self.v2x_client = V2XClient(host=V2X_SERVER_HOST, port=V2X_SERVER_PORT)
        if not args.no_v2x:
            self.v2x_client.start()

        # Physics State
        self.car_x, self.car_y, self.car_yaw = 0.5, 0.5, 0.0
        self.current_speed, self.current_steer = 0.0, 0.0
        self.keys = {'Up': False, 'Down': False, 'Left': False, 'Right': False}
        self.last_ctrl_time = time.time()
        self.current_hz = 0.0
        self.last_target_speed = 0.0
        self.last_target_steer = 0.0
        self.is_calibrating = False
        self.last_logged_cmd = None

        # ADAS State
        self.in_highway_mode = False
        self.last_lane_conf = 0.0
        self.crosswalk_timer = 0.0
        self.priority_timer = 0.0
        self.bus_lane_active = False

        # Routing State
        self.mode = "DRIVE"
        self.start_node = None; self.end_node = None; self.pass_nodes = []
        self.path = []
        self.visited_path_nodes = set()
        self.path_signs = []

        # Parking FSM State
        self.is_playing_back = False
        self.is_parking_reverse_mode = False
        self.playback_cmd = None
        self.playback_frames = 0
        self.playback_queue = []

        self.path_distance = 0.0
        self.last_path_tuple = None

        self.telemetry_logger = TelemetryLogger(buffer_size=50)
        self._last_logged_event = "NORMAL"

        # Autonomous Pipelines (Lane Detection)
        self.is_auto_mode    = False
        self.auto_start_time = 0.0

        self.camera = Camera(sim_video=None)
        self.detector = LaneDetector() if _AUTO_DRIVE_AVAILABLE else None
        self.controller = Controller() if _AUTO_DRIVE_AVAILABLE else None

        # Localization App Setup
        self.pose_queue = queue.Queue(maxsize=1)
        self.localizer_engine = RealtimeLocalizer(self.pose_queue, self.imu, self.camera, self.handler)
        self.loc_thread = threading.Thread(target=self.localizer_engine.run, daemon=True)
        self.loc_thread.start()

        self.traffic_engine, self.behavior, self.yolo = None, None, None
        if _AI_AVAILABLE:
            try:
                self.yolo = ThreadedYOLODetector(YOLO_MODEL_FILE)
                self.traffic_engine = TrafficDecisionEngine(self.yolo)
                self.behavior = BehaviorController()
            except Exception as e:
                print(f"[SYS] Warning: Failed to load AI models: {e}")

        # Mini Telemetry Dashboard (Tela)
        if not self.headless:
            self.tela = MiniTelemetryDashboard(self.root)
            self.tela.hide()  # Hidden by default; toggle with "T" key
            self.root.bind("<t>", lambda e: self.tela.show())
            self.root.bind("<T>", lambda e: self.tela.show())

        # Bindings & Loops
        if not self.headless:
            self.root.bind("<KeyPress>", self._on_key_press)
            self.root.bind("<KeyRelease>", self._on_key_release)
            self.ui.map_canvas.bind("<Button-1>", self.on_map_click)

        self.set_mode("DRIVE")
        self.control_loop()

        if not self.headless:
            self.render_map()

    def set_mode(self, m):
        self.mode = m
        if self.headless: return
        self.ui.var_main_mode.set(m)
        for w in self.ui.tool_frame.winfo_children():
            w.destroy()
        if m == "NAV":
            self.ui.build_nav_tools(self)
        elif m == "SIGN":
            self.ui.build_sign_tools(self)
            tk.Label(self.ui.tool_frame, text="Right-Click a node to DELETE sign",
                     bg=THEME["panel"], fg="yellow", font=THEME["font_p"]).pack(side=tk.LEFT, padx=10)
        else:
            tk.Label(self.ui.tool_frame,
                     text="Drive Mode Active - Click Map to Teleport Digital Twin",
                     bg=THEME["panel"], fg=THEME["success"],
                     font=THEME["font_h"]).pack(side=tk.LEFT, padx=10, pady=5)

    def _get_nearest_node(self, event):
        cx = self.ui.map_canvas.canvasx(event.x)
        cy = self.ui.map_canvas.canvasy(event.y)
        nearest_node = None
        min_dist = float('inf')
        for node_id, (px, py) in self.map_engine.node_pixels.items():
            dist = math.hypot(cx - px, cy - py)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id
        return nearest_node

    def on_map_click(self, event):
        if self.headless: return
        nearest_node = self._get_nearest_node(event)
        if not nearest_node: return

        if self.mode == "DRIVE":
            node_data = self.map_engine.G.nodes[nearest_node]
            self.car_x = float(node_data.get('x', self.car_x))
            self.car_y = float(node_data.get('y', self.car_y))
            self.render_map()

        elif self.mode == "NAV":
            nav_action = self.ui.var_path.get() if hasattr(self.ui, 'var_path') else "START"
            if nav_action == "START":
                self.start_node = nearest_node
                self.ui.log_event(f"Start Node Set: {nearest_node}", "SUCCESS")
            elif nav_action == "PASS":
                self.pass_nodes.append(nearest_node)
                self.ui.log_event(f"Pass Node Added: {nearest_node}", "SUCCESS")
            elif nav_action == "END":
                self.end_node = nearest_node
                self.ui.log_event(f"End Node Set: {nearest_node}", "SUCCESS")
            if self.start_node and self.end_node:
                self.path = self.map_engine.calc_path_nodes(self.start_node, self.end_node, self.pass_nodes)
                self.path_signs = self.map_engine.get_path_signs(self.path)
                self.ui.log_event(f"Path Calculated. {len(self.path_signs)} signs on route.", "SUCCESS")
            self.render_map()

        elif self.mode == "SIGN":
            is_delete_mode = hasattr(self.ui, 'chk_del') and self.ui.chk_del.get()
            if is_delete_mode:
                if self.map_engine.remove_sign(nearest_node):
                    self.ui.log_event(f"Sign deleted at Node: {nearest_node}", "WARN")
            else:
                sign_type = "stop-sign"
                if hasattr(self.ui, 'var_sign'):
                    sign_type = self.ui.var_sign.get()
                self.map_engine.remove_sign(nearest_node)
                x_val = float(self.map_engine.G.nodes[nearest_node].get('x', 0.0))
                y_val = float(self.map_engine.G.nodes[nearest_node].get('y', 0.0))
                new_sign = {"node": nearest_node, "type": sign_type, "x": x_val, "y": y_val}
                self.map_engine.signs.append(new_sign)
                self.map_engine.save_signs()
                self.ui.log_event(f"Sign '{sign_type}' placed at Node: {nearest_node}", "SUCCESS")
            self.render_map()

    def execute_parking_playback(self, reverse=False):
        filename = "default_parking.csv"
        if not os.path.exists(filename):
            if not self.headless:
                self.ui.log_event(f"Error: {filename} not found!", "WARN")
            return
        commands = []
        try:
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    commands.append({
                        "speed": float(row.get("speed", 0.0)),
                        "steer": float(row.get("steering", row.get("steer", 0))),
                        "pwm": float(row.get("pwm", 0.0)),
                        "direction": int(row.get("direction", 1)),
                        "duration_fr": int(row.get("duration_fr", 1))
                    })
        except Exception as e:
            if not self.headless:
                self.ui.log_event(f"CSV Read Error: {e}", "DANGER")
            return

        self.playback_queue = []
        if reverse:
            for cmd in reversed(commands):
                self.playback_queue.append({
                    "speed": cmd["speed"],
                    "steer": -cmd["steer"],
                    "pwm": cmd["pwm"],
                    "direction": -1 if cmd["direction"] == 1 else 1,
                    "duration_fr": cmd["duration_fr"]
                })
            self.is_parking_reverse_mode = True
        else:
            self.playback_queue = commands
            self.is_parking_reverse_mode = False

        self.is_playing_back = True
        self.is_auto_mode = False
        self.is_calibrating = False
        if not self.headless:
            mode_str = "REVERSE" if reverse else "FORWARD"
            self.ui.log_event(f"Starting {mode_str} parking playback...", "SUCCESS")

    def render_map(self):
        if self.headless: return
        pil = self.map_engine.render_map(
            self.car_x, self.car_y, self.car_yaw,
            self.path, self.visited_path_nodes, self.path_signs,
            True, self.start_node, self.pass_nodes, self.end_node
        )
        self.tk_map = ImageTk.PhotoImage(pil)
        self.ui.map_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_map)
        self.ui.map_canvas.config(scrollregion=self.ui.map_canvas.bbox(tk.ALL))

    def toggle_connection(self):
        if not self.is_connected:
            if self.handler.connect():
                self.is_connected = True
                if not self.headless:
                    self.ui.lbl_conn.config(text="CONNECTED", fg=THEME["success"])
                    self.ui.btn_connect.config(text="DISCONNECT", bg=THEME["danger"])
                    self.ui.log_event("Connected to STM32 Hardware successfully.", "SUCCESS")
        else:
            self.handler.disconnect(); self.is_connected = False
            if not self.headless:
                self.ui.log_event("Disconnected from STM32.", "WARN")
                self.ui.lbl_conn.config(text="DISCONNECTED", fg=THEME["danger"])
                self.ui.btn_connect.config(text="CONNECT CAR", bg=THEME["accent"])

    def save_config(self):
        """Save configuration stub (kept for dashboard compatibility)."""
        if not self.headless:
            self.ui.log_event("Configuration saved.", "SUCCESS")

    def load_config(self):
        """Load configuration stub (kept for dashboard compatibility)."""
        if not self.headless:
            self.ui.log_event("Configuration loaded.", "SUCCESS")

    def toggle_adas_mode(self):
        """Simple ADAS toggle stub (kept for dashboard compatibility)."""
        self.in_highway_mode = not self.in_highway_mode
        if not self.headless:
            if self.in_highway_mode:
                self.ui.btn_adas.config(text="ADAS ASSIST: ON", bg="#9b59b6")
                self.ui.log_event("ADAS ASSIST enabled.", "SUCCESS")
            else:
                self.ui.btn_adas.config(text="ADAS ASSIST: OFF", bg="#444")
                self.ui.log_event("ADAS ASSIST DISABLED.", "WARN")

    def toggle_auto_mode(self):
        self.is_auto_mode = not self.is_auto_mode
        self.is_playing_back = False
        if self.is_auto_mode:
            self.auto_start_time = time.time()
            self.is_calibrating  = True
            for k in self.keys: self.keys[k] = False
            if not self.headless:
                self.ui.btn_auto.config(text="MODE: AUTONOMOUS", bg="#9b59b6")
                self.ui.log_event("Switched to AUTONOMOUS. Calibrating 5s ...", "SUCCESS")
        else:
            self.is_calibrating  = False
            for k in self.keys: self.keys[k] = False
            if not self.headless:
                self.ui.btn_auto.config(text="MODE: MANUAL", bg="#444")
                self.ui.log_event("Switched to MANUAL mode.", "WARN")

    # ─────────────────────────────────────────────────────────
    # CONTROL LOOP  (20 Hz)
    # ─────────────────────────────────────────────────────────
    def control_loop(self):
        now = time.time()
        dt = max(now - self.last_ctrl_time, 0.001)
        loop_start = now  # snapshot BEFORE processing (fixes loop_time_ms bug)
        self.last_ctrl_time = now

        # Update pose from particle filter localizer
        try:
            pose = self.pose_queue.get_nowait()
            self.car_x = pose['x']
            self.car_y = pose['y']
            self.car_yaw = pose['heading']
            self.loc_confidence = pose.get('confidence', 0)
            self.loc_spread = pose.get('spread_m', 0)
        except queue.Empty:
            pass

        base_speed = float(self.ui.slider_base_speed.get() if not self.headless else 50.0)
        steer_mult = float(self.ui.slider_steer_mult.get() if not self.headless else 1.0)

        target_speed, target_steer = 0.0, 0.0
        ai_labels = []  # initialize to avoid UnboundLocalError when frame is None

        # 1. Grab Frame
        frame = self.camera.read_frame()
        lane_result = None
        t_res = None
        behav_out = None
        active_sign_cmd = None

        if frame is not None:
            # MAP BUILD RECORDING
            if getattr(self, 'is_recording_map', False):
                t = time.time() - self.record_start_time
                if not hasattr(self, 'frame_idx'): self.frame_idx = 0
                if self.frame_idx % 3 == 0:
                    cv2.imwrite(str(self.map_record_dir / 'frames' / f'frame_{self.frame_idx//3:06d}.jpg'), frame)
                
                self.imu_rows.append([t, self.imu.get_yaw(), getattr(self.imu, 'pitch', 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                self.rc_rows.append([t, self.current_steer, self.current_speed])
                self.frame_idx += 1

        if frame is not None and self.detector and self.controller:
            # 2. Process Lane Detection
            yaw_deg = self.imu.get_yaw()
            lane_result = self.detector.process(
                frame, dt=dt, velocity_ms=max(self.current_speed / 1000.0, 0.0),
                last_steering=self.current_steer, current_yaw=yaw_deg
            )

            # Pass lateral error to Localizer (convert pixels to roughly metres)
            if hasattr(self, 'localizer_engine') and lane_result is not None:
                err_m = lane_result.lateral_error_px * 0.002
                self.localizer_engine.update_lane_error(err_m)

            # --- Pass Optical Flow to IMU Fusion ---
            self.imu.update_optical_velocity(lane_result.optical_vel)

            # 3. Process AI & Semantic Traffic Rules
            ai_labels = []
            if self.traffic_engine:
                line_type = getattr(lane_result, 'lane_type', 'UNKNOWN')
                t_res = self.traffic_engine.process(frame, line_type)
                if t_res and hasattr(t_res, 'active_labels'):
                    ai_labels = t_res.active_labels

            # Bus lane detection check
            self.bus_lane_active = any("bus" in str(l).lower() for l in ai_labels)

            # 4. Update Path Sign States (Distance + AI Vision)
            detect_dist = float(self.ui.slider_sign_detect.get() if not self.headless else 5.0)
            act_dist = float(self.ui.slider_sign_act.get() if not self.headless else 2.0)
            ai_dist = getattr(t_res, 'sign_approach_m', 99.0) if t_res else 99.0
            light_status = getattr(t_res, 'light_status', 'NONE') if t_res else 'NONE'

            self.active_blocks = {
                "crosswalk": time.time() < self.crosswalk_timer,
                "priority": time.time() < self.priority_timer,
                "pedestrian": any(label.lower() in ["pedestrian", "person"] for label in ai_labels),
                "parking": getattr(self, 'is_playing_back', False) or getattr(self, 'is_waiting_for_reverse', False)
            }

            teleport_node = None
            if not self.is_playing_back and not getattr(self, 'is_waiting_for_reverse', False):
                active_sign_cmd, self.path_signs, teleport_node = self.map_engine.update_sign_statuses(
                    self.path_signs, ai_labels, ai_dist, detect_dist=detect_dist, act_dist=act_dist,
                    light_status=light_status, active_blocks=self.active_blocks
                )

            # --- TELEPORT CAR TO COMPLETED SIGN ---
            if teleport_node and teleport_node in self.map_engine.G.nodes:
                node_data = self.map_engine.G.nodes[teleport_node]
                self.car_x = float(node_data.get('x', self.car_x))
                self.car_y = float(node_data.get('y', self.car_y))
                if self.path and len(self.path) > 1:
                    acc_dist = 0.0
                    for i in range(len(self.path) - 1):
                        n1 = str(self.path[i])
                        if n1 == str(teleport_node):
                            self.path_distance = acc_dist
                            break
                        n2 = str(self.path[i+1])
                        if n1 in self.map_engine.G.nodes and n2 in self.map_engine.G.nodes:
                            x1 = float(self.map_engine.G.nodes[n1].get('x', 0))
                            y1 = float(self.map_engine.G.nodes[n1].get('y', 0))
                            x2 = float(self.map_engine.G.nodes[n2].get('x', 0))
                            y2 = float(self.map_engine.G.nodes[n2].get('y', 0))
                            acc_dist += math.hypot(x2 - x1, y2 - y1)
                if not self.headless:
                    self.ui.log_event(f"Teleported to completed sign node: {teleport_node}", "SUCCESS")

            # --- OVERRIDE TIMERS ---
            if active_sign_cmd:
                if "crosswalk" in active_sign_cmd.lower() or "pedestrian" in active_sign_cmd.lower():
                    self.crosswalk_timer = time.time() + 5.0
                elif "priority" in active_sign_cmd.lower():
                    self.priority_timer = time.time() + 10.0

            if active_sign_cmd and not self.headless:
                if active_sign_cmd != self.last_logged_cmd:
                    self.ui.log_event(f"Responding to active sign: {active_sign_cmd}", "WARN")
                    self.last_logged_cmd = active_sign_cmd
            elif not active_sign_cmd:
                self.last_logged_cmd = None

            # 5. Calculate Steering & Speed Control
            if self.is_auto_mode and not self.is_playing_back:
                if time.time() - self.auto_start_time > 5.0 and self.imu.is_calibrated:
                    self.is_calibrating = False
                    fused_vel_pwm = self.imu.get_fused_velocity()
                    fused_vel_ms = max(fused_vel_pwm / 1000.0, 0.0)
                    ctrl_out = self.controller.compute(lane_result, velocity_ms=fused_vel_ms,
                                                        base_speed=base_speed, dt=dt)
                    target_speed = ctrl_out.speed_pwm
                    target_steer = ctrl_out.steer_angle_deg * steer_mult

                    if self.behavior:
                        behav_out = self.behavior.compute(
                            lane_result, t_res, dt, base_speed=base_speed,
                            base_steer=ctrl_out.steer_angle_deg, current_yaw=yaw_deg
                        )
                        target_speed = behav_out.speed_pwm
                        target_steer = behav_out.steer_deg

                    # --- TELEMETRY CSV LOGGER ---
                    if hasattr(self, 'telemetry_logger'):
                        loop_time_ms = (time.time() - loop_start) * 1000.0
                        fps = 1000.0 / max(loop_time_ms, 1.0)

                        event_str = active_sign_cmd if active_sign_cmd else "NORMAL"
                        if lane_result.confidence < 0.3:
                            event_str = f"SAFE_MODE ({event_str})"
                        if lane_result.anchor != "LANE_FOLLOW":
                            event_str = f"LOST_LANE ({event_str})"

                        force_flush = False
                        if "SAFE_MODE" in event_str or "LOST_LANE" in event_str:
                            force_flush = True
                        if lane_result.confidence < 0.2:
                            force_flush = True
                        if event_str != self._last_logged_event:
                            force_flush = True
                        self._last_logged_event = event_str

                        steer_delta = target_steer - self.last_target_steer
                        speed_delta = target_speed - self.last_target_speed
                        conf_delta = lane_result.confidence - self.last_lane_conf
                        latency_flag = loop_time_ms > 80.0
                        steer_saturated = abs(target_steer) >= 29.0

                        root_cause = "NORMAL"
                        if lane_result.confidence < 0.2:
                            root_cause = "VISION_FAILURE"
                        elif latency_flag:
                            root_cause = "LATENCY_SPIKE"
                        elif abs(steer_delta) > 10.0:
                            root_cause = "CONTROL_INSTABILITY"

                        self.last_target_steer = target_steer
                        self.last_target_speed = target_speed
                        self.last_lane_conf = lane_result.confidence

                        self.telemetry_logger.log([
                            round(time.time(), 3),
                            "AUTO",
                            event_str,
                            round(lane_result.confidence, 2),
                            round(conf_delta, 3),
                            round(lane_result.target_x, 1),
                            round(lane_result.curvature, 5),
                            round(yaw_deg, 2),
                            round(self.imu.get_pitch(), 2),
                            round(fused_vel_pwm, 1),
                            round(target_steer, 1),
                            round(target_speed, 1),
                            round(steer_delta, 1),
                            round(speed_delta, 1),
                            steer_saturated,
                            lane_result.confidence < 0.3,
                            lane_result.anchor,
                            round(fps, 1),
                            round(loop_time_ms, 1),
                            latency_flag,
                            root_cause
                        ], force_flush=force_flush)

                        # --- TELEMETRY TELA update ---
                        if not self.headless:
                            self.tela.update(
                                lane_type=lane_result.lane_type,
                                anchor=lane_result.anchor,
                                conf=lane_result.confidence,
                                curvature=lane_result.curvature,
                                bus_lane_active=self.bus_lane_active,
                                speed=target_speed,
                                steer=target_steer,
                                yaw=yaw_deg,
                                mode="AUTONOMOUS",
                                hz=fps,
                                latency_ms=loop_time_ms,
                                ai_count=len(ai_labels) if ai_labels else 0,
                                root_cause=root_cause
                            )

                        # --- SPEED MULTIPLIER AND IMMEDIATE OVERRIDE LOGIC ---
                        if active_sign_cmd and "highway" in active_sign_cmd.lower():
                            if "entry" in active_sign_cmd.lower():
                                self.in_highway_mode = True
                            elif "exit" in active_sign_cmd.lower():
                                self.in_highway_mode = False

                        is_highway = False
                        if self.in_highway_mode or "highway" in getattr(behav_out, "zone_mode", "").lower():
                            is_highway = True

                        # Hard overrides (absolute halt)
                        halt_cmds = ["red-light"]
                        if active_sign_cmd in halt_cmds or (active_sign_cmd == "traffic-light" and "GREEN" not in light_status):
                            target_speed = 0.0

                        # Dynamic Pedestrian Halt
                        if any(label.lower() in ["pedestrian", "person"] for label in ai_labels):
                            target_speed = 0.0

                        # Multipliers
                        if time.time() < self.crosswalk_timer:
                            target_speed *= 0.8
                        if time.time() < self.priority_timer:
                            target_speed *= 0.8
                        if is_highway:
                            target_speed *= 1.3

                        # --- IMU Pitch Gravity Control ---
                        pitch = self.imu.get_pitch()
                        if pitch > 5.0:
                            pitch_correction = max(0.0, min(30.0, pitch * 0.4))
                            target_speed += pitch_correction
                        elif pitch < -5.0:
                            pitch_correction = max(0.0, min(40.0, abs(pitch) * 0.3))
                            target_speed -= pitch_correction

                        if behav_out.speed_pwm > 0 and target_speed < 0:
                            target_speed = 0.0
                else:
                    self.is_calibrating = True
                    target_speed, target_steer = 0.0, 0.0

            # 6. Dashboard CAM + BEV Render
            if not self.headless:
                final_cam = t_res.yolo_debug_frame if (t_res and getattr(t_res, 'yolo_debug_frame', None) is not None) else frame
                if final_cam is not None:
                    final_cam = cv2.cvtColor(final_cam, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(final_cam).resize((440, 330))
                    self.ui.cam_label.imgtk = ImageTk.PhotoImage(image=img)
                    self.ui.cam_label.configure(image=self.ui.cam_label.imgtk)

                if lane_result is not None and hasattr(lane_result, 'lane_dbg') and lane_result.lane_dbg is not None:
                    dbg = lane_result.lane_dbg.copy()

                    cv2.putText(dbg, lane_result.anchor, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(dbg, f"Target X: {lane_result.target_x:.1f}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(dbg, f"Lat Error: {lane_result.lateral_error_px:+.1f}px", (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                    cv2.putText(dbg, f"Type: {lane_result.lane_type}", (10, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

                    steer_color = (100, 255, 100) if abs(self.current_steer) < 15 else (100, 100, 255)
                    cv2.putText(dbg, f"STEER: {self.current_steer:+.1f} deg", (420, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, steer_color, 2)
                    cv2.putText(dbg, f"SPEED: {self.current_speed:.0f} PWM", (420, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)

                    if t_res is not None and behav_out is not None:
                        cv2.putText(dbg, f"STATE: {behav_out.state}", (420, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        cv2.putText(dbg, f"ZONE: {behav_out.zone_mode}", (420, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 255), 2)
                        y_offset = 120
                        if ai_labels:
                            cv2.putText(dbg, "YOLO Detections:", (10, 115),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            for label in ai_labels:
                                cv2.putText(dbg, f"- {label}", (10, y_offset),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                                y_offset += 20

                    if self.bus_lane_active:
                        cv2.putText(dbg, "BUS LANE AVOID!", (200, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                    bev = cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)
                    img_bev = Image.fromarray(bev).resize((440, 330))
                    self.ui.bev_label.imgtk = ImageTk.PhotoImage(image=img_bev)
                    self.ui.bev_label.configure(image=self.ui.bev_label.imgtk)

        # ── PARKING PLAYBACK OVERRIDE ─────────────────────────
        if self.is_playing_back:
            self.is_calibrating = False
            if self.playback_cmd is None or self.playback_frames <= 0:
                if self.playback_queue:
                    self.playback_cmd = self.playback_queue.pop(0)
                    self.playback_frames = self.playback_cmd.get("duration_fr", 1)
                else:
                    self.playback_cmd = None
            if self.playback_cmd:
                self.playback_frames -= 1
                cmd = self.playback_cmd
                dir_mult = cmd.get("direction", 1)
                if dir_mult == 0: dir_mult = 1
                if cmd.get("pwm", 0) > 0:
                    target_speed = cmd["pwm"] * dir_mult
                else:
                    target_speed = cmd["speed"] * dir_mult
                target_steer = cmd["steer"]
                if not getattr(self, '_last_park_log_time', 0) or time.time() - self._last_park_log_time > 1.0:
                    self._last_park_log_time = time.time()
                    if not self.headless:
                        self.ui.log_event(f"Parking Steer: {target_steer:.1f} | Spd: {target_speed:.1f}", "INFO")
            else:
                is_finishing_reverse = self.is_parking_reverse_mode
                self.is_playing_back = False
                self.is_parking_reverse_mode = False
                target_speed = 0.0
                target_steer = 0.0
                if not is_finishing_reverse and hasattr(self.ui, 'chk_parking') and self.ui.chk_parking.get():
                    self.is_waiting_for_reverse = True
                    self.reverse_timer = time.time() + 10.0
                    if not self.headless:
                        self.ui.log_event("Parking reached. Waiting 10s for Auto-Reverse...", "WARN")
                else:
                    if not self.headless:
                        self.ui.log_event("Parking sequence fully complete.", "SUCCESS")

        elif getattr(self, 'is_waiting_for_reverse', False):
            self.is_calibrating = False
            target_speed = 0.0
            target_steer = 0.0
            if time.time() > self.reverse_timer:
                self.is_waiting_for_reverse = False
                self.execute_parking_playback(reverse=True)

        # ── MANUAL OVERRIDES ──────────────────────────────────
        elif not self.is_auto_mode:
            self.is_calibrating = False
            target_speed = (base_speed if self.keys['Up'] else (-base_speed if self.keys['Down'] else 0))
            target_steer = (-25 * steer_mult if self.keys['Left'] else (25 * steer_mult if self.keys['Right'] else 0))

        # ── SMOOTH APPLICATION ────────────────────────────────
        if target_speed == 0:
            self.current_speed = 0.0
        else:
            self.current_speed += (target_speed - self.current_speed) * 0.2

        if target_steer == 0:
            self.current_steer = 0.0
        else:
            self.current_steer += (target_steer - self.current_steer) * 0.2

        # ── HARDWARE OUTPUT ───────────────────────────────────
        if self.is_connected:
            if not self.imu.is_calibrated and self.is_auto_mode:
                self.handler.set_speed(0)
                self.handler.set_steering(0)
            else:
                self.handler.set_speed(int(self.current_speed))
                self.handler.set_steering(self.current_steer)

        # ── V2X TELEMETRY PUSH ────────────────────────────────
        self.v2x_client.update_state(
            x=self.car_x, y=self.car_y, yaw=self.imu.get_yaw(), speed=self.current_speed
        )

        # ── KINEMATICS SIMULATION (Map update) ────────────────
        if abs(self.current_speed) < 1:  self.current_speed = 0
        if abs(self.current_steer) < 0.5: self.current_steer = 0

        sim_mult = float(self.ui.slider_sim_speed.get() if not self.headless else 1.0)
        v_ms = (self.current_speed / 1000.0) * sim_mult * 1.5

        if self.path and len(self.path) > 1:
            path_tuple = tuple(self.path)
            if self.last_path_tuple != path_tuple:
                self.last_path_tuple = path_tuple
                self.path_distance = 0.0
            self.path_distance += v_ms * dt
            if self.path_distance < 0: self.path_distance = 0

            acc_dist = 0.0
            found_segment = False
            for i in range(len(self.path) - 1):
                n1 = str(self.path[i])
                n2 = str(self.path[i+1])
                if n1 not in self.map_engine.G.nodes or n2 not in self.map_engine.G.nodes:
                    continue
                x1 = float(self.map_engine.G.nodes[n1].get('x', 0))
                y1 = float(self.map_engine.G.nodes[n1].get('y', 0))
                x2 = float(self.map_engine.G.nodes[n2].get('x', 0))
                y2 = float(self.map_engine.G.nodes[n2].get('y', 0))
                seg_len = math.hypot(x2 - x1, y2 - y1)
                if self.path_distance <= acc_dist + seg_len:
                    ratio = (self.path_distance - acc_dist) / seg_len if seg_len > 0 else 0
                    self.car_x = x1 + ratio * (x2 - x1)
                    self.car_y = y1 + ratio * (y2 - y1)
                    self.car_yaw = math.atan2(y2 - y1, x2 - x1)
                    found_segment = True
                    self.visited_path_nodes.add(n1)
                    break
                acc_dist += seg_len
            if not found_segment:
                n_end = str(self.path[-1])
                if n_end in self.map_engine.G.nodes:
                    self.car_x = float(self.map_engine.G.nodes[n_end].get('x', 0))
                    self.car_y = float(self.map_engine.G.nodes[n_end].get('y', 0))
                self.current_speed = 0.0
        else:
            steer_rad = math.radians(self.current_steer)
            self.car_yaw -= (v_ms / max(WHEELBASE_M, 0.01)) * math.tan(steer_rad) * dt
            self.car_yaw  = (self.car_yaw + math.pi) % (2 * math.pi) - math.pi
            self.car_x   += v_ms * math.cos(self.car_yaw) * dt
            self.car_y   += v_ms * math.sin(self.car_yaw) * dt
            self.path_distance = 0.0
            self.last_path_tuple = None

        # ── UI UPDATES ────────────────────────────────────────
        if not self.headless:
            hz = 1.0 / dt if dt > 0 else 0.0
            self.current_hz = 0.8 * self.current_hz + 0.2 * hz
            self.ui.lbl_hz.config(text=f"{self.current_hz:.1f} Hz", fg="cyan")

            mode_str = "AUTONOMOUS" if self.is_auto_mode else "MANUAL"
            if self.is_playing_back: mode_str = "REVERSE PARKING" if self.is_parking_reverse_mode else "PARKING PLAYBACK"
            if self.is_calibrating: mode_str = "CALIBRATING..."

            self.ui.lbl_telemetry.config(
                text=f"SPD: {int(self.current_speed)} | STR: {self.current_steer:.1f} | LMT: {base_speed} | [{mode_str}]"
            )

            # --- Update Indicators ---
            active_keys = []
            if active_sign_cmd:
                cmd_l = active_sign_cmd.lower()
                if 'stop' in cmd_l: active_keys.append('stop_sign')
                elif 'no_entry' in cmd_l or 'no-entry' in cmd_l: active_keys.append('no_entry')
                elif 'pedestrian' in cmd_l or 'crosswalk' in cmd_l: active_keys.append('pedestrian')
                elif 'highway' in cmd_l: active_keys.append('highway')
                elif 'park' in cmd_l: active_keys.append('park')
                else: active_keys.append('caution')

            if getattr(self, 'active_blocks', None):
                if self.active_blocks.get('crosswalk') or self.active_blocks.get('pedestrian'):
                    if 'pedestrian' not in active_keys: active_keys.append('pedestrian')
                if self.active_blocks.get('priority'):
                    if 'caution' not in active_keys: active_keys.append('caution')

            if getattr(self, 'in_highway_mode', False) and 'highway' not in active_keys:
                active_keys.append('highway')

            if behav_out:
                ls = getattr(behav_out, 'light_status', '')
                if 'RED' in ls: active_keys.append('red_light')
                elif 'YELLOW' in ls: active_keys.append('yellow_light')
                elif 'GREEN' in ls: active_keys.append('green_light')
                if getattr(behav_out, 'parking_state', 'NONE') not in ('NONE', 'DONE'): active_keys.append('park')
                if getattr(behav_out, 'state', '') == 'SYS_LANE_CHANGE_LEFT': active_keys.append('overtake')
                if getattr(behav_out, 'zone_mode', '') == 'HIGHWAY': active_keys.append('highway')

            if self.bus_lane_active and 'caution' not in active_keys:
                active_keys.append('caution')

            self.ui.update_indicators(active_keys)

            # HUD update
            parking_state = 'NONE'
            if behav_out is not None:
                parking_state = getattr(behav_out, 'parking_state', parking_state)
            elif t_res is not None:
                parking_state = getattr(t_res, 'parking_state', parking_state)

            lane_status = 'LOST' if (lane_result is None or getattr(lane_result, 'lost', False)) else 'OK'
            ai_status = "ON" if ai_labels else "OFF"
            imu_yaw_val = self.imu.get_yaw()

            try:
                imu_connected = self.imu.get_has_hardware()
                self.ui.update_hud(speed=self.current_speed, steer=self.current_steer, mode=mode_str,
                                    parking_state=parking_state, imu_yaw=imu_yaw_val,
                                    imu_connected=imu_connected, lane_status=lane_status,
                                    ai_status=ai_status)
            except Exception:
                pass

            self.render_map()

        if self.headless:
            print(f"[CTRL] Spd:{int(self.current_speed):4d} | Str:{self.current_steer:5.1f} "
                  f"| Yaw:{self.imu.get_yaw():5.1f} | Pos:({self.car_x:.1f},{self.car_y:.1f}) "
                  f"| {1/dt:.0f}Hz", end="\r")

        # Non-blocking reschedule — works for BOTH headless and GUI modes
        self.root.after(50, self.control_loop)

    # ========================================================
    # LOCALIZATION MAPPING PIELINE
    # ========================================================
    def toggle_map_recording(self):
        self.is_recording_map = getattr(self, 'is_recording_map', False)
        
        if not self.is_recording_map:
            import shutil, pathlib
            self.map_record_dir = pathlib.Path("recordings/gui_run")
            if self.map_record_dir.exists():
                shutil.rmtree(self.map_record_dir, ignore_errors=True)
            self.map_record_dir.mkdir(parents=True, exist_ok=True)
            (self.map_record_dir / 'frames').mkdir(exist_ok=True)
            
            self.record_start_time = time.time()
            self.imu_rows = []
            self.rc_rows = []
            self.frame_idx = 0
            self.is_recording_map = True
            
            if not self.headless:
                self.ui.btn_rec_map.config(text="⏹ STOP RECORDING", bg="black")
                self.ui.log_event("Started Mapping Recording...")
        else:
            self.is_recording_map = False
            import pandas as pd
            pd.DataFrame(self.imu_rows, columns=['t','yaw','pitch','roll','accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z']).to_csv(self.map_record_dir / 'imu.csv', index=False)
            pd.DataFrame(self.rc_rows, columns=['t','steering_deg','speed_mms']).to_csv(self.map_record_dir / 'rc.csv', index=False)
            if not self.headless:
                from config import THEME
                self.ui.btn_rec_map.config(text="🔴 START RECORD MANUAL DRIVE", bg=THEME["danger"])
                self.ui.log_event(f"Saved {self.frame_idx//3} frames and telemetry.", "SUCCESS")

    def build_visual_map(self):
        if not hasattr(self, 'map_record_dir') or not self.map_record_dir.exists():
            if not self.headless: self.ui.log_event("No GUI run recording found! Record first.", "CRITICAL")
            return
            
        def worker():
            if not self.headless: self.ui.log_event("Starting map solve and graph extraction... This may take up to 30s.", "WARN")
            import subprocess, sys
            res = subprocess.run([sys.executable, "-m", "localization.build_map_pipeline"], capture_output=True, text=True)
            if not self.headless:
                if res.returncode != 0:
                    self.ui.log_event("Map Compilation FAILED:", "CRITICAL")
                    for line in res.stderr.split('\n'):
                        if line.strip(): self.ui.log_event(line, "CRITICAL")
                else:
                    self.ui.log_event("Map Compilation SUCCESS!", "SUCCESS")
                    for line in res.stdout.split('\n'):
                        if line.strip() and "[MAP BUILDER]" in line: self.ui.log_event(line, "INFO")
                    
        import threading
        threading.Thread(target=worker, daemon=True).start()

    def clear_route(self):
        self.start_node = None; self.end_node = None; self.pass_nodes = []; self.path = []
        self.path_signs = []
        self.visited_path_nodes.clear()
        if not self.headless:
            for item in self.ui.tree.get_children():
                self.ui.tree.delete(item)
            self.render_map()
            self.ui.log_event("Route & sign history cleared. Ready for new run.", "WARN")

    def _on_key_press(self, e):
        if self.is_auto_mode or self.is_playing_back: return
        if e.keysym in self.keys: self.keys[e.keysym] = True

    def _on_key_release(self, e):
        if e.keysym in self.keys: self.keys[e.keysym] = False

    def on_close(self):
        if hasattr(self, 'localizer_engine') and self.localizer_engine:
            self.localizer_engine.shutdown()
        self.camera.stop()
        if self.yolo: self.yolo.stop()
        if self.is_connected:
            self.handler.set_speed(0)
            self.handler.set_steering(0)
            self.handler.disconnect()
        self.imu.stop()
        self.v2x_client.stop()
        if not self.headless:
            self.root.destroy()


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BFMC 2026 Unified Autonomous Stack")
    parser.add_argument("--headless", action="store_true", help="Run in terminal only, no Tkinter GUI")
    parser.add_argument("--no-v2x", action="store_true", help="Do not start V2X client")
    args = parser.parse_args()

    try:
        if args.headless:
            class FakeRoot:
                def after(self, ms, cb):
                    pass
            app = BFMC_App(FakeRoot(), args)
        else:
            root = tk.Tk()
            app = BFMC_App(root, args)
            root.protocol("WM_DELETE_WINDOW", app.on_close)
            root.mainloop()
    except KeyboardInterrupt:
        print("\n[SYS] Interrupted by user.")
    except Exception as e:
        import traceback
        print("\n[SYS] FATAL ERROR:")
        traceback.print_exc()
    finally:
        if not args.headless:
            try: app.on_close()
            except: pass