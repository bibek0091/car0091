# Autonomous Driving Stack Architecture & Technical Documentation
**Platform**: Raspberry Pi 5 / STM32 (Mini Tesla / BFMC)
**Architecture**: Visual-Inertial Odometry + Semantic Anchoring (ADAS)

---

# 🧩 SECTION 1: SYSTEM OVERVIEW

The BFMC (Bosch Future Mobility Challenge) autonomous driving system is a highly robust, relative-localization-based ADAS (Advanced Driver Assistance System). Unlike traditional systems that rely heavily on absolute global positioning (GPS/SLAM), this system is built on **Visual-Inertial Odometry (VIO) coupled with Semantic Anchoring**. 

It answers three fundamental questions at 30Hz:
1. **Where am I relative to the lane?** (Lateral Control via Vision)
2. **How is my motion changing?** (Forward Velocity & Yaw via IMU + Optical Flow)
3. **What semantic rules apply here?** (Traffic Lights, Crosswalks, Roundabouts via YOLO & Map Graph)

### System Architecture Pipeline

```text
[SENSORS]
   │
   ├── PiCamera (RGB Frames) ──────┐
   │                               │
   └── STM32 (IMU Yaw/Pitch) ──────┤
       (Encoders/PWM limits)       │
                                   ▼
[PERCEPTION LAYER]
   │  ├── LaneDetector (Sobel-X + Perspective Transform)
   │  ├── VisualOdometry (Lucas-Kanade Optical Flow)
   │  └── YOLOv8 / TrafficEngine (Object & Sign Detection)
   │                               │
   ▼                               ▼
[LOCALIZATION LAYER (VIO)]
   │  ├── IMU Sensor Fusion (Complementary Filter)
   │  ├── DeadReckoningNavigator (IMU + Last Motion)
   │  └── Semantic Graph Anchoring (Path Node Teleportation)
   │                               │
   ▼                               ▼
[PLANNING & BEHAVIOR LAYER]
   │  ├── TrafficDecisionEngine (State Machine Manager)
   │  ├── BehaviorController (Speed/Steering Overrides)
   │  └── ParkingSequenceFSM (Closed-loop IMU maneuvering)
   │                               │
   ▼                               ▼
[CONTROL LAYER]
   │  ├── StanleyController (Cross-track & Heading error)
   │  ├── Pitch Compensator (Gravity/Ramp modifiers)
   │  └── Safe Mode Limiter (Panic swerve prevention)
   │                               │
   ▼                               ▼
[ACTUATION]
   └── SerialHandler (STM32 UART ──> Steering Servo & DC Motor PWM)
```

### The Execution Loop (`main.py`)
1. **Capture**: Fetch camera frame and latest IMU telemetry.
2. **Perceive**: Extract lane curvature, center pixel (`target_x`), and optical flow velocity.
3. **Analyze**: Run YOLO to detect traffic signs, cars, pedestrians, and lights.
4. **Localize**: Update dead reckoning state. If vision is lost, fall back to IMU yaw constraints. Fuse optical velocity with motor commands.
5. **Decide**: The `TrafficDecisionEngine` determines the active state (e.g., `ROUNDABOUT`, `PARKING`, `NORMAL`).
6. **Control**: The `StanleyController` computes the exact steering angle required. The speed controller scales PWM based on curvature, IMU pitch, and safe-mode bounds.
7. **Actuate & Log**: Send commands to STM32 and log system state to `telemetry_log.csv`.

---

# 📁 SECTION 2: MODULE-BY-MODULE BREAKDOWN

## 1. `main.py`
### Purpose
The central nervous system. It initializes all threads, orchestrates the main 30Hz `while` loop, and handles hardware I/O dispatching.
### Key Logic
* **Gravity Compensation**: Calculates `pitch_correction = clip(pitch * 0.4, -40, 30)` to dynamically boost speed uphill and apply brakes downhill.
* **Telemetry Logging**: Writes `timestamp, speed_pwm, steer_deg, lane_conf, anchor, fused_vel, pitch, event` to a CSV for post-mortem analysis.

## 2. `perception/lane_detector.py`
### Purpose
Translates raw RGB pixels into a mathematical lane model and tracks motion visually.
### Core Techniques
* **Bird's-Eye View**: Perspective transform (`cv2.warpPerspective`) removes depth distortion.
* **Sobel-X Edge Detection**: Identifies vertical lane boundaries. It is color-agnostic (works on white-on-black or black-on-white).
* **EMA Hysteresis**: Tracks lane confidence. Drops instantly on loss, but requires 3 consecutive solid frames to regain trust.
* **Optical Flow**: Contains `VisualOdometry`, tracking pixel shift to calculate velocity independent of wheel encoders.

## 3. `perception/lane_tracker.py`
### Purpose
Tracks lane polynomials and provides the ultimate Fail-Safe: The `DeadReckoningNavigator`.
### Key Logic
* **Curve vs Straight Split**: If lines are lost in a curve, it blends the last known target with IMU data. If lost on a straight, it forces the target to center and counteracts IMU drift.
* **Motion Consistency Decay**: Triggers a 30% confidence slash if the IMU registers a sudden, violent yaw spin (`> 20 deg/sec`).

## 4. `control/controller.py`
### Purpose
Translates the desired path (`target_x`, `curvature`) into mechanical commands (`steer_angle_deg`, `speed_pwm`).
### Core Techniques
* **Stanley Controller**: A non-linear steering controller that minimizes both cross-track error (distance from lane center) and heading error (alignment with lane).
* **SAFE MODE**: If `confidence < 0.3`, it artificially limits the `MAX_STEER_RATE` to `15.0` (down from 60.0) and halves base speed, physically preventing the car from swerving.

## 5. `hardware/imu_sensor.py`
### Purpose
Runs an asynchronous thread to maintain the freshest IMU data (`yaw`, `pitch`) parsed from the STM32.
### Sensor Fusion
* Contains `get_fused_velocity()`. Uses a Complementary Filter (`alpha = 0.8`) to merge high-frequency Optical Flow velocity with low-frequency commanded PWM. 

## 6. `traffic/traffic_module.py`
### Purpose
The Semantic Brain. It parses YOLO outputs and routes them into specific state machines.
### Key Logic
* **Roundabout Yielding**: Slices the camera frame's left ROI to detect vehicles already circulating. Halts the car until the bounding box clears.
* **Traffic Light State Machine**: Tracks RED vs GREEN states with temporal persistence.

## 7. `traffic/behavior_controller.py`
### Purpose
Overrides standard lane-following control when executing complex maneuvers (Overtaking, Parking).
### Key Logic
* **ParkingSequenceFSM**: A closed-loop IMU state machine. It waits for a parking spot, triggers a 90-degree turn purely tracking IMU delta-yaw, drives into the spot, and parks.

---

# 🧠 SECTION 3: CORE ALGORITHMS (DETAILED)

## Lane Detection (Sobel-X + CLAHE)
Standard thresholding fails in shadows. The pipeline utilizes CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize shadows, followed by a **Sobel-X derivative**. Sobel-X highlights pixels with rapid horizontal color changes (the edges of the lane lines). This makes the system virtually immune to track color inversions.

## Visual Odometry (Lucas-Kanade)
The camera tracks `GoodFeaturesToTrack` (Shi-Tomasi corners) on the asphalt between frames. By calculating the median `dx` and `dy` of these corners, it derives an angular yaw rate and forward velocity. This serves as a "virtual encoder," crucial for detecting wheel slip.

## Sensor Fusion (Complementary Filter)
Because optical flow is noisy at low speeds, and motor PWM is blind to physical resistance (friction/hills), they are fused:
```python
fused_velocity = (0.8 * optical_velocity) + (0.2 * commanded_pwm)
```
This guarantees the controller always bases its lookahead distance on the *actual* physical speed of the chassis.

## Dead Reckoning (VIO Fallback)
When vision drops:
1. **Straight**: `target_x = 320.0 - (delta_yaw_deg * 20.0)`. If the IMU drifts +5 degrees right, the target shifts -100 pixels left. The Stanley Controller violently attacks this error, forcing the car back straight.
2. **Curve**: `target = (0.8 * last_target) + (0.2 * imu_drift)`. The car holds the mechanical steering angle of the curve, softening it slightly if the IMU registers the car straightening out.

## Stanley Steering Controller
Unlike PID, which reacts to error, Stanley geometry predicts it.
```python
cross_track_error = (target_x - 320) * pixel_to_meter_ratio
steer_angle = heading_error + arctan(k * cross_track_error / max(velocity, min_v))
```
This guarantees asymptotic convergence to the lane line without the "sine-wave" oscillation typical of poorly tuned PID loops.

---

# 🧭 SECTION 4: LOCALIZATION SYSTEM

The car does not use a global coordinate map (x, y). It uses **Visual-Inertial Odometry + Semantic Anchoring**.

* **Orientation**: The BNO055 IMU provides absolute gravity-referenced yaw.
* **Motion**: Optical flow provides forward displacement.
* **Correction**: The lane lines provide a continuous zero-drift lateral constraint. As long as the car sees lines, cross-track drift is mathematically impossible.
* **Anchoring**: The graph map (e.g., node 14 is a crosswalk) acts as a checkpoint. When YOLO sees a crosswalk, the internal state "teleports" the car to Node 14. This resets any longitudinal drift accumulated over the lap.

This system asks: *"Am I still aligned with where I should be?"* rather than *"What are my exact global GPS coordinates?"*

---

# 🧠 SECTION 5: STATE MACHINE & BEHAVIOR SYSTEM

The `TrafficDecisionEngine` ensures the car is never guessing what to do. The states are strictly hierarchical:

1. **SYS_STOP**: Hardware failure, collision imminent. Overrides everything.
2. **INTERSECTION / TRAFFIC_LIGHT**: Wait line detected. Observe colors.
3. **ROUNDABOUT**: Check left ROI -> Yield -> Enter -> Arc tracking -> Exit.
4. **PARKING**: 90-degree IMU precision maneuvering.
5. **LANE_FOLLOW**: The default state.

State transitions are triggered by YOLO bounding box distances (`sign_approach_m`) mapped against internal semantic graph edges.

---

# ⚠️ SECTION 6: FAIL-SAFE & ROBUSTNESS DESIGN

The system is designed to survive chaos.

1. **Hysteresis**: Lane confidence drops instantly (safety first) but requires 3 consecutive perfect frames to regain trust. Eliminates 0.9 → 0.1 → 0.8 bouncing.
2. **SAFE MODE**: Triggered natively when `confidence < 0.3`. The car slows down and clamps the steering rate limiter (`MAX_STEER_RATE = 15.0`). The car is physically restricted from panic-swerving.
3. **Pitch Clamping**: Downhill gravity forces are clamped to `-40 PWM`. No matter how steep the ramp, the brakes will not lock up the tires.
4. **Curve Lock Over-commit Protection**: Dead reckoning blends the last known curve with IMU data. It assumes the curve continues, but allows 20% adaptability in case the track straightens mid-dropout.

---

# 📊 SECTION 7: LOGGING & DEBUGGING SYSTEM

A 10Hz lightweight CSV logger runs in the main control loop, capturing:
`timestamp, speed_pwm, steer_deg, lane_conf, anchor, fused_vel, pitch, event`

**Debugging Example**:
If the car spins out entering the roundabout:
* Open `telemetry_log.csv`.
* Filter for `event == 'ROUNDABOUT'`.
* Check `lane_conf`. If it drops to `0.0`, vision failed (glare/shadow).
* Check `anchor`. If it says `IMU_DR`, Dead Reckoning activated.
* Check `steer_deg`. If it spiked to `45.0` instantly, you know `SAFE MODE` failed to clamp the steering rate.

This log transitions the engineering process from "guessing" to "mathematical proof."

---

# ⚙️ SECTION 9: PERFORMANCE & LIMITATIONS

* **CPU Load (Raspberry Pi 5)**: The system relies heavily on the Pi 5's upgraded CPU. YOLOv8 (even ONNX optimized) alongside Lucas-Kanade and Perspective Transforms will consume significant cores.
* **Latency**: The system targets 30Hz (~33ms loop time). If CPU throttling occurs and latency exceeds 100ms, the Stanley Controller will become unstable because the `target_x` it is chasing is in the past.
* **Optical Flow Limitations**: Lucas-Kanade requires textured asphalt. If the track is perfectly smooth and featureless, `optical_vel` will read `0.0`. The Complementary Filter mitigates this by trusting `cmd_speed` when optical flow fails.

---

# 🚀 SECTION 10: FUTURE IMPROVEMENTS

1. **Extended Kalman Filter (EKF)**: Replace the simple Complementary Filter with an EKF to mathematically fuse wheel RPM, IMU acceleration, and Optical Flow based on their real-time variance.
2. **PID Speed Control**: Currently, the system maps desired speed directly to PWM. Implementing a PID controller on the STM32 to target a specific `m/s` (using encoder ticks) will drastically improve low-speed torque.
3. **Dynamic Lookahead**: The Stanley `k` gain is currently fixed. Scaling `k` dynamically based on `fused_velocity` will improve high-speed highway stability.

---

# 🧠 SECTION 11: DESIGN PHILOSOPHY

This codebase proves that you do not need LiDAR or SLAM to drive autonomously.

Absolute localization (GPS) is fragile indoors. Map-building (SLAM) is computationally exhaustive. 

By framing the problem as **Relative Localization**, the system becomes computationally lightweight and incredibly robust. The car treats the lane lines as rails. As long as it is on the rails, it is safe. It uses the IMU to bridge the gaps when the rails disappear, and uses Semantic Signs to reset its progress counter along the track.

This is a defensive, mathematically bounded architecture designed specifically to win endurance-style autonomous challenges.
