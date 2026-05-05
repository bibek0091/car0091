"""
Physical constants and configurations for the localization system.
"""

WHEELBASE_M      = 0.257        # L — axle-to-axle distance in metres (1:10 scale)
CAMERA_HEIGHT_M  = 0.082        # h_cam — lens centre above ground plane
FOCAL_LENGTH_PX  = 572.0        # f — from calibration (640-px-wide frame)
FRAME_W, FRAME_H = 640, 480
FLOW_ROI_FRAC    = 0.35         # use bottom 35% of frame for ground flow
IMU_HZ           = 50           # BNO055 polling rate
CAMERA_FPS       = 15
LOCALIZER_HZ     = 10           # particle filter update rate
N_PARTICLES      = 500
RESAMPLE_THRESH  = 0.5          # resample when N_eff / N < this
MAP_W_M, MAP_H_M = 20.5, 14.0
SVG_W_PX, SVG_H_PX = 800, 540

# Steering servo calibration (read from assets/car_actions.csv)
PWM_CENTER       = 1500         # μs — straight ahead
PWM_RANGE        = 500          # μs — full deflection each side
DELTA_MAX_RAD    = 0.524        # 30° in radians — max front-wheel angle

# Noise parameters (tune on real hardware)
SIGMA_PSI        = 0.15         # rad — heading observation noise
SIGMA_LANE       = 0.05         # m   — lateral error observation noise
SIGMA_APP        = 0.30         # χ² appearance distance noise
SIGMA_V          = 0.05         # m/s — velocity process noise (additive)
SIGMA_DELTA      = 0.02         # rad — steering angle process noise

# Convergence threshold
CONVERGENCE_SPREAD_M = 0.30     # top-10 particles must be within this radius
CONVERGENCE_MIN_CONF = 0.70     # minimum normalised weight sum of top-10
