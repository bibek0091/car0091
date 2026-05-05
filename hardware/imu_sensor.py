import threading
import time
import math

try:
    from hardware.serial_handler import SHARED_STATE
except ImportError:
    # Fallback if imported without context
    SHARED_STATE = {
        "imu_yaw": 0.0, 
        "imu_calibrated": True,
        "cmd_speed": 0.0, 
        "cmd_steer": 0.0,
        "last_imu_time": 0.0
    }

try:
    from config import WHEELBASE_M, ENABLE_IMU_FUSION, ENABLE_OPTICAL_FLOW
except ImportError:
    WHEELBASE_M = 0.23
    ENABLE_IMU_FUSION = False
    ENABLE_OPTICAL_FLOW = False

class IMUSensor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = False
        
        # Enforce True immediately to prevent the autonomous logic from blocking
        self.is_calibrated = True 
        
        self.simulated_yaw = 0.0
        self.last_update_time = time.time()
        self.has_hardware = False

    def run(self):
        self.running = True
        print("[IMU/Odometry] Adaptive Tracker Started")
        
        while self.running:
            now = time.time()
            dt = now - self.last_update_time
            self.last_update_time = now
            
            last_hw_time = SHARED_STATE.get("last_imu_time", 0.0)
            
            # Rely EXCLUSIVELY on proper hardware IMU data from STM32 telemetry
            self.simulated_yaw = SHARED_STATE.get("imu_yaw", 0.0)
            self.pitch = SHARED_STATE.get("imu_pitch", 0.0)
            
            # Check connection timeout (1.5s)
            if now - last_hw_time > 1.5:
                self.is_calibrated = False
                self.has_hardware = False
                if hasattr(self, '_hw_logged'):
                    delattr(self, '_hw_logged') # Reset log flag
            else:
                self.is_calibrated = SHARED_STATE.get("imu_calibrated", True)
                self.has_hardware = True
                if not hasattr(self, '_hw_logged'):
                    print("[IMU] Hardware IMU data connection established")
                    self._hw_logged = True
                
            time.sleep(0.02) # 50Hz update loop

    def stop(self):
        self.running = False

    def update_optical_velocity(self, opt_vel: float):
        if ENABLE_OPTICAL_FLOW:
            self.optical_velocity = opt_vel

    def get_fused_velocity(self):
        if ENABLE_IMU_FUSION:
            # Simple Complementary Filter: Trust optical flow for high frequency,
            # but bound it to the commanded speed (low frequency expected)
            cmd_speed = SHARED_STATE.get("cmd_speed", 0.0)
            alpha = 0.8
            self.fused_velocity = alpha * getattr(self, 'optical_velocity', 0.0) + (1 - alpha) * cmd_speed
            return self.fused_velocity
        return SHARED_STATE.get("cmd_speed", 0.0)

    def get_yaw(self):
        return self.simulated_yaw

    def get_pitch(self):
        return getattr(self, 'pitch', 0.0)

    def get_has_hardware(self):
        return self.has_hardware
