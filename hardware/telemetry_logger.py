import csv
import time
import os
import atexit

class TelemetryLogger:
    def __init__(self, filename="telemetry_log.csv", buffer_size=50):
        self.filename = filename
        self.buffer_size = buffer_size
        self.buffer = []
        self.last_flush_time = time.time()
        self.last_flush_frame = -1
        self.frame_count = 0
        
        # Initialize file and write headers if it doesn't exist or is empty
        file_exists = os.path.isfile(self.filename)
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists or os.stat(self.filename).st_size == 0:
                writer.writerow([
                    "timestamp", "frame_id", "mode", "event", "lane_conf", "conf_delta",
                    "target_x", "curvature", "yaw", "pitch", "fused_vel", "steer", 
                    "speed", "steer_delta", "speed_delta", "steer_saturated", 
                    "safe_mode", "anchor", "fps", "loop_time_ms", "latency_flag", 
                    "root_cause"
                ])
        
        # Register flush on exit for data safety
        atexit.register(self.flush)

    def log(self, data, force_flush=False):
        """
        Appends data to the memory buffer.
        Flushes to disk if buffer is full, 1s has passed, or force_flush is True.
        """
        # data is expected to be a list. We prepend the frame_count.
        self.buffer.append([self.frame_count] + data)
        self.frame_count += 1
        
        time_since_flush = time.time() - self.last_flush_time
        
        # Duplicate flush protection: Don't flush more than once per frame
        if self.frame_count == self.last_flush_frame:
            return

        if len(self.buffer) >= self.buffer_size or time_since_flush > 1.0 or force_flush:
            self.flush()
            
    def flush(self):
        """
        Writes the memory buffer to the CSV file to prevent data loss.
        """
        if not self.buffer:
            return
        
        try:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.buffer)
            self.buffer.clear()
            self.last_flush_time = time.time()
            self.last_flush_frame = self.frame_count
        except Exception:
            # Silent fail to avoid disrupting main loop if file is locked
            pass

    def close(self):
        """
        Force flush the buffer before closing the application.
        """
        self.flush()
