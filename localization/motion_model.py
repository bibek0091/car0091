"""
Motion model for predicting movement based on velocity and steering.
"""

from localization.config import PWM_CENTER, PWM_RANGE, DELTA_MAX_RAD

def pwm_to_delta(pwm_us: float) -> float:
    """Returns front-wheel steering angle in radians.
    Positive = left turn (counter-clockwise yaw rate).
    """
    normalised = (pwm_us - PWM_CENTER) / PWM_RANGE   # ∈ [-1, 1]
    normalised = max(-1.0, min(1.0, normalised))
    return normalised * DELTA_MAX_RAD
