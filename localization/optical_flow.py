"""
Optical flow based velocity estimation.
"""
import cv2
import numpy as np
from localization.config import FLOW_ROI_FRAC, CAMERA_HEIGHT_M, FOCAL_LENGTH_PX

lk_params = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

def get_roi(bird_eye_frame):
    h = bird_eye_frame.shape[0]
    top = int(h * (1.0 - FLOW_ROI_FRAC))
    return bird_eye_frame[top:, :]

def estimate_velocity(prev_roi, curr_roi, dt: float) -> float:
    """Returns forward velocity in m/s."""
    if dt <= 0:
        return 0.0

    prev_pts = cv2.goodFeaturesToTrack(
        prev_roi, maxCorners=80, qualityLevel=0.01, minDistance=5
    )
    if prev_pts is None or len(prev_pts) < 4:
        return 0.0

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_roi, curr_roi, prev_pts, None, **lk_params
    )

    good_prev = prev_pts[status == 1]
    good_curr = curr_pts[status == 1]

    if len(good_prev) == 0:
        return 0.0

    # Flow vectors in pixels
    flow = good_curr - good_prev  # shape (N, 2)

    # Keep only points with predominantly vertical (forward) flow
    # In bird's-eye view, forward motion → points move toward top of ROI (negative dy)
    vert_mask = np.abs(flow[:, 1]) > np.abs(flow[:, 0]) * 1.5
    if vert_mask.sum() < 3:
        return 0.0

    vert_flow = np.abs(flow[vert_mask, 1])  # pixels/frame
    median_flow_px = float(np.median(vert_flow))  # pixels/frame

    # Scale: pixels/frame → m/s
    v = median_flow_px * (CAMERA_HEIGHT_M / FOCAL_LENGTH_PX) * (1.0 / dt)
    return float(np.clip(v, 0.0, 3.0))  # cap at 3 m/s
