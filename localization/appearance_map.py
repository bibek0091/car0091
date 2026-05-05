"""
Appearance map.
"""
import os
import pickle
import numpy as np
import cv2
from skimage.feature import hog

HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS    = 9
HOG_SIZE            = 64  # resize crop to 64×64 before HOG

def extract_descriptor(frame_bgr):
    """Extract HOG descriptor from bottom-centre crop of frame."""
    if len(frame_bgr.shape) == 3:
        h, w = frame_bgr.shape[:2]
        crop = frame_bgr[h//2:, w//4: 3*w//4]   # bottom-centre strip
    else:
        h, w = frame_bgr.shape
        crop = frame_bgr[h//2:, w//4: 3*w//4]
    crop = cv2.resize(crop, (HOG_SIZE, HOG_SIZE))
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    descriptor = hog(gray,
                     orientations=HOG_ORIENTATIONS,
                     pixels_per_cell=HOG_PIXELS_PER_CELL,
                     cells_per_block=HOG_CELLS_PER_BLOCK,
                     feature_vector=True)
    # L2 normalise
    norm = np.linalg.norm(descriptor)
    return descriptor / (norm + 1e-8)

def chi2_distance(a, b):
    """χ² distance between two normalised histograms. Range [0, ∞)."""
    denom = a + b + 1e-10
    return float(np.sum((a - b)**2 / denom))

def _nearest_node(G, x_m, y_m):
    best_node = None
    min_dist = float('inf')
    for node, data in G.nodes(data=True):
        dn = np.hypot(float(data['x']) - x_m, float(data['y']) - y_m)
        if dn < min_dist:
            min_dist = dn
            best_node = node
    return best_node

def build_appearance_map(pose_log, frame_dir, G):
    """
    pose_log: list of dicts {timestamp, x, y, heading, edge_id}
              (output of offline particle filter on Phase 1 recording)
    frame_dir: path to saved JPEG frames from recorder.py
    Returns: dict {node_id: np.ndarray descriptor}
    """
    app_map = {}

    for i, pose in enumerate(pose_log):
        frame_path = os.path.join(frame_dir, f"frame_{i:06d}.jpg")
        if not os.path.exists(frame_path):
            continue
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        desc  = extract_descriptor(frame)

        # Find nearest graph node to this pose
        x, y  = pose['x'], pose['y']
        nearest = _nearest_node(G, x, y)
        if nearest not in app_map:
            app_map[nearest] = desc
        else:
            # Running average (later visits refine the descriptor)
            app_map[nearest] = 0.7 * app_map[nearest] + 0.3 * desc

    return app_map

def save_appearance_map(app_map, path='localization/appearance_map.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(app_map, f)

def load_appearance_map(path='localization/appearance_map.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)
