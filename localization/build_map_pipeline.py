import pandas as pd
import pathlib
import json
import csv
import sys
from localization.offline_localizer import run_offline
from localization.appearance_map import build_appearance_map, save_appearance_map
from localization.graph_utils import load_graph
from localization.save_prior import save_prior

def run_pipeline():
    recording_dir = "recordings/gui_run"
    output_path = "recordings/gui_run/pose_log.csv"
    
    print("[MAP BUILDER] Starting Unified Offline Particle Filter...", flush=True)
    try:
        # 1. Run offline localizer
        run_offline(recording_dir, output_path)
        
        # 2. Build Appearance Map
        print("[MAP BUILDER] Generating graphical appearance map from recorded frames...", flush=True)
        pose_log = []
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pose_log.append({
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'heading': float(row['heading']),
                    'edge_id': str(row['edge_id'])
                })
        G = load_graph()
        app_map = build_appearance_map(pose_log, str(pathlib.Path(recording_dir) / 'frames'), G)
        save_appearance_map(app_map, 'localization/appearance_map.pkl')
        print("[MAP BUILDER] appearance_map.pkl saved successfully.", flush=True)
        
        # 3. Save Prior
        print("[MAP BUILDER] Saving starting position prior...", flush=True)
        save_prior(output_path, 'localization/last_pose.json')
        
        print("[MAP BUILDER] ALL TASKS COMPLETE! Map is ready for live use.", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"[MAP BUILDER FATAL] {e}", file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == '__main__':
    run_pipeline()
