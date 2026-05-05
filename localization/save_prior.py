"""
Save initial last_pose.json (park position).
Usage: python -m localization.save_prior --pose_log recordings/run_001/pose_log.csv
"""
import argparse
import pandas as pd
import json

def save_prior(pose_log_path, output_path='localization/last_pose.json'):
    df = pd.read_csv(pose_log_path)
    if len(df) == 0:
        print("Empty pose log.")
        return
    last = df.iloc[-1]
    
    pose = {
        'x': float(last['x']),
        'y': float(last['y']),
        'heading': float(last['heading']),
        'edge_id': str(last['edge_id']),
        'confidence': 1.0,
        'spread_m': 0.1
    }
    
    with open(output_path, 'w') as f:
        json.dump(pose, f, indent=2)
    print(f"Saved prior to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_log', type=str, required=True)
    args = parser.parse_args()
    save_prior(args.pose_log)
