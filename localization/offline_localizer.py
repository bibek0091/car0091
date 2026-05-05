"""
Offline Particle Filter for Phase 1.
Usage: python -m localization.offline_localizer --recording recordings/run_001 --output recordings/run_001/pose_log.csv
"""
import argparse
import pandas as pd
import pathlib
import json
import csv
from localization.graph_utils import load_graph, compute_edge_curvatures
from localization.particle_filter import (
    init_particles_uniform, predict, update_weights, effective_n, resample, map_estimate
)
from localization.config import N_PARTICLES, RESAMPLE_THRESH

def run_offline(recording_dir, output_path):
    d = pathlib.Path(recording_dir)
    imu_df = pd.read_csv(d / 'imu.csv')
    rc_df  = pd.read_csv(d / 'rc.csv')

    G = load_graph()
    compute_edge_curvatures(G)
    particles = init_particles_uniform(G, N_PARTICLES)

    out_rows = []
    
    # Merge and sort
    imu_df['type'] = 'imu'
    rc_df['type'] = 'rc'
    merged = pd.concat([imu_df, rc_df]).sort_values('t')
    
    prev_t = merged['t'].iloc[0]
    last_rc = {'steering_deg': 0.0, 'speed_mms': 0.0}

    print("Running offline particle filter...")

    for _, row in merged.iterrows():
        dt = float(row['t'] - prev_t)
        prev_t = row['t']
        
        if row['type'] == 'rc':
            last_rc['steering_deg'] = row['steering_deg']
            last_rc['speed_mms'] = row['speed_mms']
            continue
            
        if row['type'] == 'imu':
            psi_imu = float(row['yaw'])
            
            import math
            delta = math.radians(last_rc['steering_deg'])
            speed_mms = last_rc['speed_mms']
            
            v_est = 0.3 if speed_mms > 200 else 0.0

            predict(particles, v_est, delta, dt, G, psi_imu)
            update_weights(particles, psi_imu, None, None, G, {}, delta)
            
            if effective_n(particles) / N_PARTICLES < RESAMPLE_THRESH:
                particles = resample(particles)
                
            pose = map_estimate(particles, G)
            out_rows.append([row['t'], pose['x'], pose['y'], pose['heading'], pose['edge_id']])

    with open(output_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp', 'x', 'y', 'heading', 'edge_id'])
        w.writerows(out_rows)
    print(f"Saved {len(out_rows)} poses to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recording', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    run_offline(args.recording, args.output)
