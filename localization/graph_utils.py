"""
Graph utilities.
"""
import networkx as nx
import numpy as np

def load_graph(path='assets/Competition_track_graph.graphml'):
    G = nx.read_graphml(path)
    # Precompute for each edge: length, direction angle, curvature
    for u, v, data in G.edges(data=True):
        x_u = float(G.nodes[u]['x'])
        y_u = float(G.nodes[u]['y'])
        x_v = float(G.nodes[v]['x'])
        y_v = float(G.nodes[v]['y'])
        dx = x_v - x_u
        dy = y_v - y_u
        data['length'] = float(np.hypot(dx, dy))
        data['theta']  = float(np.arctan2(dy, dx))  # heading of edge in radians
        data['dotted'] = data.get('dotted', 'False') in ('True', True)
    return G

def compute_edge_curvatures(G):
    """Adds 'kappa' attribute to each edge = signed curvature in 1/m."""
    for v in G.nodes():
        in_edges  = list(G.in_edges(v, data=True))
        out_edges = list(G.out_edges(v, data=True))
        if not in_edges or not out_edges:
            continue
        theta_in = in_edges[0][2]['theta']
        for _, w, data in out_edges:
            theta_out = data['theta']
            dtheta = theta_out - theta_in
            # Wrap to [-π, π]
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            L_avg = (in_edges[0][2]['length'] + data['length']) / 2.0
            data['kappa'] = dtheta / L_avg if L_avg > 0 else 0.0

def edge_to_xy(G, u, v, s):
    """
    Returns (x_m, y_m) for a particle at arc-length s along edge u→v.
    s is clamped to [0, edge_length].
    """
    x_u = float(G.nodes[u]['x'])
    y_u = float(G.nodes[u]['y'])
    x_v = float(G.nodes[v]['x'])
    y_v = float(G.nodes[v]['y'])
    L   = G[u][v]['length']
    t   = np.clip(s / L, 0.0, 1.0) if L > 0 else 0.0
    return x_u + t*(x_v - x_u), y_u + t*(y_v - y_u)

def curvature_weight(kappa_edge, kappa_steering, sigma=0.3):
    """Gaussian weight: high when edge curvature matches steering curvature."""
    diff = kappa_edge - kappa_steering
    return float(np.exp(-0.5 * (diff / sigma) ** 2))
