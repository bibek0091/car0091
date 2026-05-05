"""
Particle filter for Monte Carlo Localization.
"""
from dataclasses import dataclass
import copy
import numpy as np
from localization.config import (
    N_PARTICLES, SIGMA_V, SIGMA_PSI, SIGMA_LANE,
    SIGMA_APP, WHEELBASE_M, RESAMPLE_THRESH,
    CONVERGENCE_SPREAD_M
)
from localization.graph_utils import curvature_weight, edge_to_xy
from localization.appearance_map import chi2_distance, _nearest_node

@dataclass
class Particle:
    u: str          # source node id (string, as in graphml)
    v: str          # target node id
    s: float        # arc-length progress along edge (metres)
    w: float = 1.0  # unnormalised weight

def init_particles_uniform(G, N=N_PARTICLES):
    edges    = list(G.edges())
    lengths  = np.array([G[u][v]['length'] for u,v in edges])
    probs    = lengths / lengths.sum()
    counts   = np.random.multinomial(N, probs)
    particles = []
    for (u, v), count in zip(edges, counts):
        L = G[u][v]['length']
        for _ in range(count):
            particles.append(Particle(u=u, v=v, s=np.random.uniform(0, L)))
    return particles

def init_particles_warm(G, prior_pose, N=N_PARTICLES):
    """Cluster all particles around saved (x, y, ψ) with Gaussian spread σ=0.3 m, constrained to nearest 5 edges."""
    x_prior = prior_pose['x']
    y_prior = prior_pose['y']
    sigma = 0.3

    # Distance to the midpoint of each edge
    edge_dists = []
    for u, v, data in G.edges(data=True):
        x_u = float(G.nodes[u]['x']); y_u = float(G.nodes[u]['y'])
        x_v = float(G.nodes[v]['x']); y_v = float(G.nodes[v]['y'])
        xm = (x_u + x_v) / 2.0
        ym = (y_u + y_v) / 2.0
        d = np.hypot(xm - x_prior, ym - y_prior)
        edge_dists.append((d, u, v, data['length']))

    edge_dists.sort(key=lambda x: x[0])
    top_edges = edge_dists[:5]

    particles = []
    for _ in range(N):
        _, u, v, L = top_edges[np.random.randint(0, len(top_edges))]
        x_p = x_prior + np.random.normal(0, sigma)
        y_p = y_prior + np.random.normal(0, sigma)

        # Project x_p, y_p onto edge
        x_u = float(G.nodes[u]['x']); y_u = float(G.nodes[u]['y'])
        x_v = float(G.nodes[v]['x']); y_v = float(G.nodes[v]['y'])
        dx = x_v - x_u
        dy = y_v - y_u
        if L > 0:
            t = ((x_p - x_u) * dx + (y_p - y_u) * dy) / (L * L)
            s = np.clip(t * L, 0, L)
        else:
            s = 0.0
        particles.append(Particle(u=u, v=v, s=s))

    return particles

def _random_particle(G):
    edges = list(G.edges())
    u, v = edges[np.random.randint(0, len(edges))]
    L = G[u][v]['length']
    return Particle(u=u, v=v, s=np.random.uniform(0, L))

def predict(particles, v_meas, delta, dt, G, psi_imu):
    v_noisy = v_meas + np.random.normal(0, SIGMA_V)
    v_noisy = max(0.0, v_noisy)

    for p in particles:
        # Advance arc-length
        p.s += v_noisy * dt

        # If particle has reached the end of its edge → transition
        while p.s >= G[p.u][p.v]['length']:
            p.s -= G[p.u][p.v]['length']
            successors = list(G.successors(p.v))
            if not successors:
                # dead end: reset particle to random edge
                reset_p = _random_particle(G)
                p.u = reset_p.u
                p.v = reset_p.v
                p.s = reset_p.s
                break

            kappa_st = np.tan(delta) / WHEELBASE_M

            scores = []
            for w in successors:
                theta_cand = G[p.v][w]['theta']
                kappa_cand = G[p.v][w].get('kappa', 0.0)
                h_score = np.cos(psi_imu - theta_cand)
                k_score = curvature_weight(kappa_cand, kappa_st)
                scores.append(max(0.0, h_score) * k_score)

            scores = np.array(scores)
            total  = scores.sum()
            if total < 1e-9:
                probs = np.ones(len(successors)) / len(successors)
            else:
                probs = scores / total

            next_node = np.random.choice(successors, p=probs)
            p.u = p.v
            p.v = next_node

def update_weights(particles, psi_imu, lateral_error, app_descriptor, G,
                   app_map, delta):
    kappa_st = np.tan(delta) / WHEELBASE_M
    is_turning = abs(delta) > 0.17     # 10 degrees

    for p in particles:
        theta_edge = G[p.u][p.v]['theta']
        kappa_edge = G[p.u][p.v].get('kappa', 0.0)

        # --- Heading weight ---
        dpsi = psi_imu - theta_edge
        dpsi = (dpsi + np.pi) % (2*np.pi) - np.pi   # wrap
        w_heading = np.exp(-0.5 * (dpsi / SIGMA_PSI)**2)

        # --- Lane lateral weight ---
        w_lane = np.exp(-0.5 * (lateral_error / SIGMA_LANE)**2) if lateral_error is not None else 1.0

        # --- Curvature weight (only when turning) ---
        if is_turning:
            w_curve = curvature_weight(kappa_edge, kappa_st)
        else:
            w_curve = 1.0

        # --- Appearance weight ---
        x_p, y_p = edge_to_xy(G, p.u, p.v, p.s)
        nearest_node = _nearest_node(G, x_p, y_p)
        if nearest_node in app_map and app_descriptor is not None:
            stored = app_map[nearest_node]
            chi2   = chi2_distance(app_descriptor, stored)
            w_app  = np.exp(-0.5 * (chi2 / SIGMA_APP)**2)
        else:
            w_app = 1.0

        p.w *= w_heading * w_lane * w_curve * w_app

    # Normalise
    total = sum(p.w for p in particles)
    if total < 1e-300:
        for p in particles:
            p.w = 1.0 / len(particles)
    else:
        for p in particles:
            p.w /= total

def effective_n(particles):
    weights = np.array([p.w for p in particles])
    return 1.0 / ((weights ** 2).sum() + 1e-300)

def resample(particles):
    """Systematic resampling — O(N), low variance."""
    N       = len(particles)
    weights = np.array([p.w for p in particles])
    cumsum  = np.cumsum(weights)
    step    = 1.0 / N
    pos     = np.random.uniform(0, step)
    indices = []
    i = 0
    for _ in range(N):
        while i < N-1 and pos > cumsum[i]:
            i += 1
        indices.append(i)
        pos += step
    new_particles = [copy.deepcopy(particles[i]) for i in indices]
    for p in new_particles:
        p.w = 1.0 / N
    return new_particles

def map_estimate(particles, G):
    """Weighted mean of top-20% particles."""
    particles_sorted = sorted(particles, key=lambda p: p.w, reverse=True)
    top = particles_sorted[:max(1, len(particles)//5)]

    xs = []; ys = []; sins = []; coss = []
    total_w = sum(p.w for p in top)

    for p in top:
        x, y   = edge_to_xy(G, p.u, p.v, p.s)
        theta  = G[p.u][p.v]['theta']
        w_norm = p.w / total_w if total_w > 0 else 1.0 / len(top)
        xs.append(x * w_norm)
        ys.append(y * w_norm)
        sins.append(np.sin(theta) * w_norm)
        coss.append(np.cos(theta) * w_norm)

    x_est     = sum(xs)
    y_est     = sum(ys)
    psi_est   = float(np.arctan2(sum(sins), sum(coss)))
    edge_id   = f"{top[0].u}->{top[0].v}"

    # Convergence confidence: spread of top-10 particles
    top10     = particles_sorted[:10]
    x10 = [edge_to_xy(G, p.u, p.v, p.s)[0] for p in top10]
    y10 = [edge_to_xy(G, p.u, p.v, p.s)[1] for p in top10]
    spread    = float(np.sqrt(np.var(x10) + np.var(y10)))
    confidence = float(spread < CONVERGENCE_SPREAD_M)

    return dict(x=x_est, y=y_est, heading=psi_est,
                edge_id=edge_id, confidence=confidence,
                spread_m=spread)
