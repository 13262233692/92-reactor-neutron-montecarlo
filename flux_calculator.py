import numpy as np
from geometry import (
    CORE_X_MIN, CORE_X_MAX, CORE_Y_MIN, CORE_Y_MAX,
    CORE_Z_MIN, CORE_Z_MAX
)
from neutron_transport import REACTION_ABSORB, REACTION_FISSION, REACTION_LEAK


def compute_flux_map(results, nx=80, ny=80, nz=40):
    fx = results['final_x']
    fy = results['final_y']
    fz = results['final_z']
    rx = results['reaction']

    mask = (rx == REACTION_ABSORB) | (rx == REACTION_FISSION)
    x = fx[mask]
    y = fy[mask]
    z = fz[mask]

    x_edges = np.linspace(CORE_X_MIN, CORE_X_MAX, nx + 1)
    y_edges = np.linspace(CORE_Y_MIN, CORE_Y_MAX, ny + 1)
    z_edges = np.linspace(CORE_Z_MIN, CORE_Z_MAX, nz + 1)

    flux, _ = np.histogramdd(np.column_stack([x, y, z]),
                             bins=[x_edges, y_edges, z_edges])

    flux = flux.astype(np.float64)
    max_val = flux.max()
    if max_val > 0:
        flux /= max_val

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    return flux, x_centers, y_centers, z_centers


def compute_radial_flux(results, n_radial=50, n_axial=30):
    fx = results['final_x']
    fy = results['final_y']
    fz = results['final_z']
    rx = results['reaction']

    mask = (rx == REACTION_ABSORB) | (rx == REACTION_FISSION)
    x = fx[mask]
    y = fy[mask]
    z = fz[mask]

    r = np.sqrt(x ** 2 + y ** 2)
    core_center_r = 0.0
    r_max = np.sqrt(((CORE_X_MAX - CORE_X_MIN) / 2) ** 2 +
                    ((CORE_Y_MAX - CORE_Y_MIN) / 2) ** 2)

    r_edges = np.linspace(0, r_max, n_radial + 1)
    z_edges = np.linspace(CORE_Z_MIN, CORE_Z_MAX, n_axial + 1)

    flux_rz, _, _ = np.histogram2d(r, z, bins=[r_edges, z_edges])
    flux_rz = flux_rz.astype(np.float64)
    max_val = flux_rz.max()
    if max_val > 0:
        flux_rz /= max_val

    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    return flux_rz, r_centers, z_centers


def select_interesting_traces(traces_x, traces_y, traces_z,
                               reactions, n_select=3000):
    n_traces = len(traces_x)
    if n_traces <= n_select:
        return list(range(n_traces))

    lengths = np.array([len(t) for t in traces_x], dtype=np.float64)
    complexity = lengths.copy()
    for i in range(n_traces):
        if len(traces_x[i]) >= 3:
            dx = np.diff(traces_x[i])
            dy = np.diff(traces_y[i])
            dz = np.diff(traces_z[i])
            dirs = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            if len(dirs) > 1:
                dd = np.diff(dirs)
                complexity[i] += np.sum(np.abs(dd)) * 2

    complexity += np.random.random(len(complexity)) * 0.1
    indices = np.argsort(-complexity)[:n_select]

    return sorted(indices.tolist())
