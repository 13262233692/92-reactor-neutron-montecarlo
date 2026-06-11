import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from geometry import (
    CORE_X_MIN, CORE_X_MAX, CORE_Y_MIN, CORE_Y_MAX,
    CORE_Z_MIN, CORE_Z_MAX, CORE_RADIUS, CORE_HEIGHT,
    PITCH, FUEL_ROD_RADIUS, ASSEMBLY_SIZE,
    N_ASSEMBLIES_X, N_ASSEMBLIES_Y,
    get_fuel_rod_positions_full
)
from neutron_transport import REACTION_ABSORB, REACTION_FISSION, REACTION_LEAK
from flux_calculator import select_interesting_traces


def create_fuel_rod_mesh(n_sample=3):
    positions = get_fuel_rod_positions_full()
    center_x = (CORE_X_MIN + CORE_X_MAX) / 2.0
    center_y = (CORE_Y_MIN + CORE_Y_MAX) / 2.0

    if n_sample > 0 and len(positions) > 200:
        step = max(1, len(positions) // 200)
        positions = positions[::step]

    theta = np.linspace(0, 2 * np.pi, 9)
    z_pts = np.array([CORE_Z_MIN, CORE_Z_MAX])

    all_x = []
    all_y = []
    all_z = []
    all_i = []
    all_j = []
    all_k = []
    vertex_offset = 0

    for px, py in positions:
        for z_val in z_pts:
            for t in theta:
                all_x.append(px + FUEL_ROD_RADIUS * np.cos(t))
                all_y.append(py + FUEL_ROD_RADIUS * np.sin(t))
                all_z.append(z_val)

        n_theta = len(theta)
        for i_t in range(n_theta - 1):
            i0 = vertex_offset + i_t
            i1 = vertex_offset + i_t + 1
            i2 = vertex_offset + n_theta + i_t
            i3 = vertex_offset + n_theta + i_t + 1
            all_i.extend([i0, i0])
            all_j.extend([i1, i2])
            all_k.extend([i2, i3])

        vertex_offset += 2 * n_theta

    return np.array(all_x), np.array(all_y), np.array(all_z), \
           np.array(all_i), np.array(all_j), np.array(all_k)


def build_flux_volume_trace(flux, x_centers, y_centers, z_centers):
    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

    n_total = flux.size
    n_nonzero = np.count_nonzero(flux)
    sparsity = n_nonzero / n_total if n_total > 0 else 1.0

    opacity_factor = min(0.3, 0.05 / max(sparsity, 0.001))

    threshold = np.percentile(flux[flux > 0], 20) if n_nonzero > 0 else 0
    display_flux = np.where(flux > threshold, flux, 0)

    surface_count = min(30, max(5, int(opacity_factor * 200)))

    return go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=display_flux.flatten(),
        isomin=0.05,
        isomax=1.0,
        opacity=opacity_factor,
        surface_count=surface_count,
        colorscale='Inferno',
        colorbar=dict(title='相对通量', x=1.02, len=0.8),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='中子通量密度'
    )


def build_neutron_traces(traces_x, traces_y, traces_z, trace_reactions, n_select=3000):
    indices = select_interesting_traces(traces_x, traces_y, traces_z,
                                        trace_reactions, n_select)

    scatter_traces = []
    n_draw = min(2000, len(indices))

    for idx in indices[:n_draw]:
        tx = traces_x[idx]
        ty = traces_y[idx]
        tz = traces_z[idx]

        rx = trace_reactions[idx] if idx < len(trace_reactions) else REACTION_ABSORB

        if rx == REACTION_FISSION:
            color = 'rgba(255, 50, 50, 0.15)'
        elif rx == REACTION_ABSORB:
            color = 'rgba(50, 150, 255, 0.12)'
        else:
            color = 'rgba(150, 150, 150, 0.08)'

        scatter_traces.append(go.Scatter3d(
            x=tx, y=ty, z=tz,
            mode='lines',
            line=dict(width=0.8, color=color),
            showlegend=False,
            hoverinfo='skip'
        ))

    return scatter_traces


def build_core_outline():
    theta = np.linspace(0, 2 * np.pi, 50)
    r = CORE_RADIUS
    cx = (CORE_X_MIN + CORE_X_MAX) / 2.0
    cy = (CORE_Y_MIN + CORE_Y_MAX) / 2.0

    traces = []

    x_ring = cx + r * np.cos(theta)
    y_ring = cy + r * np.sin(theta)
    z_bot = np.full_like(theta, CORE_Z_MIN)
    z_top = np.full_like(theta, CORE_Z_MAX)

    traces.append(go.Scatter3d(
        x=x_ring, y=y_ring, z=z_bot,
        mode='lines', line=dict(width=2, color='gray'),
        showlegend=False, hoverinfo='skip'
    ))
    traces.append(go.Scatter3d(
        x=x_ring, y=y_ring, z=z_top,
        mode='lines', line=dict(width=2, color='gray'),
        showlegend=False, hoverinfo='skip'
    ))

    for t in np.linspace(0, 2 * np.pi, 9)[:-1]:
        x_line = [cx + r * np.cos(t), cx + r * np.cos(t)]
        y_line = [cy + r * np.sin(t), cy + r * np.sin(t)]
        z_line = [CORE_Z_MIN, CORE_Z_MAX]
        traces.append(go.Scatter3d(
            x=x_line, y=y_line, z=z_line,
            mode='lines', line=dict(width=1, color='rgba(128,128,128,0.3)'),
            showlegend=False, hoverinfo='skip'
        ))

    return traces


def build_fuel_rod_scatter():
    positions = get_fuel_rod_positions_full()
    cx = (CORE_X_MIN + CORE_X_MAX) / 2.0
    cy = (CORE_Y_MIN + CORE_Y_MAX) / 2.0

    filtered = [(px, py) for px, py in positions
                if np.sqrt((px - cx)**2 + (py - cy)**2) < CORE_RADIUS]

    step = max(1, len(filtered) // 500)
    sampled = filtered[::step]

    x = [p[0] for p in sampled]
    y = [p[1] for p in sampled]
    z_mid = [(CORE_Z_MIN + CORE_Z_MAX) / 2.0] * len(sampled)

    return go.Scatter3d(
        x=x, y=y, z=z_mid,
        mode='markers',
        marker=dict(size=2, color='green', opacity=0.3),
        name='燃料棒截面',
        hoverinfo='skip'
    )


def build_reaction_scatter(results, n_sample=10000):
    fx = results['final_x']
    fy = results['final_y']
    fz = results['final_z']
    rx = results['reaction']

    mask_absorb = rx == REACTION_ABSORB
    mask_fission = rx == REACTION_FISSION

    n_a = min(n_sample, mask_absorb.sum())
    n_f = min(n_sample, mask_fission.sum())

    idx_a = np.where(mask_absorb)[0]
    idx_f = np.where(mask_fission)[0]

    if len(idx_a) > n_a:
        idx_a = np.random.choice(idx_a, n_a, replace=False)
    if len(idx_f) > n_f:
        idx_f = np.random.choice(idx_f, n_f, replace=False)

    traces = []

    if len(idx_f) > 0:
        traces.append(go.Scatter3d(
            x=fx[idx_f], y=fy[idx_f], z=fz[idx_f],
            mode='markers',
            marker=dict(size=2, color='red', opacity=0.4,
                       symbol='circle'),
            name='裂变点',
        ))

    if len(idx_a) > 0:
        traces.append(go.Scatter3d(
            x=fx[idx_a], y=fy[idx_a], z=fz[idx_a],
            mode='markers',
            marker=dict(size=1.5, color='blue', opacity=0.2,
                       symbol='circle'),
            name='吸收点',
        ))

    return traces


def create_dashboard(results, traced_results=None, flux_data=None):
    if flux_data is None:
        from flux_calculator import compute_flux_map
        flux, xc, yc, zc = compute_flux_map(results)
        flux_data = (flux, xc, yc, zc)

    flux, xc, yc, zc = flux_data

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[dict(type='scene')]],
    )

    fig.add_trace(build_flux_volume_trace(flux, xc, yc, zc), row=1, col=1)

    for trace in build_core_outline():
        fig.add_trace(trace, row=1, col=1)

    fig.add_trace(build_fuel_rod_scatter(), row=1, col=1)

    for trace in build_reaction_scatter(results):
        fig.add_trace(trace, row=1, col=1)

    if traced_results is not None:
        neutron_traces = build_neutron_traces(
            traced_results['traces_x'],
            traced_results['traces_y'],
            traced_results['traces_z'],
            traced_results['trace_reactions'],
            n_select=3000
        )
        for trace in neutron_traces:
            fig.add_trace(trace, row=1, col=1)

    n_total = len(results['final_x'])
    n_absorb = results.get('n_absorb', np.sum(results['reaction'] == REACTION_ABSORB))
    n_fission = results.get('n_fission', np.sum(results['reaction'] == REACTION_FISSION))
    n_leak = results.get('n_leak', np.sum(results['reaction'] == REACTION_LEAK))
    k_eff = (n_fission * 2.43) / n_total if n_total > 0 else 0

    fig.update_layout(
        title=dict(
            text=f'堆芯三维中子通量实时监控台 | '
                 f'中子数: {n_total:,} | '
                 f'k-eff ≈ {k_eff:.4f} | '
                 f'裂变: {n_fission:,} | '
                 f'吸收: {n_absorb:,} | '
                 f'泄漏: {n_leak:,}',
            font=dict(size=14)
        ),
        scene=dict(
            xaxis=dict(title='X (cm)', range=[CORE_X_MIN, CORE_X_MAX],
                      backgroundcolor='rgb(20,20,30)'),
            yaxis=dict(title='Y (cm)', range=[CORE_Y_MIN, CORE_Y_MAX],
                      backgroundcolor='rgb(20,20,30)'),
            zaxis=dict(title='Z (cm)', range=[CORE_Z_MIN, CORE_Z_MAX],
                      backgroundcolor='rgb(20,20,30)'),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1.5),
            bgcolor='rgb(10,10,20)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        paper_bgcolor='rgb(15,15,25)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=60, b=0),
        height=900,
        showlegend=True,
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(size=10, color='white')
        )
    )

    return fig
