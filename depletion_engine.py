import numpy as np
import time
from numba import njit, prange
from geometry import (
    CORE_X_MIN, CORE_X_MAX, CORE_Y_MIN, CORE_Y_MAX,
    CORE_Z_MIN, CORE_Z_MAX, get_material, MAT_FUEL, MAT_MODERATOR, MAT_COOLANT
)
from depletion import (
    N_NUCLIDES, INITIAL_DENSITY, NUCLIDE_NAMES,
    i_U235, i_U238, i_Pu239, i_Xe135, i_Sm149, i_I135, i_Np239, i_Cs137,
    SIGMA_FISSION, SIGMA_CAPTURE, LAMBDA, BARN, SEC_PER_DAY,
    build_depletion_matrix, solve_bateman_step
)


def build_3d_flux_from_results(results, nx=30, ny=30, nz=20):
    from neutron_transport import REACTION_ABSORB, REACTION_FISSION
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

    flux_counts, _ = np.histogramdd(
        np.column_stack([x, y, z]),
        bins=[x_edges, y_edges, z_edges]
    )

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    dv = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]) * (z_edges[1] - z_edges[0])
    n_total = len(x)
    total_source = results.get('n_fission', 0) * 2.43 + results.get('n_absorb', 0)
    source_per_neutron = 1.0 / n_total if n_total > 0 else 0.0

    peak = flux_counts.max()
    flux_thermal_cm2s = np.zeros_like(flux_counts, dtype=np.float64)
    if peak > 0:
        normalized = flux_counts / peak
        flux_thermal_cm2s = normalized * 2.5e14

    fuel_mask = np.zeros((nx, ny, nz), dtype=bool)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                m = get_material(x_centers[ix], y_centers[iy], z_centers[iz])
                if m == MAT_FUEL:
                    fuel_mask[ix, iy, iz] = True

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if not fuel_mask[ix, iy, iz]:
                    flux_thermal_cm2s[ix, iy, iz] = 0.0

    return flux_thermal_cm2s, x_centers, y_centers, z_centers, fuel_mask


@njit(cache=True)
def _deplete_voxel_kernel(flux_3d, fuel_mask, nx, ny, nz,
                          n_time_steps, dt_seconds,
                          density_grid_out,
                          reaction_rate_grid_out):
    n = N_NUCLIDES
    total = 0
    for ix in prange(nx):
        for iy in range(ny):
            for iz in range(nz):
                if not fuel_mask[ix, iy, iz]:
                    for k in range(n):
                        density_grid_out[ix, iy, iz, k] = 0.0
                    for k in range(4):
                        reaction_rate_grid_out[ix, iy, iz, k] = 0.0
                    continue

                flux = flux_3d[ix, iy, iz]
                if flux <= 1e-10:
                    for k in range(n):
                        density_grid_out[ix, iy, iz, k] = INITIAL_DENSITY[k]
                    for k in range(4):
                        reaction_rate_grid_out[ix, iy, iz, k] = 0.0
                    continue

                dens = np.zeros(n, dtype=np.float64)
                for k in range(n):
                    dens[k] = INITIAL_DENSITY[k]

                rx_rate_total = np.zeros(4, dtype=np.float64)
                sigma_f_U235 = SIGMA_FISSION[i_U235] * BARN
                sigma_f_Pu239 = SIGMA_FISSION[i_Pu239] * BARN
                sigma_a_Xe135 = SIGMA_CAPTURE[i_Xe135] * BARN
                sigma_a_Sm149 = SIGMA_CAPTURE[i_Sm149] * BARN

                for t in range(n_time_steps):
                    dens = solve_bateman_step(flux, dens, dt_seconds)

                    fission_rate = flux * (
                        dens[i_U235] * sigma_f_U235 +
                        dens[i_Pu239] * sigma_f_Pu239
                    )
                    capture_Xe135 = flux * dens[i_Xe135] * sigma_a_Xe135
                    capture_Sm149 = flux * dens[i_Sm149] * sigma_a_Sm149
                    rx_rate_total[0] += fission_rate
                    rx_rate_total[1] += capture_Xe135
                    rx_rate_total[2] += capture_Sm149

                for k in range(n):
                    density_grid_out[ix, iy, iz, k] = dens[k]

                reaction_rate_grid_out[ix, iy, iz, 0] = rx_rate_total[0] / max(1, n_time_steps)
                reaction_rate_grid_out[ix, iy, iz, 1] = rx_rate_total[1] / max(1, n_time_steps)
                reaction_rate_grid_out[ix, iy, iz, 2] = rx_rate_total[2] / max(1, n_time_steps)
                reaction_rate_grid_out[ix, iy, iz, 3] = dens[i_Xe135]


def run_depletion_simulation(results, flux_grid_info=None,
                             burnup_days=90, time_steps_per_day=4,
                             nx=30, ny=30, nz=20):
    print("[Depletion] 构建三维通量场...")
    if flux_grid_info is None:
        flux_3d, xc, yc, zc, fuel_mask = build_3d_flux_from_results(
            results, nx=nx, ny=ny, nz=nz
        )
    else:
        flux_3d, xc, yc, zc = flux_grid_info
        fuel_mask = np.zeros_like(flux_3d, dtype=bool)
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    m = get_material(xc[ix], yc[iy], zc[iz])
                    if m == MAT_FUEL:
                        fuel_mask[ix, iy, iz] = True

    n_steps = int(burnup_days * time_steps_per_day)
    dt_seconds = SEC_PER_DAY / time_steps_per_day

    print(f"[Depletion] 燃耗推演: {burnup_days} 天, {n_steps} 时间步, dt = {dt_seconds:.1f}s")
    print(f"[Depletion] 网格: {nx}×{ny}×{nz} = {nx*ny*nz} 体素, 燃料体素: {int(fuel_mask.sum())}")

    density_grid = np.zeros((nx, ny, nz, N_NUCLIDES), dtype=np.float64)
    reaction_rates = np.zeros((nx, ny, nz, 4), dtype=np.float64)

    t0 = time.time()
    _deplete_voxel_kernel(
        flux_3d, fuel_mask, nx, ny, nz,
        n_steps, dt_seconds,
        density_grid, reaction_rates
    )

    elapsed = time.time() - t0
    n_fuel = int(fuel_mask.sum())
    solves = n_fuel * n_steps
    print(f"[Depletion] 完成: {elapsed:.1f}s, {solves:,} Bateman 求解, "
          f"{solves/max(elapsed,1e-9):,.0f}/s")

    xe135_grid = density_grid[:, :, :, i_Xe135]
    sm149_grid = density_grid[:, :, :, i_Sm149]
    pu239_grid = density_grid[:, :, :, i_Pu239]
    u235_grid = density_grid[:, :, :, i_U235]

    xe135_max = xe135_grid.max() if xe135_grid.size > 0 else 0
    u235_initial = INITIAL_DENSITY[i_U235]
    burnup_pct = np.where(
        u235_grid > 0,
        100.0 * (u235_initial - u235_grid) / u235_initial,
        0.0
    )

    print(f"[Depletion] 核素浓度峰值:")
    print(f"    U-235  初始: {u235_initial:.4e}  残留: {u235_grid.max():.4e} at/cm³")
    print(f"    Pu-239 峰值: {pu239_grid.max():.4e} at/cm³")
    print(f"    Xe-135 峰值: {xe135_max:.4e} at/cm³")
    print(f"    Sm-149 峰值: {sm149_grid.max():.4e} at/cm³")
    print(f"    最大燃耗: {np.nanmax(burnup_pct):.2f}%")

    return {
        'density_grid': density_grid,
        'reaction_rates': reaction_rates,
        'xe135_grid': xe135_grid,
        'sm149_grid': sm149_grid,
        'pu239_grid': pu239_grid,
        'u235_grid': u235_grid,
        'burnup_pct': burnup_pct,
        'flux_3d': flux_3d,
        'x_centers': xc, 'y_centers': yc, 'z_centers': zc,
        'fuel_mask': fuel_mask,
        'burnup_days': burnup_days,
        'nuclide_names': NUCLIDE_NAMES,
        'i_U235': i_U235, 'i_U238': i_U238, 'i_Pu239': i_Pu239,
        'i_I135': i_I135, 'i_Xe135': i_Xe135,
        'i_Cs137': i_Cs137, 'i_Sm149': i_Sm149, 'i_Np239': i_Np239
    }
