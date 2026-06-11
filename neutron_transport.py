import numpy as np
from numba import njit, prange
from nuclear_data import (
    N_ENERGY_GROUPS, NU_BAR,
    MAT_FUEL, MAT_MODERATOR, MAT_COOLANT,
    get_energy_group, get_macro_xs, get_A,
    sample_fission_energy,
    ENERGY_GROUPS
)
from geometry import (
    CORE_X_MIN, CORE_X_MAX, CORE_Y_MIN, CORE_Y_MAX,
    CORE_Z_MIN, CORE_Z_MAX,
    get_material, source_position_from_index
)

REACTION_ABSORB = 0
REACTION_SCATTER = 1
REACTION_FISSION = 2
REACTION_LEAK = 3

MAX_HISTORY = 200


@njit(cache=True)
def sample_direction():
    mu = 2.0 * np.random.random() - 1.0
    phi = 2.0 * np.pi * np.random.random()
    sin_mu = np.sqrt(1.0 - mu * mu)
    ux = sin_mu * np.cos(phi)
    uy = sin_mu * np.sin(phi)
    uz = mu
    return ux, uy, uz


@njit(cache=True)
def rotate_direction(ux, uy, uz, mu_cm, phi_cm):
    sin_cm = np.sqrt(1.0 - mu_cm * mu_cm)
    cos_phi = np.cos(phi_cm)
    sin_phi = np.sin(phi_cm)
    if abs(uz) > 0.999:
        sign = 1.0 if uz > 0 else -1.0
        ux_new = sin_cm * cos_phi
        uy_new = sign * sin_cm * sin_phi
        uz_new = sign * mu_cm
    else:
        denom = np.sqrt(1.0 - uz * uz)
        ux_new = (sin_cm * (ux * uz * cos_phi - uy * sin_phi)) / denom + ux * mu_cm
        uy_new = (sin_cm * (uy * uz * cos_phi + ux * sin_phi)) / denom + uy * mu_cm
        uz_new = -sin_cm * cos_phi * denom + uz * mu_cm
    norm = np.sqrt(ux_new ** 2 + uy_new ** 2 + uz_new ** 2)
    if norm > 0:
        ux_new /= norm
        uy_new /= norm
        uz_new /= norm
    return ux_new, uy_new, uz_new


@njit(cache=True)
def elastic_scatter_energy(energy, A, mu_cm):
    alpha = ((A - 1.0) / (A + 1.0)) ** 2
    energy_out = energy * (1.0 + alpha + (1.0 - alpha) * mu_cm) / 2.0
    if energy_out < ENERGY_GROUPS[0]:
        energy_out = ENERGY_GROUPS[0] * (1.0 + np.random.random())
    if energy_out > ENERGY_GROUPS[-1]:
        energy_out = ENERGY_GROUPS[-1]
    return energy_out


@njit(cache=True)
def simulate_neutron(x0, y0, z0, energy0):
    x = x0
    y = y0
    z = z0
    energy = energy0
    ux, uy, uz = sample_direction()

    history_x = np.empty(MAX_HISTORY, dtype=np.float64)
    history_y = np.empty(MAX_HISTORY, dtype=np.float64)
    history_z = np.empty(MAX_HISTORY, dtype=np.float64)
    history_x[0] = x
    history_y[0] = y
    history_z[0] = z
    n_history = 1

    final_x = x
    final_y = y
    final_z = z
    final_reaction = REACTION_LEAK
    fission_neutrons = 0

    for step in range(MAX_HISTORY):
        mat = get_material(x, y, z)
        if mat < 0:
            final_reaction = REACTION_LEAK
            break

        eg = get_energy_group(energy)
        sigma_t, sigma_f, sigma_c, sigma_s = get_macro_xs(mat, eg)

        if sigma_t <= 0:
            final_reaction = REACTION_LEAK
            break

        mfp = -np.log(1.0 - np.random.random()) / sigma_t
        x += ux * mfp
        y += uy * mfp
        z += uz * mfp

        if n_history < MAX_HISTORY:
            history_x[n_history] = x
            history_y[n_history] = y
            history_z[n_history] = z
            n_history += 1

        if (x < CORE_X_MIN or x > CORE_X_MAX or
            y < CORE_Y_MIN or y > CORE_Y_MAX or
            z < CORE_Z_MIN or z > CORE_Z_MAX):
            final_reaction = REACTION_LEAK
            final_x = x
            final_y = y
            final_z = z
            break

        mat = get_material(x, y, z)
        if mat < 0:
            final_reaction = REACTION_LEAK
            break

        eg = get_energy_group(energy)
        sigma_t, sigma_f, sigma_c, sigma_s = get_macro_xs(mat, eg)

        if sigma_t <= 0:
            continue

        xi = np.random.random() * sigma_t

        if xi < sigma_f:
            final_reaction = REACTION_FISSION
            final_x = x
            final_y = y
            final_z = z
            n_fission = int(NU_BAR)
            if np.random.random() < (NU_BAR - n_fission):
                n_fission += 1
            fission_neutrons = n_fission
            break
        elif xi < sigma_f + sigma_c:
            final_reaction = REACTION_ABSORB
            final_x = x
            final_y = y
            final_z = z
            break
        else:
            A = get_A(mat)
            mu_cm = 2.0 * np.random.random() - 1.0
            phi_cm = 2.0 * np.pi * np.random.random()
            energy = elastic_scatter_energy(energy, A, mu_cm)
            ux, uy, uz = rotate_direction(ux, uy, uz, mu_cm, phi_cm)

    return (final_x, final_y, final_z, final_reaction,
            fission_neutrons, n_history, history_x, history_y, history_z)


@njit(cache=True)
def simulate_batch_slim(n_neutrons, neutron_id_offset, total_source_pins):
    results_final_x = np.empty(n_neutrons, dtype=np.float64)
    results_final_y = np.empty(n_neutrons, dtype=np.float64)
    results_final_z = np.empty(n_neutrons, dtype=np.float64)
    results_reaction = np.empty(n_neutrons, dtype=np.int32)
    results_fission_n = np.empty(n_neutrons, dtype=np.int32)
    results_history_len = np.empty(n_neutrons, dtype=np.int32)

    for i in prange(n_neutrons):
        global_idx = neutron_id_offset + i
        src_idx = global_idx % total_source_pins
        x0, y0 = source_position_from_index(src_idx)
        z0 = CORE_Z_MIN + np.random.random() * (CORE_Z_MAX - CORE_Z_MIN)
        e0 = sample_fission_energy()

        (fx, fy, fz, rx, fn, hl,
         hx, hy, hz) = simulate_neutron(x0, y0, z0, e0)

        results_final_x[i] = fx
        results_final_y[i] = fy
        results_final_z[i] = fz
        results_reaction[i] = rx
        results_fission_n[i] = fn
        results_history_len[i] = hl

    return (results_final_x, results_final_y, results_final_z,
            results_reaction, results_fission_n, results_history_len)


@njit(cache=True)
def simulate_batch(n_neutrons, source_positions_x, source_positions_y, source_energies):
    n_source = len(source_positions_x)
    results_final_x = np.empty(n_neutrons, dtype=np.float64)
    results_final_y = np.empty(n_neutrons, dtype=np.float64)
    results_final_z = np.empty(n_neutrons, dtype=np.float64)
    results_reaction = np.empty(n_neutrons, dtype=np.int32)
    results_fission_n = np.empty(n_neutrons, dtype=np.int32)
    results_history_len = np.empty(n_neutrons, dtype=np.int32)

    for i in prange(n_neutrons):
        src_idx = i % n_source
        x0 = source_positions_x[src_idx]
        y0 = source_positions_y[src_idx]
        z0 = CORE_Z_MIN + np.random.random() * (CORE_Z_MAX - CORE_Z_MIN)
        e0 = source_energies[src_idx]

        (fx, fy, fz, rx, fn, hl,
         hx, hy, hz) = simulate_neutron(x0, y0, z0, e0)

        results_final_x[i] = fx
        results_final_y[i] = fy
        results_final_z[i] = fz
        results_reaction[i] = rx
        results_fission_n[i] = fn
        results_history_len[i] = hl

    return (results_final_x, results_final_y, results_final_z,
            results_reaction, results_fission_n, results_history_len)


def simulate_traced_batch(n_neutrons, source_positions_x, source_positions_y, source_energies):
    n_source = len(source_positions_x)
    all_traces_x = []
    all_traces_y = []
    all_traces_z = []
    all_reactions = []

    results_final_x = np.empty(n_neutrons, dtype=np.float64)
    results_final_y = np.empty(n_neutrons, dtype=np.float64)
    results_final_z = np.empty(n_neutrons, dtype=np.float64)
    results_reaction = np.empty(n_neutrons, dtype=np.int32)

    for i in range(n_neutrons):
        src_idx = i % n_source
        x0 = source_positions_x[src_idx]
        y0 = source_positions_y[src_idx]
        z0 = CORE_Z_MIN + np.random.random() * (CORE_Z_MAX - CORE_Z_MIN)
        e0 = source_energies[src_idx]

        (fx, fy, fz, rx, fn, hl,
         hx, hy, hz) = simulate_neutron(x0, y0, z0, e0)

        results_final_x[i] = fx
        results_final_y[i] = fy
        results_final_z[i] = fz
        results_reaction[i] = rx

        if hl > 1:
            all_traces_x.append(hx[:hl].copy())
            all_traces_y.append(hy[:hl].copy())
            all_traces_z.append(hz[:hl].copy())
            all_reactions.append(rx)

    return (results_final_x, results_final_y, results_final_z,
            results_reaction, all_traces_x, all_traces_y, all_traces_z,
            np.array(all_reactions, dtype=np.int32))
