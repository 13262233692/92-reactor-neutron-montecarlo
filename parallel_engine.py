import numpy as np
import dask
import dask.bag as db
from dask.distributed import Client, LocalCluster
from nuclear_data import sample_fission_energy, ENERGY_GROUPS
from geometry import (
    CORE_X_MIN, CORE_X_MAX, CORE_Y_MIN, CORE_Y_MAX,
    CORE_Z_MIN, CORE_Z_MAX, get_fuel_rod_positions
)
from neutron_transport import (
    simulate_batch, simulate_traced_batch,
    REACTION_ABSORB, REACTION_SCATTER, REACTION_FISSION, REACTION_LEAK
)


def create_source(n_neutrons, source_positions=None):
    if source_positions is None:
        positions = get_fuel_rod_positions(n_assemblies=3)
    else:
        positions = source_positions
    n_source = len(positions)
    src_x = np.array([p[0] for p in positions], dtype=np.float64)
    src_y = np.array([p[1] for p in positions], dtype=np.float64)
    src_e = np.array([sample_fission_energy() for _ in range(n_source)],
                     dtype=np.float64)
    return src_x, src_y, src_e


def run_simulation(n_neutrons, n_workers=None, batch_size=50000):
    if n_workers is None:
        import multiprocessing
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1,
                          memory_limit='2GB', silence_logs=True)
    client = Client(cluster)

    print(f"[*] Dask cluster started: {n_workers} workers")
    print(f"[*] Simulating {n_neutrons:,} neutrons (batch size: {batch_size:,})")

    src_x, src_y, src_e = create_source(n_neutrons)

    n_batches = max(1, n_neutrons // batch_size)
    batches = []
    for b in range(n_batches):
        n_this = batch_size if b < n_batches - 1 else (n_neutrons - b * batch_size)
        if n_this <= 0:
            continue
        batches.append((n_this, src_x, src_y, src_e))

    import neutron_transport as nt

    delayed_results = []
    for n, sx, sy, se in batches:
        res = dask.delayed(simulate_batch)(n, sx, sy, se)
        delayed_results.append(res)

    print(f"[*] Computing {len(delayed_results)} batches...")
    computed = dask.compute(*delayed_results)

    all_fx = np.concatenate([c[0] for c in computed])
    all_fy = np.concatenate([c[1] for c in computed])
    all_fz = np.concatenate([c[2] for c in computed])
    all_rx = np.concatenate([c[3] for c in computed])
    all_fn = np.concatenate([c[4] for c in computed])
    all_hl = np.concatenate([c[5] for c in computed])

    n_absorb = np.sum(all_rx == REACTION_ABSORB)
    n_fission = np.sum(all_rx == REACTION_FISSION)
    n_leak = np.sum(all_rx == REACTION_LEAK)

    print(f"[+] Simulation complete:")
    print(f"    Absorbed:  {n_absorb:>10,} ({100*n_absorb/n_neutrons:.1f}%)")
    print(f"    Fission:   {n_fission:>10,} ({100*n_fission/n_neutrons:.1f}%)")
    print(f"    Leaked:    {n_leak:>10,} ({100*n_leak/n_neutrons:.1f}%)")
    print(f"    k-eff ≈    {(n_fission * 2.43) / n_neutrons:.4f}")

    client.close()
    cluster.close()

    return {
        'final_x': all_fx, 'final_y': all_fy, 'final_z': all_fz,
        'reaction': all_rx, 'fission_n': all_fn, 'history_len': all_hl,
        'n_absorb': n_absorb, 'n_fission': n_fission, 'n_leak': n_leak
    }


def run_traced_simulation(n_neutrons, source_positions=None):
    src_x, src_y, src_e = create_source(n_neutrons, source_positions)

    print(f"[*] Tracing {n_neutrons:,} neutrons for path visualization...")

    (fx, fy, fz, rx,
     traces_x, traces_y, traces_z,
     trace_reactions) = simulate_traced_batch(
        n_neutrons, src_x, src_y, src_e
    )

    n_absorb = np.sum(rx == REACTION_ABSORB)
    n_fission = np.sum(rx == REACTION_FISSION)
    n_leak = np.sum(rx == REACTION_LEAK)

    print(f"[+] Traced simulation complete: {len(traces_x)} paths collected")

    return {
        'final_x': fx, 'final_y': fy, 'final_z': fz,
        'reaction': rx,
        'traces_x': traces_x, 'traces_y': traces_y, 'traces_z': traces_z,
        'trace_reactions': trace_reactions,
        'n_absorb': n_absorb, 'n_fission': n_fission, 'n_leak': n_leak
    }
