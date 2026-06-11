import numpy as np
import multiprocessing
import time
import os

from nuclear_data import sample_fission_energy, ENERGY_GROUPS
from geometry import (
    CORE_X_MIN, CORE_X_MAX, CORE_Y_MIN, CORE_Y_MAX,
    CORE_Z_MIN, CORE_Z_MAX, get_fuel_rod_positions,
    TOTAL_SOURCE_PINS, source_position_from_index
)
from neutron_transport import (
    simulate_batch_slim, simulate_traced_batch,
    REACTION_ABSORB, REACTION_SCATTER, REACTION_FISSION, REACTION_LEAK
)


def _get_system_memory_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        return 16.0


def _auto_batch_size(total_neutrons, n_workers):
    mem_gb = _get_system_memory_gb()
    per_neutron_bytes = 6 * 8
    worker_headroom = 0.25
    max_batch_mem = (mem_gb * (1.0 - worker_headroom)) * (1024 ** 3) / max(n_workers, 1)
    max_by_mem = int(max_batch_mem / per_neutron_bytes)
    max_by_parallel = total_neutrons // max(n_workers * 2, 1)
    batch = min(max(100000, max_by_mem), max_by_parallel, total_neutrons)
    batch = int(round(batch / 10000) * 10000)
    return max(50000, batch)


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


def _preallocate_results(n):
    return (
        np.empty(n, dtype=np.float64),
        np.empty(n, dtype=np.float64),
        np.empty(n, dtype=np.float64),
        np.empty(n, dtype=np.int32),
        np.empty(n, dtype=np.int32),
        np.empty(n, dtype=np.int32),
    )


def run_simulation_numba_multicore(n_neutrons, n_workers=None, batch_size=None):
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count())

    os.environ['NUMBA_NUM_THREADS'] = str(n_workers)

    if batch_size is None:
        batch_size = _auto_batch_size(n_neutrons, n_workers)

    print(f"[*] Mode: Pure Numba Multicore (threads={n_workers})")
    print(f"[*] Simulating {n_neutrons:,} neutrons (batch size: {batch_size:,})")
    print(f"[*] Source pins: {TOTAL_SOURCE_PINS}")

    fx, fy, fz, rx, fn, hl = _preallocate_results(n_neutrons)

    cursor = 0
    batch_id = 0
    n_batches = (n_neutrons + batch_size - 1) // batch_size
    t_total_start = time.time()

    while cursor < n_neutrons:
        end = min(cursor + batch_size, n_neutrons)
        n_this = end - cursor
        t0 = time.time()

        (bfx, bfy, bfz, brx, bfn, bhl) = simulate_batch_slim(
            n_this, cursor, TOTAL_SOURCE_PINS
        )

        fx[cursor:end] = bfx
        fy[cursor:end] = bfy
        fz[cursor:end] = bfz
        rx[cursor:end] = brx
        fn[cursor:end] = bfn
        hl[cursor:end] = bhl

        dt = time.time() - t0
        batch_id += 1
        rate = n_this / dt if dt > 0 else 0
        print(f"    Batch {batch_id:>3}/{n_batches}  n={n_this:>8,}  "
              f"{rate:>10,.0f}/s  {dt:>6.2f}s")

        cursor = end

    total_time = time.time() - t_total_start

    n_absorb = int(np.sum(rx == REACTION_ABSORB))
    n_fission = int(np.sum(rx == REACTION_FISSION))
    n_leak = int(np.sum(rx == REACTION_LEAK))

    print(f"[+] Simulation complete ({total_time:.1f}s total, "
          f"{n_neutrons/total_time:,.0f}/s peak):")
    print(f"    Absorbed:  {n_absorb:>10,} ({100*n_absorb/n_neutrons:.1f}%)")
    print(f"    Fission:   {n_fission:>10,} ({100*n_fission/n_neutrons:.1f}%)")
    print(f"    Leaked:    {n_leak:>10,} ({100*n_leak/n_neutrons:.1f}%)")
    print(f"    k-eff ≈    {(n_fission * 2.43) / n_neutrons:.4f}")

    return {
        'final_x': fx, 'final_y': fy, 'final_z': fz,
        'reaction': rx, 'fission_n': fn, 'history_len': hl,
        'n_absorb': n_absorb, 'n_fission': n_fission, 'n_leak': n_leak
    }


def run_simulation_dask_streaming(n_neutrons, n_workers=None, batch_size=None):
    import dask
    from dask.distributed import Client, LocalCluster, as_completed, secede, rejoin

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    total_pins = TOTAL_SOURCE_PINS

    if batch_size is None:
        batch_size = _auto_batch_size(n_neutrons, n_workers)

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1,
                          memory_limit='2GB', silence_logs=True)
    client = Client(cluster)

    print(f"[*] Mode: Dask Streaming (workers={n_workers})")
    print(f"[*] Simulating {n_neutrons:,} neutrons (batch size: {batch_size:,})")
    print(f"[*] Source pins: {total_pins}")

    fx, fy, fz, rx, fn, hl = _preallocate_results(n_neutrons)

    batch_specs = []
    cursor = 0
    batch_id = 0
    while cursor < n_neutrons:
        end = min(cursor + batch_size, n_neutrons)
        n_this = end - cursor
        batch_specs.append((batch_id, cursor, n_this))
        cursor = end
        batch_id += 1
    n_batches = len(batch_specs)
    print(f"[*] Dispatching {n_batches} batches with as_completed streaming...")

    t_total_start = time.time()
    futures = []
    inflight_max = min(n_workers * 3, n_batches)
    next_to_submit = 0

    def _submit(idx):
        bid, offset, n_sz = batch_specs[idx]
        return client.submit(
            simulate_batch_slim, n_sz, offset, total_pins,
            pure=False, retries=1
        )

    for _ in range(inflight_max):
        if next_to_submit < len(batch_specs):
            futures.append(_submit(next_to_submit))
            next_to_submit += 1

    completed_count = 0
    total_rate_acc = 0.0
    seq = as_completed(futures)

    for fut in seq:
        try:
            (bfx, bfy, bfz, brx, bfn, bhl) = fut.result()
        except Exception as exc:
            client.close(); cluster.close()
            raise RuntimeError(f"Batch failed: {exc}")

        bid, offset, n_sz = batch_specs[completed_count]
        fx[offset:offset + n_sz] = bfx
        fy[offset:offset + n_sz] = bfy
        fz[offset:offset + n_sz] = bfz
        rx[offset:offset + n_sz] = brx
        fn[offset:offset + n_sz] = bfn
        hl[offset:offset + n_sz] = bhl

        completed_count += 1
        dt_batch = time.time() - t_total_start
        done_n = min((completed_count) * batch_size, n_neutrons)
        rate = done_n / dt_batch if dt_batch > 0 else 0
        pct = 100 * completed_count / n_batches
        if completed_count % max(1, n_batches // 50) == 0 or completed_count == n_batches:
            print(f"    Done {completed_count:>4}/{n_batches} "
                  f"({pct:>5.1f}%)  {rate:>10,.0f}/s  "
                  f"{dt_batch:>6.1f}s")
        del fut, bfx, bfy, bfz, brx, bfn, bhl

        if next_to_submit < len(batch_specs):
            new_fut = _submit(next_to_submit)
            seq.add(new_fut)
            next_to_submit += 1

    total_time = time.time() - t_total_start

    n_absorb = int(np.sum(rx == REACTION_ABSORB))
    n_fission = int(np.sum(rx == REACTION_FISSION))
    n_leak = int(np.sum(rx == REACTION_LEAK))

    print(f"[+] Simulation complete ({total_time:.1f}s total, "
          f"{n_neutrons/total_time:,.0f}/s avg):")
    print(f"    Absorbed:  {n_absorb:>10,} ({100*n_absorb/n_neutrons:.1f}%)")
    print(f"    Fission:   {n_fission:>10,} ({100*n_fission/n_neutrons:.1f}%)")
    print(f"    Leaked:    {n_leak:>10,} ({100*n_leak/n_neutrons:.1f}%)")
    print(f"    k-eff ≈    {(n_fission * 2.43) / n_neutrons:.4f}")

    client.close()
    cluster.close()

    return {
        'final_x': fx, 'final_y': fy, 'final_z': fz,
        'reaction': rx, 'fission_n': fn, 'history_len': hl,
        'n_absorb': n_absorb, 'n_fission': n_fission, 'n_leak': n_leak
    }


def run_simulation(n_neutrons, n_workers=None, batch_size=None, mode='auto'):
    if mode == 'auto':
        mem_gb = _get_system_memory_gb()
        if n_neutrons <= 30_000_000 and mem_gb >= 16:
            mode = 'numba'
        else:
            mode = 'dask'

    if mode == 'numba':
        return run_simulation_numba_multicore(n_neutrons, n_workers, batch_size)
    elif mode == 'dask':
        return run_simulation_dask_streaming(n_neutrons, n_workers, batch_size)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'numba', 'dask', or 'auto'.")


def run_traced_simulation(n_neutrons, source_positions=None):
    if source_positions is None:
        positions = get_fuel_rod_positions(n_assemblies=3)
    else:
        positions = source_positions
    n_source = len(positions)
    src_x = np.array([p[0] for p in positions], dtype=np.float64)
    src_y = np.array([p[1] for p in positions], dtype=np.float64)
    src_e = np.array([sample_fission_energy() for _ in range(n_source)],
                     dtype=np.float64)

    print(f"[*] Tracing {n_neutrons:,} neutrons for path visualization...")

    (fx, fy, fz, rx,
     traces_x, traces_y, traces_z,
     trace_reactions) = simulate_traced_batch(
        n_neutrons, src_x, src_y, src_e
    )

    n_absorb = int(np.sum(rx == REACTION_ABSORB))
    n_fission = int(np.sum(rx == REACTION_FISSION))
    n_leak = int(np.sum(rx == REACTION_LEAK))

    print(f"[+] Traced simulation complete: {len(traces_x)} paths collected")

    return {
        'final_x': fx, 'final_y': fy, 'final_z': fz,
        'reaction': rx,
        'traces_x': traces_x, 'traces_y': traces_y, 'traces_z': traces_z,
        'trace_reactions': trace_reactions,
        'n_absorb': n_absorb, 'n_fission': n_fission, 'n_leak': n_leak
    }
