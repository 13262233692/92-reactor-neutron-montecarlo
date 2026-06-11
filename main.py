import os
import sys
import time
import argparse
import numpy as np

os.environ['NUMBA_CACHE_DIR'] = os.path.join(os.path.dirname(__file__), '__pycache__')


def main():
    parser = argparse.ArgumentParser(description='堆芯三维中子输运蒙特卡洛模拟与可视化平台')
    parser.add_argument('-n', '--neutrons', type=int, default=2000000,
                       help='模拟中子总数 (默认: 2,000,000)')
    parser.add_argument('-t', '--traced', type=int, default=10000,
                       help='路径追踪中子数 (默认: 10,000)')
    parser.add_argument('-w', '--workers', type=int, default=None,
                       help='Dask worker 数量 (默认: CPU核数-1)')
    parser.add_argument('-o', '--output', type=str, default='reactor_3d.html',
                       help='输出HTML文件名 (默认: reactor_3d.html)')
    parser.add_argument('--no-trace', action='store_true',
                       help='跳过中子路径追踪')
    parser.add_argument('--batch-size', type=int, default=50000,
                       help='Dask批大小 (默认: 50,000)')
    args = parser.parse_args()

    print("=" * 70)
    print("   堆芯三维中子输运蒙特卡洛模拟与实时推演分析台")
    print("   Reactor Core 3D Neutron Transport MC Simulation Platform")
    print("=" * 70)
    print()

    print("[1/4] 编译 Numba JIT 内核 (首次运行较慢)...")
    t0 = time.time()

    from nuclear_data import (get_energy_group, get_macro_xs, get_A,
                               sample_fission_energy)
    from neutron_transport import simulate_neutron, simulate_batch
    from geometry import get_material

    get_energy_group(1.0)
    get_macro_xs(0, 0)
    get_A(0)
    sample_fission_energy()
    get_material(0.0, 0.0, 185.0)
    simulate_neutron(0.0, 0.0, 185.0, 1e6)

    src_x = np.array([0.0], dtype=np.float64)
    src_y = np.array([0.0], dtype=np.float64)
    src_e = np.array([1e6], dtype=np.float64)
    simulate_batch(10, src_x, src_y, src_e)

    print(f"    JIT 编译完成: {time.time()-t0:.1f}s")
    print()

    print(f"[2/4] 运行主蒙特卡洛模拟 ({args.neutrons:,} 中子)...")
    t1 = time.time()

    from parallel_engine import run_simulation
    results = run_simulation(args.neutrons, n_workers=args.workers,
                            batch_size=args.batch_size)

    print(f"    模拟完成: {time.time()-t1:.1f}s")
    print()

    traced_results = None
    if not args.no_trace:
        print(f"[3/4] 运行路径追踪模拟 ({args.traced:,} 中子)...")
        t2 = time.time()

        from parallel_engine import run_traced_simulation
        traced_results = run_traced_simulation(args.traced)

        print(f"    路径追踪完成: {time.time()-t2:.1f}s")
        print()
    else:
        print("[3/4] 跳过路径追踪")
        print()

    print("[4/4] 构建 3D 可视化...")
    t3 = time.time()

    from flux_calculator import compute_flux_map
    flux_data = compute_flux_map(results, nx=60, ny=60, nz=30)

    from visualization import create_dashboard
    fig = create_dashboard(results, traced_results=traced_results,
                          flux_data=flux_data)

    output_path = os.path.join(os.path.dirname(__file__), args.output)
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"    可视化构建完成: {time.time()-t3:.1f}s")
    print()

    total_time = time.time() - t0
    print("=" * 70)
    print(f"  完成! 输出文件: {output_path}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  模拟中子: {args.neutrons:,}")
    n_a = results.get('n_absorb', 0)
    n_f = results.get('n_fission', 0)
    n_l = results.get('n_leak', 0)
    n_t = args.neutrons
    k = (n_f * 2.43) / n_t if n_t > 0 else 0
    print(f"  裂变: {n_f:,} ({100*n_f/n_t:.1f}%)  吸收: {n_a:,} ({100*n_a/n_t:.1f}%)  泄漏: {n_l:,} ({100*n_l/n_t:.1f}%)")
    print(f"  k-eff ≈ {k:.4f}")
    print("=" * 70)

    try:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(output_path))
    except Exception:
        pass


if __name__ == '__main__':
    main()
