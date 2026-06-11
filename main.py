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
                       help='Worker / Thread 数量')
    parser.add_argument('-o', '--output', type=str, default='reactor_3d.html',
                       help='输出HTML文件名 (默认: reactor_3d.html)')
    parser.add_argument('--no-trace', action='store_true',
                       help='跳过中子路径追踪')
    parser.add_argument('--no-depletion', action='store_true',
                       help='跳过燃耗毒物推演')
    parser.add_argument('--burnup-days', type=int, default=90,
                       help='燃耗天数 (默认: 90)')
    parser.add_argument('--depletion-grid', type=int, default=30,
                       help='燃耗 3D 网格分辨率 N×N×Nz (默认: 30)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批大小 (默认: 自适应)')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['auto', 'numba', 'dask'],
                       help='并行模式: auto(自动)/numba(纯Numba多核)/dask(Dask流式) (默认: auto)')
    args = parser.parse_args()

    print("=" * 70)
    print("   堆芯三维中子输运蒙特卡洛模拟与实时推演分析台")
    print("   Reactor Core 3D Neutron Transport MC Simulation Platform")
    print("=" * 70)
    print()

    print("[1/5] 编译 Numba JIT 内核 (首次运行较慢)...")
    t0 = time.time()

    from nuclear_data import (get_energy_group, get_macro_xs, get_A,
                               sample_fission_energy)
    from neutron_transport import simulate_neutron, simulate_batch_slim
    from geometry import get_material, TOTAL_SOURCE_PINS, source_position_from_index
    from depletion import solve_bateman_step, INITIAL_DENSITY
    from depletion_engine import _deplete_voxel_kernel

    get_energy_group(1.0)
    get_macro_xs(0, 0)
    get_A(0)
    sample_fission_energy()
    get_material(0.0, 0.0, 185.0)
    source_position_from_index(0)
    simulate_neutron(0.0, 0.0, 185.0, 1e6)
    simulate_batch_slim(16, 0, TOTAL_SOURCE_PINS)
    solve_bateman_step(1e14, INITIAL_DENSITY, 3600.0)

    print(f"    JIT 编译完成: {time.time()-t0:.1f}s")
    print()

    print(f"[2/5] 运行主蒙特卡洛模拟 ({args.neutrons:,} 中子, 模式: {args.mode})...")
    t1 = time.time()

    from parallel_engine import run_simulation
    results = run_simulation(args.neutrons, n_workers=args.workers,
                            batch_size=args.batch_size, mode=args.mode)

    print(f"    模拟完成: {time.time()-t1:.1f}s")
    print()

    traced_results = None
    if not args.no_trace:
        print(f"[3/5] 运行路径追踪模拟 ({args.traced:,} 中子)...")
        t2 = time.time()

        from parallel_engine import run_traced_simulation
        traced_results = run_traced_simulation(args.traced)

        print(f"    路径追踪完成: {time.time()-t2:.1f}s")
        print()
    else:
        print("[3/5] 跳过路径追踪")
        print()

    depletion_result = None
    if not args.no_depletion:
        print(f"[4/5] 运行燃耗推演 ({args.burnup_days} 天, 网格 {args.depletion_grid}×{args.depletion_grid}×{args.depletion_grid*2//3})...")
        t3 = time.time()

        from depletion_engine import run_depletion_simulation
        ng = args.depletion_grid
        depletion_result = run_depletion_simulation(
            results,
            burnup_days=args.burnup_days,
            nx=ng, ny=ng, nz=max(10, ng * 2 // 3)
        )

        print(f"    燃耗推演完成: {time.time()-t3:.1f}s")
        print()
    else:
        print("[4/5] 跳过燃耗推演")
        t3 = 0.0
        print()

    print("[5/5] 构建 3D 可视化...")
    t4 = time.time()

    from flux_calculator import compute_flux_map
    flux_data = compute_flux_map(results, nx=60, ny=60, nz=30)

    from visualization import create_dashboard
    fig = create_dashboard(results, traced_results=traced_results,
                          flux_data=flux_data, depletion_result=depletion_result)

    output_path = os.path.join(os.path.dirname(__file__), args.output)
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"    可视化构建完成: {time.time()-t4:.1f}s")
    print()

    total_time = time.time() - t0
    print("=" * 70)
    print(f"  完成! 输出文件: {output_path}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  模拟中子: {args.neutrons:,}  模式: {args.mode}")
    if not args.no_depletion:
        print(f"  燃耗天数: {args.burnup_days}  网格: {ng}×{ng}×{max(10, ng*2//3)}")
    n_a = results.get('n_absorb', 0)
    n_f = results.get('n_fission', 0)
    n_l = results.get('n_leak', 0)
    n_t = args.neutrons
    k = (n_f * 2.43) / n_t if n_t > 0 else 0
    print(f"  裂变: {n_f:,} ({100*n_f/n_t:.1f}%)  吸收: {n_a:,} ({100*n_a/n_t:.1f}%)  泄漏: {n_l:,} ({100*n_l/n_t:.1f}%)")
    print(f"  k-eff ≈ {k:.4f}")
    if depletion_result is not None:
        xe_max = depletion_result['xe135_grid'].max()
        pu_max = depletion_result['pu239_grid'].max()
        bu_max = float(np.nanmax(depletion_result['burnup_pct']))
        print(f"  Xe-135 峰值: {xe_max:.3e} at/cm³  Pu-239 峰值: {pu_max:.3e} at/cm³  最大燃耗: {bu_max:.2f}%")
    print("=" * 70)

    try:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(output_path))
    except Exception:
        pass


if __name__ == '__main__':
    main()
