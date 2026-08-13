import argparse
import csv
import time
from pathlib import Path

import numpy as np

from bpsk_simulation import (
    calculate_theoretical_ber,
    simulate_ber_vectorized,
)
from plot_results import plot_ber_curve


def parse_arguments():
    """读取用户在终端中输入的仿真参数。"""

    parser = argparse.ArgumentParser(
        description="Simulate BPSK BER over an AWGN channel."
    )

    parser.add_argument(
        "--min-eb-n0",
        type=int,
        default=0,
        help="Minimum Eb/N0 in dB. Default: 0",
    )

    parser.add_argument(
        "--max-eb-n0",
        type=int,
        default=10,
        help="Maximum Eb/N0 in dB. Default: 10",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=2,
        help="Eb/N0 step in dB. Default: 2",
    )

    parser.add_argument(
        "--target-errors",
        type=int,
        default=200,
        help="Target number of bit errors. Default: 200",
    )

    parser.add_argument(
        "--max-bits",
        type=int,
        default=10_000_000,
        help="Maximum simulated bits per point. Default: 10000000",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Number of bits per NumPy batch. Default: 10000",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    return parser.parse_args()


def main():
    """运行完整的BPSK误码率实验。"""

    # 读取命令行参数
    args = parse_arguments()

    # 检查参数是否合理
    if args.min_eb_n0 > args.max_eb_n0:
        raise ValueError(
            "min-eb-n0不能大于max-eb-n0"
        )

    if args.step <= 0:
        raise ValueError("step必须大于0")

    if args.target_errors <= 0:
        raise ValueError("target-errors必须大于0")

    if args.max_bits <= 0:
        raise ValueError("max-bits必须大于0")

    if args.batch_size <= 0:
        raise ValueError("batch-size必须大于0")

    # 确定项目目录
    project_directory = Path(__file__).resolve().parent

    csv_path = project_directory / "ber_results.csv"
    image_path = project_directory / "ber_curve_numpy.png"

    # 创建随机数生成器
    rng = np.random.default_rng(args.seed)

    # 生成需要测试的Eb/N0列表
    simulation_eb_n0_db = list(
        range(
            args.min_eb_n0,
            args.max_eb_n0 + 1,
            args.step,
        )
    )

    simulation_ber = []
    simulation_results = []

    print("Simulation settings:")
    print(f"  Eb/N0 points: {simulation_eb_n0_db}")
    print(f"  Target errors: {args.target_errors}")
    print(f"  Maximum bits: {args.max_bits}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Random seed: {args.seed}")
    print()

    # 依次仿真每一个Eb/N0
    for eb_n0_db in simulation_eb_n0_db:
        start_time = time.perf_counter()

        ber, error_count, simulated_bits = (
            simulate_ber_vectorized(
                eb_n0_db=eb_n0_db,
                rng=rng,
                target_errors=args.target_errors,
                max_bits=args.max_bits,
                batch_size=args.batch_size,
            )
        )

        elapsed_time = (
            time.perf_counter() - start_time
        )

        theoretical_ber = calculate_theoretical_ber(
            eb_n0_db
        )

        simulation_ber.append(ber)

        if error_count >= args.target_errors:
            stop_reason = "target errors reached"
        else:
            stop_reason = "maximum bits reached"

        simulation_results.append([
            eb_n0_db,
            ber,
            theoretical_ber,
            simulated_bits,
            error_count,
            elapsed_time,
            stop_reason,
        ])

        print(
            f"Eb/N0: {eb_n0_db:2d} dB | "
            f"Bits: {simulated_bits:9d} | "
            f"Errors: {error_count:4d} | "
            f"Simulation BER: {ber:.3e} | "
            f"Theoretical BER: {theoretical_ber:.3e} | "
            f"Time: {elapsed_time:.3f} s | "
            f"{stop_reason}"
        )

    # 保存CSV
    with csv_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Eb/N0 (dB)",
            "Simulation BER",
            "Theoretical BER",
            "Simulated bits",
            "Error count",
            "Elapsed time (s)",
            "Stop reason",
        ])

        writer.writerows(simulation_results)

    # 绘制曲线
    plot_ber_curve(
        simulation_eb_n0_db=simulation_eb_n0_db,
        simulation_ber=simulation_ber,
        output_path=image_path,
    )

    print()
    print(f"Results saved as: {csv_path}")
    print(f"Curve saved as:   {image_path}")


if __name__ == "__main__":
    main()