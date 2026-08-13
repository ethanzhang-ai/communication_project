import csv
import time
from pathlib import Path

import numpy as np

from bpsk_simulation import (
    calculate_theoretical_ber,
    simulate_ber_vectorized,
)
from plot_results import plot_ber_curve


def main():
    """运行完整的BPSK误码率实验。"""

    # 确定项目目录和输出文件位置
    project_directory = Path(__file__).resolve().parent

    csv_path = project_directory / "ber_results.csv"
    image_path = project_directory / "ber_curve_numpy.png"

    # 创建随机数生成器
    rng = np.random.default_rng(42)

    # 设置仿真参数
    target_errors = 200
    max_bits = 10_000_000
    batch_size = 10_000

    # 测试0、2、4、6、8、10 dB
    simulation_eb_n0_db = list(range(0, 11, 2))

    # 保存绘图和CSV数据
    simulation_ber = []
    simulation_results = []

    # 依次测试每个Eb/N0
    for eb_n0_db in simulation_eb_n0_db:
        start_time = time.perf_counter()

        ber, error_count, simulated_bits = (
            simulate_ber_vectorized(
                eb_n0_db=eb_n0_db,
                rng=rng,
                target_errors=target_errors,
                max_bits=max_bits,
                batch_size=batch_size,
            )
        )

        elapsed_time = (
            time.perf_counter() - start_time
        )

        theoretical_ber = calculate_theoretical_ber(
            eb_n0_db
        )

        simulation_ber.append(ber)

        # 判断仿真为什么停止
        if error_count >= target_errors:
            stop_reason = "target errors reached"
        else:
            stop_reason = "maximum bits reached"

        # 保存本次实验数据
        simulation_results.append([
            eb_n0_db,
            ber,
            theoretical_ber,
            simulated_bits,
            error_count,
            elapsed_time,
            stop_reason,
        ])

        # 在终端显示本次结果
        print(
            f"Eb/N0: {eb_n0_db:2d} dB | "
            f"Bits: {simulated_bits:9d} | "
            f"Errors: {error_count:4d} | "
            f"Simulation BER: {ber:.3e} | "
            f"Theoretical BER: {theoretical_ber:.3e} | "
            f"Time: {elapsed_time:.3f} s | "
            f"{stop_reason}"
        )

    # 写入CSV文件
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

    # 绘制BER曲线
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