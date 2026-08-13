import matplotlib.pyplot as plt
import numpy as np

from bpsk_simulation import calculate_theoretical_ber


def plot_ber_curve(
    simulation_eb_n0_db,
    simulation_ber,
    output_path,
):
    """绘制BPSK理论BER曲线和仿真点。"""

    # 生成平滑理论曲线所需的横坐标
    minimum_eb_n0 = min(simulation_eb_n0_db)
    maximum_eb_n0 = max(simulation_eb_n0_db)

    theory_eb_n0_db = np.linspace(
        minimum_eb_n0,
        maximum_eb_n0,
        101,
    )

    # 计算每个Eb/N0对应的理论BER
    theory_ber = np.array([
        calculate_theoretical_ber(value)
        for value in theory_eb_n0_db
    ])

    # 创建图形
    plt.figure(figsize=(7, 5))

    # 绘制理论曲线
    plt.semilogy(
        theory_eb_n0_db,
        theory_ber,
        label="Theoretical BPSK",
    )

    # 绘制仿真点
    plt.semilogy(
        simulation_eb_n0_db,
        simulation_ber,
        "o",
        label="Vectorized simulation",
    )

    # 设置图形
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BPSK over AWGN Channel")
    plt.grid(True, which="both")
    plt.legend()
    plt.ylim(1e-7, 1)

    # 保存并关闭图形
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()