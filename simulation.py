import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def simulate_ber_vectorized(
    eb_n0_db,
    rng,
    target_errors=200,
    max_bits=10_000_000,
    batch_size=10_000,
):
    """使用NumPy分批仿真BPSK在AWGN信道中的误码率。"""

    # 将Eb/N0从dB转换为线性值
    eb_n0_linear = 10 ** (eb_n0_db / 10)

    # 计算高斯噪声标准差
    noise_std = math.sqrt(1 / (2 * eb_n0_linear))

    error_count = 0
    simulated_bits = 0

    # 达到目标误码数或最大比特数后停止
    while (
        error_count < target_errors
        and simulated_bits < max_bits
    ):
        # 最后一批可能不足batch_size
        current_batch_size = min(
            batch_size,
            max_bits - simulated_bits,
        )

        # 一次产生一批随机比特
        bits = rng.integers(
            0,
            2,
            size=current_batch_size,
            dtype=np.int8,
        )

        # BPSK调制：0映射为-1，1映射为+1
        transmitted_symbols = 2 * bits - 1

        # 产生一批AWGN噪声
        noise = rng.normal(
            loc=0,
            scale=noise_std,
            size=current_batch_size,
        )

        # 信号通过AWGN信道
        received_symbols = transmitted_symbols + noise

        # 以0为门限进行判决
        received_bits = (
            received_symbols > 0
        ).astype(np.int8)

        # 统计本批误码数
        batch_errors = np.count_nonzero(
            received_bits != bits
        )

        # 累加误码数和发送比特数
        error_count += int(batch_errors)
        simulated_bits += current_batch_size

    # 计算仿真BER
    ber = error_count / simulated_bits

    return ber, error_count, simulated_bits


def calculate_theoretical_ber(eb_n0_db):
    """计算AWGN信道下相干BPSK的理论BER。"""

    eb_n0_linear = 10 ** (eb_n0_db / 10)

    return 0.5 * math.erfc(
        math.sqrt(eb_n0_linear)
    )


# 确定simulation.py所在目录
project_directory = Path(__file__).resolve().parent

# 输出文件路径
csv_path = project_directory / "ber_results.csv"
image_path = project_directory / "ber_curve_numpy.png"

# 创建NumPy随机数生成器
rng = np.random.default_rng(42)

# 仿真参数
target_errors = 200
max_bits = 10_000_000
batch_size = 10_000

# 需要进行仿真的Eb/N0点
simulation_eb_n0_db = list(range(0, 11, 2))

# 保存绘图数据
simulation_ber = []

# 保存CSV表格数据
simulation_results = []

# 依次仿真每一个Eb/N0
for eb_n0_db in simulation_eb_n0_db:
    start_time = time.perf_counter()

    ber, error_count, simulated_bits = simulate_ber_vectorized(
        eb_n0_db=eb_n0_db,
        rng=rng,
        target_errors=target_errors,
        max_bits=max_bits,
        batch_size=batch_size,
    )

    elapsed_time = time.perf_counter() - start_time

    theoretical_ber = calculate_theoretical_ber(
        eb_n0_db
    )

    simulation_ber.append(ber)

    # 判断停止原因
    if error_count >= target_errors:
        stop_reason = "target errors reached"
    else:
        stop_reason = "maximum bits reached"

    # 保存一行实验结果
    simulation_results.append([
        eb_n0_db,
        ber,
        theoretical_ber,
        simulated_bits,
        error_count,
        elapsed_time,
        stop_reason,
    ])

    # 在终端显示结果
    print(
        f"Eb/N0: {eb_n0_db:2d} dB | "
        f"Bits: {simulated_bits:9d} | "
        f"Errors: {error_count:4d} | "
        f"Simulation BER: {ber:.3e} | "
        f"Theoretical BER: {theoretical_ber:.3e} | "
        f"Time: {elapsed_time:.3f} s | "
        f"{stop_reason}"
    )


# 把实验结果写入CSV文件
with csv_path.open(
    mode="w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file)

    # 写入标题行
    writer.writerow([
        "Eb/N0 (dB)",
        "Simulation BER",
        "Theoretical BER",
        "Simulated bits",
        "Error count",
        "Elapsed time (s)",
        "Stop reason",
    ])

    # 写入全部实验结果
    writer.writerows(simulation_results)


# 生成0～10 dB的平滑理论曲线
theory_eb_n0_db = np.linspace(0, 10, 101)

theory_ber = np.array([
    calculate_theoretical_ber(value)
    for value in theory_eb_n0_db
])


# 绘制理论BER曲线
plt.semilogy(
    theory_eb_n0_db,
    theory_ber,
    label="Theoretical BPSK",
)

# 绘制仿真BER点
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

# 保存曲线
plt.savefig(
    image_path,
    dpi=200,
    bbox_inches="tight",
)

# 关闭图形，避免WSL非交互后端警告
plt.close()

print()
print(f"Results saved as: {csv_path}")
print(f"Curve saved as:   {image_path}")