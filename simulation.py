import random
import math
import matplotlib.pyplot as plt


def simulate_ber(
    eb_n0_db,
    target_errors=200,
    max_bits=2_000_000,
):
    """自适应模拟一个Eb/N0条件下的BPSK误码率。"""

    # 把Eb/N0从dB转换为线性值
    eb_n0_linear = 10 ** (eb_n0_db / 10)

    # 根据Eb/N0计算AWGN标准差
    noise_std = math.sqrt(1 / (2 * eb_n0_linear))

    error_count = 0
    simulated_bits = 0

    # 没有收集够误码，并且没有达到最大比特数时继续
    while (
        error_count < target_errors
        and simulated_bits < max_bits
    ):
        # 产生随机比特
        bit = random.randint(0, 1)

        # BPSK调制
        transmitted_signal = 1 if bit == 1 else -1

        # 通过AWGN信道
        noise = random.gauss(0, noise_std)
        received_signal = transmitted_signal + noise

        # 以0为门限进行判决
        received_bit = 1 if received_signal > 0 else 0

        # 统计误码
        if received_bit != bit:
            error_count += 1

        # 无论是否误码，都发送了一个比特
        simulated_bits += 1

    ber = error_count / simulated_bits

    return ber, error_count, simulated_bits


# 固定随机种子，保证结果可以复现
random.seed(42)

target_errors = 200
max_bits = 2_000_000

# 仿真0、2、4、6、8 dB
simulation_eb_n0_db = list(range(0, 9, 2))
simulation_ber = []

for eb_n0_db in simulation_eb_n0_db:
    ber, error_count, simulated_bits = simulate_ber(
        eb_n0_db,
        target_errors,
        max_bits,
    )

    simulation_ber.append(ber)

    theoretical_ber = 0.5 * math.erfc(
        math.sqrt(10 ** (eb_n0_db / 10))
    )

    if error_count >= target_errors:
        stop_reason = "reached target errors"
    else:
        stop_reason = "reached maximum bits"

    print(
        f"Eb/N0: {eb_n0_db:2d} dB | "
        f"Bits: {simulated_bits:8d} | "
        f"Errors: {error_count:3d} | "
        f"Simulation BER: {ber:.3e} | "
        f"Theoretical BER: {theoretical_ber:.3e} | "
        f"{stop_reason}"
    )


# 生成0～10 dB的平滑理论曲线
theory_eb_n0_db = [
    value / 10
    for value in range(0, 101)
]

theory_ber = [
    0.5 * math.erfc(
        math.sqrt(10 ** (value / 10))
    )
    for value in theory_eb_n0_db
]


# 绘制理论曲线
plt.semilogy(
    theory_eb_n0_db,
    theory_ber,
    label="Theoretical BPSK",
)

# 绘制仿真结果
plt.semilogy(
    simulation_eb_n0_db,
    simulation_ber,
    "o",
    label="Simulation",
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BPSK over AWGN Channel")
plt.grid(True, which="both")
plt.legend()
plt.ylim(1e-6, 1)

plt.savefig(
    "ber_curve.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print("Curve saved as ber_curve.png")