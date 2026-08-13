import random
import math
import matplotlib.pyplot as plt


def simulate_ber(eb_n0_db, total_bits):
    """模拟一个Eb/N0条件下的BPSK误码率。"""

    eb_n0_linear = 10 ** (eb_n0_db / 10)
    noise_std = math.sqrt(1 / (2 * eb_n0_linear))

    error_count = 0

    for _ in range(total_bits):
        # 产生随机比特
        bit = random.randint(0, 1)

        # BPSK调制：0映射为-1，1映射为+1
        transmitted_signal = 1 if bit == 1 else -1

        # AWGN信道
        noise = random.gauss(0, noise_std)
        received_signal = transmitted_signal + noise

        # 以0为门限进行判决
        received_bit = 1 if received_signal > 0 else 0

        # 统计误码
        if received_bit != bit:
            error_count += 1

    return error_count / total_bits


# 固定随机种子，使每次运行可以得到相同结果
random.seed(42)

total_bits = 100000

# 仿真点：0、2、4、6、8 dB
simulation_eb_n0_db = list(range(0, 9, 2))
simulation_ber = []

for eb_n0_db in simulation_eb_n0_db:
    ber = simulate_ber(eb_n0_db, total_bits)
    simulation_ber.append(ber)

    theoretical_ber = 0.5 * math.erfc(
        math.sqrt(10 ** (eb_n0_db / 10))
    )

    print(
        f"Eb/N0: {eb_n0_db:2d} dB | "
        f"Simulation BER: {ber:.3e} | "
        f"Theoretical BER: {theoretical_ber:.3e}"
    )


# 生成更平滑的理论曲线：0～10 dB，每隔0.1 dB取一点
theory_eb_n0_db = [value / 10 for value in range(0, 101)]

theory_ber = [
    0.5 * math.erfc(math.sqrt(10 ** (value / 10)))
    for value in theory_eb_n0_db
]


# 绘图
plt.semilogy(
    theory_eb_n0_db,
    theory_ber,
    label="Theoretical BPSK",
)

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

plt.savefig("ber_curve.png", dpi=200, bbox_inches="tight")
plt.show()