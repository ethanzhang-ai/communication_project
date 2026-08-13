import math

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

    # 计算AWGN标准差
    noise_std = math.sqrt(1 / (2 * eb_n0_linear))

    error_count = 0
    simulated_bits = 0

    # 达到目标误码数或最大比特数后停止
    while (
        error_count < target_errors
        and simulated_bits < max_bits
    ):
        # 确定当前批次发送多少个比特
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

        # 一次产生一批高斯白噪声
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

        # 累计误码数与发送比特数
        error_count += int(batch_errors)
        simulated_bits += current_batch_size

    ber = error_count / simulated_bits

    return ber, error_count, simulated_bits


def calculate_theoretical_ber(eb_n0_db):
    """计算AWGN信道下相干BPSK的理论BER。"""

    eb_n0_linear = 10 ** (eb_n0_db / 10)

    theoretical_ber = 0.5 * math.erfc(
        math.sqrt(eb_n0_linear)
    )

    return theoretical_ber