import numpy as np
import pytest

from bpsk_simulation import (
    calculate_theoretical_ber,
    simulate_ber_vectorized,
)


def test_theoretical_ber_at_zero_db():
    expected_ber = 0.07864960352514257

    actual_ber = calculate_theoretical_ber(0)

    assert actual_ber == pytest.approx(
        expected_ber,
        rel=1e-10,
    )


def test_theoretical_ber_decreases_as_eb_n0_increases():
    eb_n0_values = [0, 2, 4, 6, 8, 10]

    ber_values = []

    for eb_n0_db in eb_n0_values:
        ber = calculate_theoretical_ber(eb_n0_db)
        ber_values.append(ber)

    for i in range(len(ber_values)-1):
        assert ber_values[i + 1] < ber_values[i]


def test_simulation_is_reproducible_with_same_seed():
    seed = 2026
    rng_1 = np.random.default_rng(seed)
    rng_2 = np.random.default_rng(seed)

    result_1 = simulate_ber_vectorized(
        eb_n0_db=4,
        target_errors=20,
        max_bits=50_000,
        batch_size=1_000,
        rng=rng_1,
    )
    result_2 = simulate_ber_vectorized(
        eb_n0_db=4,
        target_errors=20,
        max_bits=50000,
        batch_size=1000,
        rng=rng_2,
    )

    assert result_1 == result_2


def test_simulation_respects_max_bits_and_returns_valid_values():
    max_bits = 2_500
    rng = np.random.default_rng(2026)

    ber, error_count, simulated_bits = simulate_ber_vectorized(
        eb_n0_db=4,
        rng=rng,
        target_errors=max_bits + 1,
        max_bits=max_bits,
        batch_size=1_000,
    )

    assert simulated_bits == max_bits

    assert 0 <= error_count <= simulated_bits

    assert ber == pytest.approx(
        error_count / simulated_bits,
        rel=1e-10
    )

    assert 0 <= ber <= 1