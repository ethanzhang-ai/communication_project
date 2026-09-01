# BPSK BER Simulation

This project simulates the bit error rate (BER) performance of BPSK transmission over an AWGN channel and compares the simulation results with the theoretical BER.

## Testing

Run the automated test suite from the project root:

```bash
python -m pytest -q
```

The current test suite checks the known theoretical BER at a reference Eb/N0 value and verifies that theoretical BER decreases as Eb/N0 increases. It also checks simulation reproducibility by using independent random number generators initialized with the same seed. Finally, the tests verify that the simulation respects the maximum-bit constraint and that the returned BER, error count, and simulated bit count are internally consistent. These automated tests help detect regressions when the simulation code is modified and provide a reproducible foundation for future extensions of the communication system.The tests avoid relying on accidental random outcomes and therefore remain stable across repeated runs.

## Simulation Model

This project simulates BPSK transmission over an AWGN channel. Randomly generated binary bits are mapped to BPSK symbols,where bit 0 is represented by -1 and bit 1 is represented by +1. Gaussian noise is added to transmitted symbols according to the Eb/N0 value. The receiver recovers binary bits using a hard decision based on whether the received symbols are greater than the decision threshold of zero.BER is calulated by comparing transmitted bits with recovered bits and the equation is The receiver recovers binary bits using a hard decision based on whether the received symbols are greater than the decision threshold of zero. The BER is calculated by comparing the transmitted bits with recovered bits and dividing the number of bits errors by the total number of transmitted bits.