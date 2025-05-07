#!/usr/bin/env python3
#demo for POC generating synthetic traces and running CPA attack
"""
poc_demo.py

Proof‑of‑Concept demo for RF Side‑Channel CPA attack.
Generates synthetic RF traces under a known key, recovers one key byte with CPA,
prints results, plots correlation peaks, and saves data to CSV.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sca_utils_v4 import generate_synthetic_traces, run_cpa

def main():
    # ----- 1) POC Parameters -----
    params = {
        'num_traces':        1000,
        'samples_per_trace': 5000,
        'fs':                1e6,    # sampling rate (Hz)
        'fc':                1e5,    # carrier frequency (Hz)
        'snr':               30,     # signal‑to‑noise ratio (dB)
        'enhanced_model':    False   # use Hamming‑weight model
    }
    byte_index = 0  # which AES key byte to recover

    # Display POC parameters
    print("=== POC Demo Parameters ===")
    for k, v in params.items():
        print(f"{k:>18}: {v}")
    print(f"{'target_byte':>18}: {byte_index}\n")

    # ----- 2) Generate synthetic traces -----
    # generate_synthetic_traces returns (traces, plaintexts, true_key_byte)
    traces, plaintexts, true_key = generate_synthetic_traces(
        num_traces=params['num_traces'],
        samples_per_trace=params['samples_per_trace'],
        fs=params['fs'],
        fc=params['fc'],
        snr=params['snr'],
        enhanced_model=params['enhanced_model']
    )

    # true_key is returned as a scalar (the byte we're attacking)
    actual = int(true_key)
    print(f"Secret key byte used in simulation = 0x{actual:02X}\n")

    # ----- 3) Run CPA Attack -----
    correlations, max_corrs, recovered = run_cpa(
        traces,
        plaintexts,
        byte_index=byte_index,
        vectorized=True
    )

    # ----- 4) Print Results -----
    print(f"True key byte   = 0x{actual:02X}")
    print(f"Recovered key   = 0x{recovered:02X}")
    print(f"Peak correlation= {max_corrs[recovered]:.4f}")

    if recovered == actual:
        print("✅ SUCCESS: CPA recovered the correct key byte!")
    else:
        print("❌ ERROR: CPA did not recover the key.")

    # ----- 5) Plot Correlation vs. Key Guess -----
    plt.figure(figsize=(8,4))
    guesses = np.arange(256)
    plt.plot(guesses, max_corrs, marker='o', linestyle='-')
    plt.scatter(recovered, max_corrs[recovered],
                s=100, facecolors='none', edgecolors='r',
                label=f'Recovered: 0x{recovered:02X}')
    plt.xlabel('Key Guess (0–255)')
    plt.ylabel('Max Pearson Correlation')
    plt.title(f'CPA Result for Byte {byte_index}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----- 6) Export CSV of max correlations -----
    df = pd.DataFrame({
        'key_guess':       guesses,
        'max_correlation': max_corrs
    })
    csv_path = 'poc_byte0_cpa.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved correlation data to {csv_path}")

if __name__ == "__main__":
    main()
