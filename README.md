# RF Side-Channel Analysis Pipeline

This project provides a streamlined RF side-channel analysis application, implemented as a Streamlit demo for educational purposes. It demonstrates the core concepts of side-channel attacks on cryptographic implementations through a synthetic trace generator and Correlation Power Analysis (CPA) engine.

---

## Key Features

1. **Synthetic Trace Generation**
   * Create amplitude-modulated traces using **Hamming Weight** or **Hamming Distance** leakage models
   * Configure carrier frequency, sampling rate, SNR, and number of traces
   * Export RF traces as WAV files for offline analysis

2. **Trace Processing**
   * Trace alignment using cross-correlation, Sum of Absolute Differences, or maximum peak methods
   * Configurable alignment window for optimization
   * Visualization of alignment effectiveness

3. **Correlation Power Analysis (CPA)**
   * Single-byte attack mode for detailed analysis
   * Full key recovery mode (all 16 bytes of AES)
   * Vectorized and per-sample implementation options for performance comparison
   * Comprehensive result visualization with correlation peaks and key byte recovery

4. **Interactive Visualization**
   * Real-time plotting of generated traces
   * CPA correlation plots, bar charts of max correlations
   * Key recovery comparison and validation
   * Diagnostic information for attack effectiveness

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rf-side-channel-analysis.git
cd rf-side-channel-analysis

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the App

Launch the Streamlit application:

```bash
streamlit run appv4.py
```

---

## Workflow Guide

### 1. Generate Synthetic Traces
- Set parameters for trace generation (number of traces, samples per trace, etc.)
- Choose between Hamming Weight (classic) or Hamming Distance (enhanced) leakage models
- Generate traces and optionally export to WAV format

### 2. Process Traces
- Align traces using your preferred method to improve attack success rate
- Compare original and aligned traces to visualize improvement

### 3. Run CPA Attack
- Choose between single-byte analysis or full 16-byte key recovery
- Run the attack and view the results with interactive visualizations
- Review detailed diagnostic information on attack effectiveness

### 4. Documentation
- Access comprehensive documentation on side-channel analysis concepts
- Review references to academic literature

---

## Using as a Python Library

Access core routines directly:

```python
from sca_utils_v4 import (
    generate_synthetic_traces, export_to_wav,
    align_traces, run_cpa, plot_cpa_results
)

# Example: Generate and attack
traces, plaintexts, true_key = generate_synthetic_traces(
    num_traces=500,
    samples_per_trace=5000,
    fs=1e6,
    fc=1e5,
    snr=30,
    enhanced_model=False
)
export_to_wav(traces, fs, 'rf_traces.wav')

# Align traces for better results
aligned_traces = align_traces(traces, method="cross_correlation")

# Run CPA attack
correlations, max_corrs, best_key = run_cpa(aligned_traces, plaintexts, byte_index=0)
```

---

## References

* Mangard, Oswald & Popp, *Power Analysis Attacks: Revealing the Secrets of Smartcards* (Springer, 2007)
* Guilley, Danger & Quisquater, *Electromagnetic Side-Channel Analysis: From Theory to Practice* (Springer, 2015)
* Collins *et al.*, *Software Defined Radio for Engineers* (Artech House, 2018)