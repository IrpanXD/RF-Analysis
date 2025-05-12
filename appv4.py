import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import time
import base64
from io import BytesIO
from scipy.signal import decimate


# Import utility functions
from sca_utils_v4 import (
    generate_synthetic_traces, export_to_wav,
    run_cpa, plot_cpa_results,
    align_traces, visualize_alignment, compare_keys, plot_full_key_recovery
)

# Page configuration
st.set_page_config(
    page_title="RF Side-Channel Analysis Demo",
    page_icon="🔒",
    layout="wide",
)

#-----------------------------------------------------------------------------
# Initialize session state variables
#-----------------------------------------------------------------------------
def initialize_session_state():
    """Initialize all session state variables if they don't exist yet."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None
    if 'traces' not in st.session_state:
        st.session_state.traces = None
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'envelope' not in st.session_state:
        st.session_state.envelope = None
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False
    if 'plaintexts' not in st.session_state:
        st.session_state.plaintexts = None
    if 'aligned_traces' not in st.session_state:
        st.session_state.aligned_traces = None
    if 'full_key_recovery' not in st.session_state:
        st.session_state.full_key_recovery = False
    if 'recovered_key' not in st.session_state:
        st.session_state.recovered_key = None
    
    # Set processing_done if we already have processed data
    if st.session_state.traces is not None and st.session_state.plaintexts is not None:
        st.session_state.processing_done = True

initialize_session_state()

#-----------------------------------------------------------------------------
# Application Header
#-----------------------------------------------------------------------------
st.title("📡 RF Side-Channel Analysis Pipeline")
st.markdown(
    """
    This app demonstrates an RF side-channel attack pipeline for AES.
    Generate synthetic traces and use them for side-channel analysis.
    """
)

#-----------------------------------------------------------------------------
# Create tabs for the workflow
#-----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Generate Synthetic", "2. Process", "3. CPA Attack", "4. Documentation"
])

#-----------------------------------------------------------------------------
# Tab 1: Generate Synthetic Traces
#-----------------------------------------------------------------------------
with tab1:
    st.header("Step 1: Generate Synthetic Traces")
    
    # Parameters configuration in two columns
    col1, col2 = st.columns(2)
    with col1:
        num_traces = st.number_input("# of traces", min_value=100, max_value=20000, value=5000, step=100)
        samples = st.number_input("Samples per trace", min_value=1000, max_value=10000, value=5000, step=100)
        fs = st.number_input("Sampling freq (Hz)", min_value=100_000, max_value=10_000_000, value=1_000_000, step=100_000)
    
    with col2:
        fc = st.number_input("Carrier freq (Hz)", min_value=10_000, max_value=500_000, value=100_000, step=10_000)
        snr = st.number_input("SNR (dB)", min_value=0, max_value=100, value=60, step=5)
        leakage_model = st.selectbox(
            "Leakage Model", 
            ["Hamming Weight (Classic)", "Hamming Distance (Enhanced)"]
        )

    # Determine which leakage model to use
    use_enhanced_model = leakage_model == "Hamming Distance (Enhanced)"
    
    # Generate traces button
    if st.button("Generate Synthetic Traces"):
        with st.spinner("Generating..."):
            traces, plaintexts, true_key = generate_synthetic_traces(
                num_traces=num_traces,
                samples_per_trace=samples,
                fs=fs, fc=fc, snr=snr,
                enhanced_model=use_enhanced_model
            )
        
        # Store key info and display to user
        st.write("### Key Information")
        if use_enhanced_model:
            st.write(f"Generated traces with enhanced model and key: {[f'0x{k:02X}' for k in true_key]}")
        else:
            st.write(f"Generated traces with classic model and key byte: 0x{true_key:02X}")
        
        # Update session state
        st.session_state.update({
            'traces': traces,
            'plaintexts': plaintexts,
            'true_key': true_key,
            'fs': fs,
            'fc': fc,
            'data_loaded': True,
            'data_source': 'synthetic',
            'processing_done': True
        })
        
        st.success(f"{len(traces)} synthetic traces generated.")
        
        # Plot example trace
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(traces[0])
        
        # Set plot title based on model
        if use_enhanced_model:
            ax.set_title("Example Trace (Enhanced Leakage Model)")
        else:
            hw = bin(plaintexts[0][0] ^ true_key).count('1')
            ax.set_title(f"Example Trace (HW = {hw})")
                
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        st.pyplot(fig)
        
        # Display success message with key info
        if isinstance(true_key, (int, np.uint8)):
            st.success(f"Generated {num_traces} synthetic traces with key byte = 0x{true_key:02X}")
        else:
            key_str = ', '.join([f"0x{k:02X}" for k in true_key])
            st.success(f"Generated {num_traces} synthetic traces with key bytes: {key_str}")
    
    # Export to WAV section
    st.subheader("Export Synthetic Traces to WAV")

    if st.session_state.get('traces') is not None and st.session_state.data_source == 'synthetic':

        if st.button("Create WAV file for download"):
            with st.spinner("Creating WAV file..."):
                # 1) Create a temporary WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    export_path = tmp.name
                    export_to_wav(st.session_state.traces, st.session_state.fs, export_path)

                # 2) Read file and create download link
                with open(export_path, "rb") as f:
                    wav_bytes = f.read()
                b64 = base64.b64encode(wav_bytes).decode()
                href = (
                    f'<a href="data:audio/wav;base64,{b64}" '
                    'download="synthetic_traces.wav">Download WAV File</a>'
                )
                st.markdown(href, unsafe_allow_html=True)

                # 3) Clean up temp file
                os.remove(export_path)

    else:
        st.info("Generate synthetic traces first to enable export.")

#-----------------------------------------------------------------------------
# Tab 2: Demodulation & Processing
#-----------------------------------------------------------------------------
with tab2:
    st.header("Step 2: Process Traces")
    
    if not st.session_state.get('data_loaded'):
        st.warning("Generate data first in Step 1.")
    else:
        # Section 2.1: Trace Alignment
        if st.session_state.get('traces') is not None:
            st.subheader("2.1 Trace Alignment")
            
            st.write("Align traces to improve attack success rate.")
            
            # Method selection
            align_method = st.selectbox(
                "Alignment Method",
                ["Cross-Correlation", "Sum of Absolute Differences", "Maximum Peak"]
            )
            
            # Reference trace options
            ref_trace_option = st.selectbox(
                "Reference Trace",
                ["First Trace", "Average of All Traces", "Trace with Highest SNR"]
            )
            
            # Window options
            use_window = st.checkbox("Use alignment window", value=True)
            if use_window:
                col1, col2 = st.columns(2)
                with col1:
                    window_start = st.number_input("Window Start (%)", min_value=0, max_value=100, value=25, step=5)
                with col2:
                    window_end = st.number_input("Window End (%)", min_value=0, max_value=100, value=75, step=5)
            else:
                window_start = 0
                window_end = 100
            
            # Convert percentages to indices
            trace_length = st.session_state.traces.shape[1]
            window_start_idx = int(trace_length * window_start / 100)
            window_end_idx = int(trace_length * window_end / 100)
            
            if st.button("Align Traces"):
                with st.spinner("Aligning traces..."):
                    # Get original traces
                    original_traces = st.session_state.traces
                    
                    # Call alignment function
                    method_param = align_method.lower().replace(" ", "_")
                    reference_param = ref_trace_option.lower().replace(" ", "_")
                    window_param = (window_start_idx, window_end_idx) if use_window else None
                    
                    aligned_traces = align_traces(
                        original_traces,
                        method=method_param,
                        reference=reference_param,
                        window=window_param
                    )
                    
                    # Store the aligned traces
                    st.session_state.aligned_traces = aligned_traces
                    
                    # Create visualization
                    fig = visualize_alignment(original_traces, aligned_traces, num_to_show=5)
                    st.pyplot(fig)
                    
                    # Add option to use aligned traces
                    if st.checkbox("Use aligned traces for attack", value=True):
                        st.session_state.traces = aligned_traces
                        st.success("Traces aligned and set for attack.")
                    else:
                        st.info("Keeping original traces. You can switch to aligned traces later.")

#-----------------------------------------------------------------------------
# Tab 3: Run CPA
#-----------------------------------------------------------------------------
with tab3:
    st.header("Step 3: Correlation Power Analysis")
    
    # Verify necessary data is available
    data_ready = (st.session_state.get('processing_done') and 
                 st.session_state.get('traces') is not None and 
                 st.session_state.get('plaintexts') is not None)
    
    if not data_ready:
        st.warning("Generate data first in Step 1.")
        
        # Show debugging information in expandable section
        debug_info = {
            "processing_done": st.session_state.get('processing_done', False),
            "traces_available": 'traces' in st.session_state and st.session_state.traces is not None,
            "plaintexts_available": 'plaintexts' in st.session_state and st.session_state.plaintexts is not None,
            "data_source": st.session_state.get('data_source', 'None')
        }
        st.expander("Debug Information").json(debug_info)
    else:
        # Choose attack mode
        attack_mode = st.radio(
            "Attack Mode",
            ["Single Byte Attack", "Full Key Recovery (All 16 Bytes)"]
        )
        
        # Set the full key recovery flag
        st.session_state.full_key_recovery = (attack_mode == "Full Key Recovery (All 16 Bytes)")
        
        #-------------------------
        # Full Key Recovery Mode
        #-------------------------
        if st.session_state.full_key_recovery:
            # Option to use aligned traces if available
            from sca_utils_v4 import align_max_peak
            attack_traces = align_max_peak(st.session_state.traces)
            if st.session_state.get('aligned_traces') is not None:
                use_aligned = st.checkbox("Use aligned traces", value=True)
                if use_aligned:
                    attack_traces = st.session_state.aligned_traces
                    st.info("Using aligned traces for attack.")
                else:
                    st.info("Using original traces for attack.")
            
            if st.button("Run Full Key Recovery"):
                with st.spinner("Running CPA on all 16 bytes..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    # Initialize arrays for storing results
                    correlations_all = []
                    max_correlations_all = []
                    recovered_key = np.zeros(16, dtype=np.uint8)
                    
                    # Run CPA for each byte
                    start_time = time.time()
                    for byte_idx in range(16):
                        cor, mcor, best = run_cpa(
                            attack_traces,
                            st.session_state.plaintexts,
                            byte_idx,
                            vectorized=False   # per‑sample CPA
                        )
                        
                        # Store results
                        correlations_all.append(cor)
                        max_correlations_all.append(mcor)
                        recovered_key[byte_idx] = best
                        
                        # Update progress
                        progress_bar.progress((byte_idx + 1) / 16)
                    
                    progress_bar.progress(100)
                    end_time = time.time()
                
                # Store the recovered key
                st.session_state.recovered_key = recovered_key
                
                # Display the key comparison
                true_key = st.session_state.get('true_key')
                fig = compare_keys(recovered_key, true_key=true_key)
                st.pyplot(fig)
                
                # Display the detailed results
                st.subheader("Detailed Results")
                fig = plot_full_key_recovery(
                    correlations_all, 
                    max_correlations_all, 
                    recovered_key, 
                    true_key=true_key
                )
                st.pyplot(fig)
                
                # Display recovered key as hex string
                key_hex = ' '.join([f"{b:02X}" for b in recovered_key])
                st.success(f"Recovered Key: {key_hex}")
                st.write(f"Attack time: {end_time - start_time:.2f} seconds")
        
        #-------------------------
        # Single Byte Attack Mode
        #-------------------------
        else:
            # Target byte index
            byte_index = st.number_input("Target byte index", min_value=0, max_value=15, value=0, step=1)
            
            # Option to use vectorized implementation
            use_vectorized = st.checkbox("Use vectorized implementation", value=True)
            
            if st.button("Run CPA Attack"):
                with st.spinner("Running CPA..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    # Run CPA
                    # 1) Align traces so the leakage bump lines up
                    from sca_utils_v4 import align_max_peak
                    aligned = align_max_peak(st.session_state.traces)

                    # 2) Run CPA per‑sample (no vectorization) on the aligned data
                    start_time = time.time()
                    cor, mcor, best = run_cpa(
                        aligned,
                        st.session_state.plaintexts,
                        byte_index,
                        vectorized=use_vectorized
                    )
                    progress_bar.progress(100)
                    end_time = time.time()

                
                # Store results
                st.session_state.update({
                    'correlations': cor,
                    'max_correlations': mcor,
                    'best_key': best
                })
                
                # Display results
                st.subheader("CPA Results")
                
                # Get the true key if available
                true_key = st.session_state.get('true_key')
                
                # Get key byte value for comparison
                key_byte_value = None
                if true_key is not None:
                    if isinstance(true_key, (int, np.uint8, np.integer)):
                        key_byte_value = true_key
                    else:
                        key_byte_value = true_key[byte_index] if byte_index < len(true_key) else None
                
                # Display success/failure message
                if key_byte_value is not None:
                    if key_byte_value == best:
                        st.success(f"✅ Correct key byte 0x{best:02X} recovered!")
                    else:
                        st.error(f"❌ Mismatch: found 0x{best:02X}, true key is 0x{key_byte_value:02X}")
                else:
                    st.info(f"Most likely key byte: 0x{best:02X}")
                
                # Plot results - handle both scalar and array cases for true_key
                true_key_for_plot = None
                if true_key is not None:
                    if isinstance(true_key, (int, np.uint8, np.integer)):
                        true_key_for_plot = true_key
                    else:
                        true_key_for_plot = true_key[byte_index] if byte_index < len(true_key) else None
                
                fig = plot_cpa_results(cor, mcor, best, true_key_for_plot)
                st.pyplot(fig)
                
                # Show detailed diagnostics if requested
                if st.checkbox("Show CPA Diagnostic Info", value=False):
                    st.write("### CPA Diagnostic Information")
                    st.write(f"Target byte index: {byte_index}")
                    st.write(f"Number of traces used: {len(st.session_state.traces)}")
                    st.write(f"Best key guess: 0x{best:02X} with correlation: {mcor[best]:.4f}")
                    
                    # Show the top 5 key guesses with correlations
                    sorted_keys = np.argsort(mcor)[::-1][:5]
                    st.write("Top 5 key guesses:")
                    for k in sorted_keys:
                        st.write(f"Key 0x{k:02X}: Correlation = {mcor[k]:.4f}")
                    
                    # If we have the true key, show its rank
                    if true_key is not None:
                        true_byte = key_byte_value
                        if true_byte is not None:
                            rank = np.where(sorted_keys == true_byte)[0]
                            if len(rank) > 0:
                                st.write(f"True key 0x{true_byte:02X} rank: {rank[0] + 1} (correlation: {mcor[true_byte]:.4f})")
                            else:
                                st.write(f"True key 0x{true_byte:02X} not in top 5 (correlation: {mcor[true_byte]:.4f})")
                                # Find actual rank
                                all_ranks = np.argsort(mcor)[::-1]
                                true_rank = np.where(all_ranks == true_byte)[0][0] + 1
                                st.write(f"True key actual rank: {true_rank} out of 256")

#-----------------------------------------------------------------------------
# Tab 4: Documentation & Overview
#-----------------------------------------------------------------------------
with tab4:
    st.header("Documentation & Overview")

    st.markdown(""" 
    ## RF Side‑Channel Attack Pipeline (Simplified Version)

    This application demonstrates a complete side‑channel attack pipeline targeting
    the first round of AES encryption by exploiting RF/em emissions or power leakage.
    It uses Correlation Power Analysis (CPA) to recover secret keys through the
    following modular workflow:
    
    ### 1. Generate Synthetic Traces
    - Adjust carrier frequency (`fc`), sampling rate (`fs`), noise level (SNR), and number of traces.
    - Choose **Hamming‑weight** or **Hamming‑distance (toggle‑count)** leakage models.
    - **Export to WAV** for offline analysis.
    - **Export Audible Audio**: down‑sample envelope to 44.1 kHz for listening demos.
    
    ### 2. Process Traces
    - **Alignment**: cross‑correlation, SAD, or peak‑based alignment to improve attack success.

    ### 3. CPA Attack
    - **Single‑byte** and **Full‑key** modes.
    - **Per‑sample** and **Vectorized** implementations for speed.
    - Interactive plots: correlation vs. sample index, recovered key table.

    ### 4. Documentation (this tab)
    - This overview and step descriptions.

    ### About Side-Channel Analysis
    Side-channel analysis exploits physical leakage (power consumption, electromagnetic emanations) 
    from electronic devices to extract secret information. This app demonstrates how correlation 
    analysis can be used to recover cryptographic keys by analyzing the relationship between 
    the processed data and the measured signals.
    """)

    # References
    st.markdown("""
    **Key References**  
    - Mangard, Oswald & Popp, *Power Analysis Attacks: Revealing the Secrets of Smartcards* (2007)  
    - Guilley, Danger & Quisquater, *Electromagnetic Side‑Channel Analysis: From Theory to Practice* (2015)  
    - Collins *et al.*, *Software Defined Radio for Engineers* (2018)  
    """)

    # Example trace preview
    traces = st.session_state.get("traces")
    if traces is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(traces[0])
        ax.set(
            title="Example Trace",
            xlabel="Sample Index",
            ylabel="Amplitude"
        )
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("No traces available. Please generate data in Step 1.")