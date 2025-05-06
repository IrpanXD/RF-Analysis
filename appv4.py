import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import time
import base64
import tempfile
import shutil
import os
from io import BytesIO
from scipy.signal import decimate


# Import utility functions
from sca_utils_v4 import (
    generate_synthetic_traces, export_to_wav, load_rf_wav, demodulate_envelope,
    slice_traces, load_public_dataset, run_cpa, plot_cpa_results,
    align_traces, align_max_peak, visualize_alignment, compare_keys, plot_full_key_recovery, load_real_sdr_dataset
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
    Generate, import, process, and attack side-channel traces all in one place.
    """
)

#-----------------------------------------------------------------------------
# Create tabs for the workflow
#-----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Generate Synthetic", "2. Load Data", "3. Process", "4. CPA Attack", "5. Documentation"
])

#-----------------------------------------------------------------------------
# Tab 1: Generate Synthetic Traces
#-----------------------------------------------------------------------------
with tab1:
    st.header("Step 1: Generate Synthetic Traces")
    
    # Parameters configuration in two columns
    col1, col2 = st.columns(2)
    with col1:
        num_traces = st.slider("# of traces", 100, 20000, 5000)
        samples = st.slider("Samples per trace", 1000, 10000, 5000)
        fs = st.number_input("Sampling freq (Hz)", 100_000, 10_000_000, 1_000_000)
    
    with col2:
        fc = st.number_input("Carrier freq (Hz)", 10_000, 500_000, 100_000)
        snr        = st.slider("SNR (dB)",     0,   100,  60)
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


        if st.button("Create Audible WAV"):
            # 1) Demodulate the envelope of the first RF trace
            env = np.hstack([
                demodulate_envelope(
                    tr,
                    st.session_state.fs,
                    fc=st.session_state.fc,
                    bw=st.session_state.fs / 2
                )
                for tr in st.session_state.traces
            ])

            # 2) Down-sample into the audible band (44.1 kHz)
            env_ds = decimate(env, int(st.session_state.fs / 44100))

            # 3) Export to a temporary WAV and offer for download
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
                export_to_wav(env_ds[np.newaxis, :], 44100, tmp2.name)
                with open(tmp2.name, "rb") as f2:
                    wav_bytes = f2.read()

            b64 = base64.b64encode(wav_bytes).decode()
            href = (
                f'<a href="data:audio/wav;base64,{b64}" '
                'download="audible_envelope.wav">Download Audible WAV</a>'
            )
            st.markdown(href, unsafe_allow_html=True)

            # Clean up temp file
            os.remove(tmp2.name)

    else:
        st.info("Generate synthetic traces first to enable export.")


#-----------------------------------------------------------------------------
# Tab 2: Load Data
#-----------------------------------------------------------------------------
with tab2:
    st.header("Step 2: Load Data")
    source = st.radio("Data source:", ["WAV File", "Public HDF5", "Real SDR Demo"])

    # WAV File loading option
    if source == "WAV File":
        uploaded_file = st.file_uploader("Upload WAV", type='wav')
        if uploaded_file:
            with st.spinner("Loading WAV..."):
                # Save to temp file and load
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    path = tmp.name
                
                raw, fs = load_rf_wav(path)
                os.remove(path)  # Clean up
                
            # Update session state
            was_processed = st.session_state.get('processing_done', False)
            st.session_state.update({
                'raw_data': raw,
                'fs': fs,
                'data_loaded': True,
                'data_source': 'wav',
                'processing_done': was_processed
            })
            
            st.success(f"Loaded WAV: {len(raw)} samples @ {fs} Hz.")
            
            # Plot raw data sample
            fig, ax = plt.subplots(figsize=(8, 3))
            samples_to_plot = min(10000, len(raw))
            ax.plot(raw[:samples_to_plot])
            ax.set_title(f"Raw RF Data Sample (First {samples_to_plot} samples)")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
            st.pyplot(fig)  

     # HDF5 File loading option
    elif source == "Public HDF5":
        st.subheader("Load Public HDF5 Dataset")
        uploaded_file = st.file_uploader("Upload HDF5", type='h5')
        if uploaded_file:
            with st.spinner("Loading dataset..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    path = tmp.name
                
                # Load parameters
                offset = st.slider("Offset", 0, 10000, 0)
                count = st.slider("# traces to load", 10, 5000, 1000)
                
                # Load the dataset
                try:
                    traces, pts, key = load_public_dataset(path, offset, count)
                    load_success = True
                    error_msg = None
                except Exception as e:
                    load_success = False
                    error_msg = str(e)
                
                os.remove(path)  # Clean up
            
            if load_success:
                # Update session state
                st.session_state.update({
                    'traces': traces,
                    'plaintexts': pts,
                    'true_key': key[0] if key is not None else None,
                    'data_loaded': True,
                    'data_source': 'hdf5',
                    'processing_done': True
                })
                
                st.success(f"Loaded {len(traces)} traces from dataset.")
                
                # Plot example trace
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(traces[0])
                ax.set_title("Example Trace from Dataset")
                ax.set_xlabel("Sample")
                ax.set_ylabel("Amplitude")
                ax.grid(True)
                st.pyplot(fig)
                
                key_info = f"with key byte = 0x{key[0]:02X}" if key is not None else "key not provided"
                st.success(f"Loaded {len(traces)} traces from dataset {key_info}")
            else:
                st.error(f"Error loading dataset: {error_msg}")
                st.info("Please check that your file is a valid HDF5 dataset with traces and plaintext data.")


    elif source == "Real SDR Demo":
        st.subheader("Real SDR Demo: Upload your I/Q captures")
        ds = st.selectbox("Dataset format:", ["screaming_channels", "drone_rf_video", "migou_mod"])
        
        # NEW: drag-and-drop uploader for SDR files
        uploaded_files = st.file_uploader(
            "Drag & drop your SDR dump files here",
            type=["h5", "bin", "pkl"],
            accept_multiple_files=True
        )
        
        offset = st.number_input("Trace offset", 0, 10000, 0)
        count  = st.number_input("Number of traces", 10, 2000, 200)
        
        # Only enable the load button once files are present
        if uploaded_files and st.button("Load SDR Captures"):
            try:
                with st.spinner("Loading real SDR traces…"):
                    # Pass the list of UploadedFile objects instead of a path
                                # 1) Dump uploads to a temp folder
                        temp_dir = tempfile.mkdtemp()
                        for up in uploaded_files:
                            with open(os.path.join(temp_dir, up.name), "wb") as f:
                                f.write(up.getvalue())

                        # 2) Load from that folder path (a str), not the list
                        traces, pts, key = load_real_sdr_dataset(temp_dir, ds, offset, count)
    
                        # 3) Clean up
                        shutil.rmtree(temp_dir)

                
                st.session_state.update({
                    'traces': traces,
                    'plaintexts': pts,
                    'true_key': key,
                    'data_loaded': True,
                    'data_source': 'sdr_real',
                    'processing_done': True
                })
                st.success(f"Loaded {len(traces)} traces from uploaded files")
                # preview
                fig, ax = plt.subplots(figsize=(8,3))
                ax.plot(traces[0])
                ax.set_title("Example Real SDR Trace")
                ax.set_xlabel("Sample")
                ax.set_ylabel("Amplitude")
                ax.grid(True)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Failed to load SDR dataset: {e}")


#-----------------------------------------------------------------------------
# Tab 3: Demodulation & Processing
#-----------------------------------------------------------------------------
with tab3:
    st.header("Step 3: Demodulate & Slice")
    
    if not st.session_state.get('data_loaded'):
        st.warning("Load or generate data first.")
    else:
        # Section 3.1: Demodulate RF Signal (only for WAV data)
        if st.session_state.data_source == 'wav':
            st.subheader("3.1 Demodulate RF Signal")
            
            fc = st.number_input("Carrier freq (Hz)", 10_000, 500_000, 100_000, key="process_fc")
            bw = st.slider("Filter bandwidth (Hz)", 1000, 50000, 10000)
            
            if st.button("Demodulate Envelope"):
                with st.spinner("Demodulating..."):
                    env = demodulate_envelope(
                        st.session_state.raw_data,
                        st.session_state.fs,
                        fc,
                        bw
                    )
                
                st.session_state.envelope = env
                st.session_state.fc = fc
                st.success("Demodulation complete.")
                
                # Plot the envelope
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(env[:min(10000, len(env))])
                ax.set_title("Demodulated Envelope")
                ax.set_xlabel("Sample")
                ax.set_ylabel("Amplitude")
                ax.grid(True)
                st.pyplot(fig)
            
            # Section 3.2: Slice into Traces (only if envelope exists)
            if st.session_state.get('envelope') is not None:
                st.subheader("3.2 Slice into Traces")
                
                duration_ms = st.slider("Trace duration (ms)", 1, 50, 5)
                duration_sec = duration_ms / 1000
                
                # Calculate number of samples per trace
                samples_per_trace = int(duration_sec * st.session_state.fs)
                max_traces = len(st.session_state.envelope) // samples_per_trace
                
                # Number of traces to extract
                trace_count = st.slider("# traces to extract", 10, min(500, max_traces), 100)
                
                # Offset in samples
                offset = st.slider("Offset (samples)", 0, 
                                  len(st.session_state.envelope) - trace_count * samples_per_trace, 0)
                
                if st.button("Slice Traces"):
                    with st.spinner("Slicing..."):
                        # Slice the envelope into traces
                        sliced_traces = slice_traces(
                            st.session_state.envelope,
                            st.session_state.fs,
                            duration_sec,
                            trace_count,
                            offset
                        )
                        
                        # Generate random plaintexts for the sliced traces
                        plaintexts = np.random.randint(0, 256, (trace_count, 16), dtype=np.uint8)
                    
                    # Update session state
                    st.session_state.traces = sliced_traces
                    st.session_state.plaintexts = plaintexts
                    st.session_state.processing_done = True  

                    st.success("Traces sliced successfully. Ready for CPA attack.")
                    
                    # Plot an example trace
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(sliced_traces[0])
                    ax.set_title("Example Sliced Trace")
                    ax.set_xlabel("Sample")
                    ax.set_ylabel("Amplitude")
                    ax.grid(True)
                    st.pyplot(fig)
        
        # Section 3.3: Trace Alignment (for all data types if traces exist)
        if st.session_state.get('traces') is not None:
            st.subheader("3.3 Trace Alignment")
            
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
                    window_start = st.slider("Window Start (%)", 0, 100, 25)
                with col2:
                    window_end = st.slider("Window End (%)", 0, 100, 75)
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
# Tab 4: Run CPA
#-----------------------------------------------------------------------------
with tab4:
    st.header("Step 4: Correlation Power Analysis")
    
    # Verify necessary data is available
    data_ready = (st.session_state.get('processing_done') and 
                 st.session_state.get('traces') is not None and 
                 st.session_state.get('plaintexts') is not None)
    
    if not data_ready:
        st.warning("Process data first before running CPA.")
        
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
            byte_index = st.slider("Target byte index", 0, 15, 0)
            
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
                        vectorized=False        # force sample‑by‑sample CPA
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
# Tab 5: Documentation & Overview
#-----------------------------------------------------------------------------
with tab5:
    st.header("Documentation & Overview")

    st.markdown("""\ 
    ## RF Side‑Channel Attack Pipeline (v4)

    This application demonstrates a complete side‑channel attack pipeline targeting
    the first round of AES encryption by exploiting RF/em emissions or power leakage.
    It uses Correlation Power Analysis (CPA) to recover secret keys through the
    following modular workflow:
    
    ### 1. Generate Synthetic
    - Adjust carrier frequency (`fc`), sampling rate (`fs`), noise level (SNR), and number of traces.
    - Choose **Hamming‑weight** or **Hamming‑distance (toggle‑count)** leakage models.
    - **Export to WAV** for offline CPA.
    - **Export Audible Audio**: down‑sample envelope to 44.1 kHz for listening demos (not CPA‑processable).
    
    ### 2. Load Data
    - Import RF traces from WAV files (synthetic or live SDR captures).
    - Load public HDF5 datasets: ASCAD (GitHub) and SPERO (GitHub).
    - Support for offline WAV, HDF5, and raw SDR dump formats.

    ### 3. Signal Processing
    - **Demodulation**: Butterworth band‑pass filter + Hilbert transform.
    - **Alignment**: cross‑correlation, SAD, or peak‑based alignment.
    - **Slicing**: segment continuous data into fixed‑length traces per byte.

    ### 4. CPA Attack
    - **Single‑byte** and **Full‑key** modes.
    - **Per‑sample** and **Vectorized** implementations for speed.
    - Interactive plots: correlation vs. sample index, recovered key table.

    ### 5. Documentation (this tab)
    - This overview, step descriptions, and example plots.

    ### 6. About
    - Version history, change log, and attribution.

    **Tip: Increasing Upload Size for Large WAVs**

    To allow larger file uploads, start the app with:
    ```bash
    streamlit run appv4.py --server.maxUploadSize=16384
    ```
    *(Value in KB → 16384 KB = 16 MB)*  
    Alternatively, create `./.streamlit/config.toml`:
    ```toml
    [server]
    maxUploadSize = 16384
    ```
    Then run `streamlit run appv4.py` normally.
    """)

    # References
    st.markdown("""\ 
    **Key References & Datasets**  
    - Mangard, Oswald & Popp, *Power Analysis Attacks: Revealing the Secrets of Smartcards* (2007)  
    - Guilley, Danger & Quisquater, *Electromagnetic Side‑Channel Analysis: From Theory to Practice* (2015)  
    - Collins *et al.*, *Software Defined Radio for Engineers* (2018)  
    - ASCAD Dataset: https://github.com/ANSSI-FR/ASCAD  
    - SPERO Dataset: https://github.com/YunkaiUF/SPERO  
    - Zenodo RF Traces: https://zenodo.org/records/4264467  
    - Mendeley Side‑Channel: https://data.mendeley.com/datasets/fkwr8mzndr/1  
    """)

    # Example trace preview
    traces = st.session_state.get("traces")
    if traces is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(traces[0])
        ax.set(
            title="Example Demodulated Trace (Envelope)",
            xlabel="Sample Index",
            ylabel="Amplitude"
        )
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("No traces available. Please generate or load data in Steps 1 or 2 above.")
