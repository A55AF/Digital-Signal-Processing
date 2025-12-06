import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from signal_framework import Signal
from utils import (
    compute_fir_coff, downsample, normalize_freq, shift_freq, choose_window, compute_N,
    read_signal, apply_conv, save_txt, resample_signal, list_signals, upsample
)
import os

st.set_page_config(page_title="Task 7 - Filtering & Resampling", page_icon="▼", layout="wide")

st.title("Task 7: FIR Filtering and Resampling")

# Sidebar: Choose operation
operation = st.sidebar.radio(
    "Choose Operation",
    ["Filtering", "Resampling"]
)

if operation == "Filtering":
    st.header("FIR Filter Design and Application")
    
    # Filter specifications
    st.sidebar.subheader("Filter Specifications")
    
    filter_type = st.sidebar.radio(
        "Choose filter type",
        ["Low Pass", "High Pass", "Band Pass", "Band Stop"],
    )
    
    fs = st.sidebar.number_input("Sampling Frequency (fs)", min_value=1, value=8000, max_value=100000)
    
    fc = None
    f1 = None
    f2 = None
    
    if filter_type in ["Low Pass", "High Pass"]:
        fc = st.sidebar.number_input(
            "Cutoff Frequency (fc)", min_value=1, value=1500, max_value=int(fs/2)
        )
    else:
        f1 = st.sidebar.number_input(
            "Lower Cutoff Frequency (f1)", min_value=1, value=1000, max_value=int(fs/2)
        )
        f2 = st.sidebar.number_input(
            "Upper Cutoff Frequency (f2)", min_value=1, value=2000, max_value=int(fs/2)
        )
    
    delta_s = st.sidebar.number_input("Stopband Attenuation (δs in dB)", min_value=1, value=50, max_value=100)
    delta_f = st.sidebar.number_input("Transition Band (Δf)", min_value=1, value=500, max_value=int(fs/2))
    
    # Compute filter parameters
    half_delta_f = delta_f / 2.0
    
    if filter_type in ["Low Pass", "High Pass"]:
        fc_shifted = shift_freq(filter_type, half_delta_f, fc=fc)
        fc_norm = normalize_freq(filter_type, fs, fc=fc_shifted)
        f1_norm = None
        f2_norm = None
    else:
        f1_shifted, f2_shifted = shift_freq(filter_type, half_delta_f, f1=f1, f2=f2)
        f1_norm, f2_norm = normalize_freq(filter_type, fs, f1=f1_shifted, f2=f2_shifted)
        fc_norm = None
    
    N = compute_N(delta_f, fs, delta_s)
    window_type = choose_window(delta_s)
    
    # Compute filter coefficients
    h, h_ideal, w = compute_fir_coff(filter_type, fc_norm, f1_norm, f2_norm, N, window_type)
    
    # Display filter info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filter Order (N)", N)
    with col2:
        st.metric("Window Type", window_type.capitalize())
    with col3:
        st.metric("Number of Coefficients", len(h))
    
    # Plot filter coefficients
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Coefficients
    m = (N - 1) // 2
    indices = np.arange(-m, m + 1)
    axes[0, 0].stem(indices, h, basefmt=' ')
    axes[0, 0].set_title('Filter Coefficients h(n)')
    axes[0, 0].set_xlabel('n')
    axes[0, 0].set_ylabel('h(n)')
    axes[0, 0].grid(True)
    
    # Ideal impulse response
    axes[0, 1].stem(indices, h_ideal, basefmt=' ')
    axes[0, 1].set_title('Ideal Impulse Response hd(n)')
    axes[0, 1].set_xlabel('n')
    axes[0, 1].set_ylabel('hd(n)')
    axes[0, 1].grid(True)
    
    # Window function
    axes[1, 0].stem(indices, w, basefmt=' ')
    axes[1, 0].set_title(f'Window Function ({window_type})')
    axes[1, 0].set_xlabel('n')
    axes[1, 0].set_ylabel('w(n)')
    axes[1, 0].grid(True)
    
    # Frequency response
    freq = np.fft.fftfreq(1024, 1/fs)
    H = np.fft.fft(h, 1024)
    magnitude = np.abs(H)
    
    axes[1, 1].plot(freq[:512], 20*np.log10(magnitude[:512] + 1e-10))
    axes[1, 1].set_title('Frequency Response (Magnitude)')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude (dB)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Save coefficients
    if st.button("Save Filter Coefficients"):
        os.makedirs("signals/task7/coff", exist_ok=True)
        save_txt(h, "signals/task7/coff/coefficients.txt")
        st.success("Filter coefficients saved to signals/task7/coff/coefficients.txt")
    
    # Apply filter to signal
    st.subheader("Apply Filter to Signal")
    
    # File uploader or select from signals
    input_method = st.radio("Input Method", ["Upload File", "Select from Signals"])
    
    # Initialize session state for filtering signal
    if 'filter_signal' not in st.session_state:
        st.session_state.filter_signal = None
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload signal file", type="txt")
        if uploaded_file is not None:
            # Save temporarily and read
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.filter_signal = read_signal(temp_path)
            os.remove(temp_path)
    else:
        # List available signals
        signal_folders = ["task6/general", "task7", "task1", "task2"]
        available_signals = {}
        for folder in signal_folders:
            folder_path = os.path.join("signals", folder)
            if os.path.exists(folder_path):
                time_sigs, _ = list_signals(folder)
                for name, path in time_sigs.items():
                    available_signals[f"{folder}/{name}"] = path
        
        if available_signals:
            selected_signal = st.selectbox("Select Signal", list(available_signals.keys()))
            if st.button("Load Signal"):
                st.session_state.filter_signal = read_signal(available_signals[selected_signal])
    
    input_signal = st.session_state.filter_signal
    
    if input_signal is not None:
        t_in, a_in = input_signal.split()
        
        # Apply convolution
        a_filtered = apply_conv(a_in, h)
        t_out = t_in[:len(a_filtered)]
        
        # Plot results
        fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10))
        
        # Input signal
        axes2[0].plot(t_in, a_in, 'b-', label='Input Signal')
        axes2[0].set_title('Input Signal')
        axes2[0].set_xlabel('Time')
        axes2[0].set_ylabel('Amplitude')
        axes2[0].grid(True)
        axes2[0].legend()
        
        # Filter coefficients
        axes2[1].stem(np.arange(len(h)), h, basefmt=' ')
        axes2[1].set_title('Filter Coefficients')
        axes2[1].set_xlabel('n')
        axes2[1].set_ylabel('h(n)')
        axes2[1].grid(True)
        
        # Filtered signal
        axes2[2].plot(t_out, a_filtered, 'r-', label='Filtered Signal')
        axes2[2].set_title('Filtered Signal')
        axes2[2].set_xlabel('Time')
        axes2[2].set_ylabel('Amplitude')
        axes2[2].grid(True)
        axes2[2].legend()
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Save filtered signal
        if st.button("Save Filtered Signal"):
            os.makedirs("signals/task7/filtered", exist_ok=True)
            output_path = "signals/task7/filtered/filtered_output.txt"
            filtered_signal = Signal(False, input_signal.is_periodic, np.column_stack((t_out, a_filtered)))
            # Save in signal format
            with open(output_path, 'w') as f:
                f.write("0\n")  # Time domain
                f.write("0\n")  # Not periodic
                f.write(f"{len(a_filtered)}\n")
                for i in range(len(a_filtered)):
                    f.write(f"{t_out[i]} {a_filtered[i]}\n")
            st.success(f"Filtered signal saved to {output_path}")

else:  # Resampling
    st.header("Signal Resampling")
    
    st.sidebar.subheader("Resampling Parameters")
    
    M = st.sidebar.number_input("Decimation Factor (M)", min_value=0, value=0, max_value=100,
                                help="Set to 0 for no decimation")
    L = st.sidebar.number_input("Interpolation Factor (L)", min_value=0, value=2, max_value=100,
                                help="Set to 0 for no interpolation")
    
    if M == 0 and L == 0:
        st.error("Both M and L cannot be zero!")
    else:
        st.sidebar.subheader("Low-Pass Filter Specifications")
        fs_resample = st.sidebar.number_input("Sampling Frequency (fs)", min_value=1, value=8000, max_value=100000)
        fc_resample = st.sidebar.number_input("Cutoff Frequency (fc)", min_value=1, value=1500, max_value=int(fs_resample/2))
        delta_s_resample = st.sidebar.number_input("Stopband Attenuation (δs)", min_value=1, value=50, max_value=100)
        delta_f_resample = st.sidebar.number_input("Transition Band (Δf)", min_value=1, value=500, max_value=int(fs_resample/2))
        
        # Compute filter specs for resampling
        half_delta_f_r = delta_f_resample / 2.0
        fc_shifted_r = fc_resample + half_delta_f_r
        fc_norm_r = fc_shifted_r / fs_resample
        N_r = compute_N(delta_f_resample, fs_resample, delta_s_resample)
        window_type_r = choose_window(delta_s_resample)
        
        filter_specs = {
            'fc_norm': fc_norm_r,
            'N': N_r,
            'window_type': window_type_r
        }
        
        # Display resampling info
        col1, col2, col3 = st.columns(3)
        with col1:
            if M != 0 and L != 0:
                st.metric("Resampling Ratio", f"{L}/{M}")
            elif M == 0:
                st.metric("Upsampling Factor", f"{L}")
            else:
                st.metric("Downsampling Factor", f"{M}")
        with col2:
            st.metric("Filter Order (N)", N_r)
        with col3:
            st.metric("Window Type", window_type_r.capitalize())
        
        # Load signal
        input_method = st.radio("Input Method", ["Upload File", "Select from Signals"])
        
        # Initialize session state for resampling signal
        if 'resample_signal' not in st.session_state:
            st.session_state.resample_signal = None
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader("Upload signal file", type="txt")
            if uploaded_file is not None:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.resample_signal = read_signal(temp_path)
                os.remove(temp_path)
        else:
            signal_folders = ["task6/general", "task7", "task1", "task2"]
            available_signals = {}
            for folder in signal_folders:
                folder_path = os.path.join("signals", folder)
                if os.path.exists(folder_path):
                    time_sigs, _ = list_signals(folder)
                    for name, path in time_sigs.items():
                        available_signals[f"{folder}/{name}"] = path
            
            if available_signals:
                selected_signal = st.selectbox("Select Signal", list(available_signals.keys()))
                if st.button("Load Signal"):
                    st.session_state.resample_signal = read_signal(available_signals[selected_signal])
        
        input_signal = st.session_state.resample_signal
        
        if input_signal is not None:
            t_in, a_in = input_signal.split()
            
            # Perform resampling
            resampled_signal, filter_h = resample_signal(input_signal, M, L, filter_specs)
            t_out, a_out = resampled_signal.split()
            
            # Create practical before/after comparison
            fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8))
            
            # Original signal
            axes3[0].plot(t_in, a_in, 'b-', marker='o', markersize=4, linewidth=1.5)
            axes3[0].set_title(f'Original Signal (N={len(a_in)} samples)')
            axes3[0].set_xlabel('Sample Index')
            axes3[0].set_ylabel('Amplitude')
            axes3[0].grid(True, alpha=0.3)
            
            # Resampled signal
            axes3[1].plot(t_out, a_out, 'r-', marker='o', markersize=4, linewidth=1.5)
            if M != 0 and L != 0:
                axes3[1].set_title(f'Resampled Signal (N={len(a_out)} samples, Ratio={L}/{M}={L/M:.2f})')
            elif M == 0:
                axes3[1].set_title(f'Upsampled Signal (N={len(a_out)} samples, Factor=×{L})')
            else:
                axes3[1].set_title(f'Downsampled Signal (N={len(a_out)} samples, Factor=÷{M})')
            axes3[1].set_xlabel('Sample Index')
            axes3[1].set_ylabel('Amplitude')
            axes3[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig3)
            
            # Display statistics
            st.subheader("Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Input Samples", len(a_in))
            with col2:
                st.metric("Output Samples", len(a_out))
            with col3:
                ratio = len(a_out) / len(a_in)
                st.metric("Actual Ratio", f"{ratio:.4f}")
            
            # Save resampled signal
            if st.button("Save Resampled Signal"):
                os.makedirs("signals/task7/resampled", exist_ok=True)
                output_path = "signals/task7/resampled/resampled_output.txt"
                with open(output_path, 'w') as f:
                    f.write("0\n")
                    f.write("0\n")
                    f.write(f"{len(a_out)}\n")
                    for i in range(len(a_out)):
                        f.write(f"{t_out[i]} {a_out[i]}\n")
                st.success(f"Resampled signal saved to {output_path}")

