import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from utils import list_signals, read_signal, FFT_IFFT

st.set_page_config(page_title="Task 5", page_icon="📈", layout="wide")

try:
    time_sigs, freq_sigs = list_signals(5)
    st.session_state["time-signals"] = time_sigs
    st.session_state["freq-signals"] = freq_sigs
except Exception as e:
    st.session_state["time-signals"] = {}
    st.session_state["freq-signals"] = {}


def ensure_state_keys():
    if "last-fft" not in st.session_state:
        st.session_state["last-fft"] = None
    if "last-ifft" not in st.session_state:
        st.session_state["last-ifft"] = None


ensure_state_keys()

source_type = st.sidebar.radio(
    "Operation", ["FFT (Fast Fourier Transform)", "IFFT (Inverse FFT)"], index=0
)

fs = st.sidebar.number_input(
    "Enter sampling frequency (Hz)", min_value=0.0001, value=1.0, format="%.4f"
)

if source_type.startswith("FFT"):
    time_signal_names = list(st.session_state["time-signals"].keys())
    if not time_signal_names:
        st.sidebar.write("No time signals available for FFT.")
        chosen = None
    else:
        chosen = st.sidebar.selectbox(
            "Choose time signal (for FFT)", [None] + time_signal_names
        )

    if chosen:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.header(f"Frequency analysis (FFT) of: {chosen}")
        with col2:
            if st.button("Compute FFT", key=f"fft-{chosen}"):
                try:
                    time_sig = read_signal(st.session_state["time-signals"][chosen])
                    t, a = time_sig.split()

                    freq_sig = FFT_IFFT(time_sig, inverse=0)

                    N = len(freq_sig.matrix[:, 0])
                    freqs = np.arange(N) * fs / N

                    amps = freq_sig.matrix[:, 0].astype(float)
                    phases = freq_sig.matrix[:, 1].astype(float)

                    st.session_state["last-fft"] = {
                        "name": chosen,
                        "fs": fs,
                        "freqs": freqs.tolist(),
                        "amps": amps.tolist(),
                        "phases": phases.tolist(),
                    }
                    st.success(f"FFT computed for {chosen} (N={N})")
                except Exception as e:
                    import traceback

                    st.error(f"FFT computation failed: {e}")
                    st.text(traceback.format_exc())

    if (
        st.session_state.get("last-fft")
        and chosen
        and st.session_state["last-fft"].get("name") == chosen
    ):
        data = st.session_state["last-fft"]
        freqs = np.array(data["freqs"])
        amps = np.array(data["amps"], dtype=float)
        phases = np.array(data["phases"], dtype=float)

        st.subheader("Frequency vs Amplitude")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.stem(freqs, amps, basefmt=" ")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude")
        ax1.set_title(f"Frequency Domain - Amplitude Spectrum ({data.get('name', 'Signal')})")
        ax1.grid(True)
        st.pyplot(fig1)

        st.subheader("Frequency vs Phase")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.stem(freqs, phases, basefmt=" ")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (rad)")
        ax2.set_title(f"Frequency Domain - Phase Spectrum ({data.get('name', 'Signal')})")
        ax2.grid(True)
        st.pyplot(fig2)

else:
    freq_signal_names = list(st.session_state["freq-signals"].keys())
    if not freq_signal_names:
        st.sidebar.write("No frequency signals available for IFFT.")
        chosen = None
    else:
        chosen = st.sidebar.selectbox(
            "Choose frequency signal (for IFFT)", [None] + freq_signal_names
        )

    if chosen:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.header(f"Time reconstruction (IFFT) of: {chosen}")
        with col2:
            if st.button("Compute IFFT", key=f"ifft-{chosen}"):
                try:
                    freq_sig = read_signal(st.session_state["freq-signals"][chosen])

                    amps, phases = freq_sig.split()
                    amps = np.array(amps, dtype=float)
                    phases = np.array(phases, dtype=float)

                    time_sig = FFT_IFFT(freq_sig, inverse=1)

                    t_idx = time_sig.matrix[:, 0].astype(float)
                    a_recon = time_sig.matrix[:, 1].astype(float)
                    t_seconds = t_idx / fs

                    st.session_state["last-ifft"] = {
                        "name": chosen,
                        "fs": fs,
                        "t": t_seconds.tolist(),
                        "a": a_recon.tolist(),
                    }
                    st.success(f"IFFT computed for {chosen} (N={len(a_recon)})")
                except Exception as e:
                    import traceback

                    st.error(f"IFFT computation failed: {e}")
                    st.text(traceback.format_exc())

    if (
        st.session_state.get("last-ifft")
        and chosen
        and st.session_state["last-ifft"].get("name") == chosen
    ):
        data = st.session_state["last-ifft"]
        t_seconds = np.array(data["t"])
        a_recon = np.array(data["a"])

        st.subheader("Reconstructed Time Signal")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(t_seconds, a_recon, label="Reconstructed signal (IFFT)", linewidth=2)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Amplitude")
        ax3.set_title(f"Time Domain - Reconstructed Signal ({data.get('name', 'Signal')})")
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)

