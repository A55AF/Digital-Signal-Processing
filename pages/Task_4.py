import cmath
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
from utils import list_signals, normalize, read_signal, dft, remove_dc_component
from signal_framework import Signal

st.set_page_config(page_title="Task 4", page_icon="📈", layout="wide")

(st.session_state["time-signals"], st.session_state["freq-signals"]) = list_signals(4)


def ensure_state_keys():
    if "viewer-selected" not in st.session_state:
        st.session_state["viewer-selected"] = {
            s: False for s in st.session_state["time-signals"].keys()
        }
    if "last-dft" not in st.session_state:
        st.session_state["last-dft"] = None


ensure_state_keys()

source_type = st.sidebar.radio(
    "Source type", ["Time-domain (DFT)", "Frequency-domain (IDFT)"], index=0
)

fs = st.sidebar.number_input(
    "Enter sampling frequency (Hz)", min_value=0.0001, value=1.0, format="%.4f"
)

if source_type.startswith("Time"):
    time_signal_names = list(st.session_state["time-signals"].keys())
    if not time_signal_names:
        st.sidebar.write("No time signals available for DFT.")
        chosen = None
    else:
        chosen = st.sidebar.selectbox(
            "Choose time signal (for DFT)", [None] + time_signal_names
        )

        if chosen:
            remove_dc = st.sidebar.checkbox(
                "Remove DC component (F(0))", key=f"remove-dc-{chosen}"
            )

    if chosen:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.header(f"Frequency analysis (DFT) of: {chosen}")
        with col2:
            if st.button("Compute DFT", key=f"dft-{chosen}"):
                try:
                    time_sig = read_signal(st.session_state["time-signals"][chosen])
                    x, a, *rest = time_sig.split()

                    freq_sig = dft(time_sig, inv=False)

                    N = len(freq_sig.matrix[:, 0])
                    freqs = np.arange(N) * fs / N

                    amps = freq_sig.matrix[:, 0].astype(float)
                    phases = freq_sig.matrix[:, 1].astype(float)

                    max_amp = np.max(amps) if np.max(amps) != 0 else 1.0
                    norm_amps = amps / max_amp

                    st.session_state["last-dft"] = {
                        "name": chosen,
                        "fs": fs,
                        "freqs": freqs.tolist(),
                        "amps": amps.tolist(),
                        "phases": phases.tolist(),
                        "orig_time_x": x.tolist(),
                        "orig_time_a": a.tolist(),
                        "source": "time",
                    }
                    st.success(f"DFT computed for {chosen} (N={N})")
                    st.write(f"freqs length={len(freqs)}, amps len={len(amps)}")
                except Exception as e:
                    import traceback

                    st.error(f"DFT computation failed: {e}")
                    st.text(traceback.format_exc())
else:
    freq_signal_names = list(st.session_state["freq-signals"].keys())
    if not freq_signal_names:
        st.sidebar.write("No frequency signals available for IDFT.")
        chosen = None
    else:
        chosen = st.sidebar.selectbox(
            "Choose frequency signal (for IDFT)", [None] + freq_signal_names
        )

    if chosen:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.header(f"Time reconstruction (IDFT) of: {chosen}")
        with col2:
            if st.button("Compute IDFT", key=f"idft-src-{chosen}"):
                try:
                    freq_sig = read_signal(st.session_state["freq-signals"][chosen])

                    amps, phases = freq_sig.split()
                    amps = np.array(amps, dtype=float)
                    phases = np.array(phases, dtype=float)

                    time_sig = dft(freq_sig, inv=True)

                    t_idx = time_sig.matrix[:, 0].astype(float)
                    a_recon = time_sig.matrix[:, 1].astype(float)
                    t_seconds = t_idx / fs

                    st.session_state["last-idft"] = {
                        "name": chosen,
                        "fs": fs,
                        "t": t_seconds.tolist(),
                        "a": a_recon.tolist(),
                        "source": "freq",
                    }
                    st.success(f"IDFT computed for {chosen} (N={len(a_recon)})")
                except Exception as e:
                    import traceback

                    st.error(f"IDFT computation failed: {e}")
                    st.text(traceback.format_exc())

if (
    st.session_state.get("last-dft")
    and st.session_state["last-dft"].get("name") == chosen
    and source_type.startswith("Time")
):
    data = st.session_state["last-dft"]
    freqs = np.array(data["freqs"])
    amps = np.array(data["amps"], dtype=float)
    phases = np.array(data["phases"], dtype=float)
    max_amp = np.max(amps) if np.max(amps) != 0 else 1.0
    norm_amps = amps / max_amp

    fig1, ax1 = plt.subplots()
    ax1.stem(freqs, norm_amps, basefmt=" ", label="All frequencies")

    dominant_mask = norm_amps > 0.5
    if np.any(dominant_mask):
        dominant_freqs = freqs[dominant_mask]
        dominant_amps = norm_amps[dominant_mask]
        ax1.stem(
            dominant_freqs,
            dominant_amps,
            basefmt=" ",
            linefmt="red",
            markerfmt="ro",
            label="Dominant (>0.5)",
        )

    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Normalized Amplitude")
    ax1.axhline(
        y=0.5, color="orange", linestyle="--", alpha=0.7, label="Threshold (0.5)"
    )
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # phase plot
    fig2, ax2 = plt.subplots()
    ax2.stem(freqs, phases, basefmt=" ")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (rad)")
    ax2.grid(True)
    st.pyplot(fig2)

    # dominant frequencies
    dominant_mask = norm_amps > 0.5
    dominant_freqs = freqs[dominant_mask]
    dominant_amps = norm_amps[dominant_mask]
    if dominant_freqs.size > 0:
        st.subheader("Dominant frequencies (normalized amplitude > 0.5)")
        for f, a_norm in zip(dominant_freqs, dominant_amps):
            st.write(f"{f:.4f} Hz — normalized amplitude: {a_norm:.4f}")
    else:
        st.info("No dominant frequencies (normalized amplitude > 0.5) found.")

    # component editor
    st.subheader("Modify components (amplitude & phase)")
    cols = st.columns([1, 2, 2, 2])
    cols[0].write("#")
    cols[1].write("Amplitude")
    cols[2].write("Phase (rad)")
    cols[3].write("Normalized Amp")

    new_amps = []
    new_phases = []
    max_slider_amp = max_amp * 2 if max_amp > 0 else 1.0
    for i, (f, a, p) in enumerate(zip(freqs, amps, phases)):
        c0, c1, c2, c3 = st.columns([0.6, 2.0, 2.0, 1.2])
        c0.write(f"k={i}\n{f:.2f}Hz")
        new_a = c1.slider(
            f"Amp {i}",
            min_value=0.0,
            max_value=float(max_slider_amp),
            value=float(a),
            key=f"amp-{chosen}-{i}",
        )
        new_p = c2.slider(
            f"Phase {i}",
            min_value=-np.pi,
            max_value=np.pi,
            value=float(p),
            key=f"phase-{chosen}-{i}",
        )
        c3.write(f"{(a/max_amp if max_amp!=0 else 0):.3f}")
        new_amps.append(new_a)
        new_phases.append(new_p)

    if st.button("Reconstruct signal (IDFT)", key=f"idft-{chosen}"):
        orig_matrix = np.column_stack(
            (np.array(amps, dtype=float), np.array(phases, dtype=float))
        )
        orig_freq_signal = Signal(True, False, orig_matrix)
        orig_time_recon = dft(orig_freq_signal, inv=True)
        _, orig_a_recon = orig_time_recon.split()
        orig_t_seconds = np.arange(len(orig_a_recon)) / data["fs"]

        mod_matrix = np.column_stack(
            (np.array(new_amps, dtype=float), np.array(new_phases, dtype=float))
        )
        freq_signal_for_idft = Signal(True, False, mod_matrix)

        if remove_dc:
            freq_signal_for_idft = remove_dc_component(freq_signal_for_idft)

        time_recon = dft(freq_signal_for_idft, inv=True)

        t_idx = time_recon.matrix[:, 0].astype(float)
        a_recon = time_recon.matrix[:, 1].astype(float)

        t_seconds = t_idx / data["fs"]

        orig_x = np.array(data["orig_time_x"], dtype=float)
        orig_a = np.array(data["orig_time_a"], dtype=float)

        if len(orig_a) != len(a_recon):
            N = len(a_recon)
            T = orig_x[-1]
            t_new = np.linspace(0, T, N)
            orig_a_rs = np.interp(t_new, orig_x, orig_a)
            t_plot = t_new
        else:
            t_plot = orig_x
            orig_a_rs = orig_a

        fig3, ax3 = plt.subplots()
        ax3.plot(t_plot, orig_a_rs, label="Original (time domain)", linewidth=2)
        ax3.plot(
            t_seconds,
            a_recon,
            label=f"Modified reconstruction{', DC removed' if remove_dc else ''}",
            linewidth=2,
        )

        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Amplitude")
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)

        st.session_state["last-reconstruction"] = {
            "t": t_seconds.tolist(),
            "a": a_recon.tolist(),
            "from": data.get("name"),
        }

elif (
    st.session_state.get("last-idft")
    and st.session_state["last-idft"].get("name") == chosen
    and source_type.startswith("Frequency")
):
    data = st.session_state["last-idft"]
    t_seconds = np.array(data["t"])
    a_recon = np.array(data["a"])

    fig3, ax3 = plt.subplots()
    ax3.plot(t_seconds, a_recon, label="Reconstructed (IDFT)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)
