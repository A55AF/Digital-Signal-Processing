import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import (
    list_signals,
    read_signal,
    moving_average,
    first_derivative,
    second_derivative,
    delay_advance,
    fold_signal,
    fold_and_shift,
    remove_dc_time,
    convolve_signals,
    normalized_cross_correlation,
    normalized_autocorrelation,
    periodic_cross_correlation,
    time_delay_analysis,
)

st.set_page_config(page_title="Task 6 - Time Domain", page_icon="⏱️", layout="wide")

def _get_signals_for_operation(operation):
    """Return dict of signals relevant to the operation."""
    op_to_folders = {
        "Smoothing (Moving Average)": [],
        "Sharpening (1st derivative)": [],
        "Sharpening (2nd derivative)": [],
        "Delay / Advance": [],
        "Fold Signal": [],
        "Fold then Delay/Advance": [],
        "Remove DC (time)": ["remove_dc"],
        "Convolution (two signals)": ["convolution"],
        "Normalized Cross-correlation": ["correlation/point1"],
        "Normalized Autocorrelation": ["correlation/point1", "correlation/different_length"],
        "Periodic Cross-correlation": ["correlation/point1", "correlation/different_length"],
        "Time Delay Analysis (periodic)": ["correlation/point1"],
    }

    time_signals = {}
    freq_signals = {}
    candidates = op_to_folders.get(operation, ["general"])
    # Always include general folder for all operations
    if "general" not in candidates:
        candidates.append("general")
    
    for cand in candidates:
        try:
            t_signals, f_signals = list_signals(os.path.join("task6", cand))
            time_signals.update(t_signals)
            freq_signals.update(f_signals)
        except Exception:
            pass

    return time_signals, freq_signals


def list_signals_selection():
    if len(operation_time_signals):
        st.sidebar.title("Time Signals")
        for name in operation_time_signals.keys():
            checked = st.sidebar.checkbox(
                name,
                key=f"time-{name}",
                disabled=st.session_state["type"] is True,
            )
            st.session_state["selected-time"][name] = checked
        st.sidebar.divider()

    if len(operation_freq_signals):
        st.sidebar.title("Frequency Signals")
        for name in operation_freq_signals.keys():
            checked = st.sidebar.checkbox(
                name,
                key=f"freq-{name}",
                disabled=st.session_state["type"] is False,
            )
            st.session_state["selected-freq"][name] = checked
        st.sidebar.divider()


if "selected-time" not in st.session_state:
    st.session_state["selected-time"] = {}
if "selected-freq" not in st.session_state:
    st.session_state["selected-freq"] = {}


task_path = "task6"
signals_path = os.listdir(os.path.join("signals", task_path))
for i in range(0,len(signals_path)):
    dr = os.path.join(task_path, signals_path[i])
    t, f = list_signals(dr)
    for signal_name in t.values():
        st.session_state["selected-time"][signal_name] = False
    for signal_name in f.values():
        st.session_state["selected-freq"][signal_name] = False

if "type" not in st.session_state:
    st.session_state["type"] = None

if "cont" not in st.session_state:
    st.session_state["cont"] = False

fig, ax = plt.subplots(figsize=(12, 6))
ax.grid(True, alpha=0.3)

if st.session_state["type"] is True:
    for name in st.session_state["selected-time"].keys():
        st.session_state["selected-time"][name] = False
        st.session_state[f"time-{name}"] = False
elif st.session_state["type"] is False:
    for name in st.session_state["selected-freq"].keys():
        st.session_state["selected-freq"][name] = False
        st.session_state[f"freq-{name}"] = False

current_type = None

if any(st.session_state["selected-time"].values()):
    current_type = False
elif any(st.session_state["selected-freq"].values()):
    current_type = True

if st.session_state["type"] != current_type:
    st.session_state["type"] = current_type

for signal, checked in st.session_state["selected-time"].items():
    if current_type == True:
        st.session_state["selected-time"][signal] = False
    else:
        st.session_state["selected-time"][signal] = checked

for signal, checked in st.session_state["selected-freq"].items():
    if current_type == False:
        st.session_state["selected-freq"][signal] = False
    else:
        st.session_state["selected-freq"][signal] = checked

st.sidebar.checkbox("Continuous", key="cont")

# Operations
option = st.sidebar.radio(
    "Operations",
    [
        "None",
        "Smoothing (Moving Average)",
        "Sharpening (1st derivative)",
        "Sharpening (2nd derivative)",
        "Delay / Advance",
        "Fold Signal",
        "Fold then Delay/Advance",
        "Remove DC (time)",
        "Convolution (two signals)",
        "Normalized Cross-correlation",
        "Normalized Autocorrelation",
        "Periodic Cross-correlation",
        "Time Delay Analysis (periodic)",
    ],
    index=0,
    key="op",
)

operation_time_signals, operation_freq_signals = _get_signals_for_operation(option) if option != "None" else ({},{})


time_selected = any(st.session_state["selected-time"].values())
freq_selected = any(st.session_state["selected-freq"].values())
selected_signals = time_selected or freq_selected

# Process operations
if option == "Smoothing (Moving Average)":
    list_signals_selection()
    window = st.sidebar.number_input(
        "Window size (points)", value=3, min_value=1, step=1
    )
    op_selected = st.session_state["selected-time"]
    for name, selected in op_selected.items():
        if selected:
            try:
                sig_path = operation_time_signals[name]
                if sig_path:
                    signal = read_signal(sig_path)
                    if signal is not None:
                        res = moving_average(signal, int(window))
                        if res is not None:
                            res.draw(
                                ax,
                                label=f"MA({window}) - {name}",
                                is_cont=st.session_state["cont"],
                                scat=False,
                            )
            except Exception as e:
                st.sidebar.error(f"Error processing {name}: {str(e)}")

elif option == "Sharpening (1st derivative)":
    list_signals_selection()
    op_selected = st.session_state["selected-time"]
    for name, selected in op_selected.items():
        if selected:
            try:
                sig_path = operation_time_signals[name]
                if sig_path:
                    signal = read_signal(sig_path)
                    if signal is not None:
                        res = first_derivative(signal)
                        if res is not None:
                            res.draw(
                                ax,
                                label=f"1st Derivative - {name}",
                                is_cont=st.session_state["cont"],
                                scat=False,
                            )
            except Exception as e:
                st.sidebar.error(f"Error processing {name}: {str(e)}")

elif option == "Sharpening (2nd derivative)":
    list_signals_selection()
    op_selected = st.session_state["selected-time"]
    for name, selected in op_selected.items():
        if selected:
            try:
                sig_path = operation_time_signals[name]
                if sig_path:
                    signal = read_signal(sig_path)
                    if signal is not None:
                        res = second_derivative(signal)
                        if res is not None:
                            res.draw(
                                ax,
                                label=f"2nd Derivative - {name}",
                                is_cont=st.session_state["cont"],
                                scat=False,
                            )
            except Exception as e:
                st.sidebar.error(f"Error processing {name}: {str(e)}")

elif option == "Delay / Advance":
    list_signals_selection()
    k = st.sidebar.number_input(
        "k (number of steps, + delay, - advance)", value=0, step=1
    )
    op_selected = st.session_state["selected-time"]
    for name, selected in op_selected.items():
        if selected:
            try:
                sig_path = operation_time_signals[name]
                if sig_path:
                    signal = read_signal(sig_path)
                    if signal is not None:
                        res = delay_advance(signal, int(k))
                        if res is not None:
                            res.draw(
                                ax,
                                label=f"Delay/Advance ({k}) - {name}",
                                is_cont=st.session_state["cont"],
                                scat=False,
                            )
            except Exception as e:
                st.sidebar.error(f"Error processing {name}: {str(e)}")

elif option == "Fold Signal":
    list_signals_selection()
    op_selected = st.session_state["selected-time"]
    for name, selected in op_selected.items():
        if selected:
            try:
                sig_path = operation_time_signals[name]
                if sig_path:
                    signal = read_signal(sig_path)
                    if signal is not None:
                        res = fold_signal(signal)
                        if res is not None:
                            res.draw(
                                ax,
                                label=f"Folded - {name}",
                                is_cont=st.session_state["cont"],
                                scat=False,
                            )
            except Exception as e:
                st.sidebar.error(f"Error processing {name}: {str(e)}")

elif option == "Fold then Delay/Advance":
    list_signals_selection()
    k = st.sidebar.number_input("k (steps)", value=0, step=1)
    op_selected = st.session_state["selected-time"]
    for name, selected in op_selected.items():
        if selected:
            try:
                sig_path = operation_time_signals[name]
                if sig_path:
                    signal = read_signal(sig_path)
                    if signal is not None:
                        res = fold_and_shift(signal, int(k))
                        if res is not None:
                            res.draw(
                                ax,
                                label=f"Folded+Shift ({k}) - {name}",
                                is_cont=st.session_state["cont"],
                                scat=False,
                            )
            except Exception as e:
                st.sidebar.error(f"Error processing {name}: {str(e)}")

elif option == "Remove DC (time)":
    list_signals_selection()
    op_selected = st.session_state["selected-time"]
    for name, selected in op_selected.items():
        if selected:
            try:
                sig_path = operation_time_signals[name]
                if sig_path:
                    signal = read_signal(sig_path)
                    if signal is not None:
                        res = remove_dc_time(signal)
                        if res is not None:
                            res.draw(
                                ax,
                                label=f"DC Removed - {name}",
                                is_cont=st.session_state["cont"],
                                scat=False,
                            )
            except Exception as e:
                st.sidebar.error(f"Error processing {name}: {str(e)}")

elif option == "Convolution (two signals)":
    op_selected_list = [
        name
        for name, path in operation_time_signals.items()
    ]
    if len(op_selected_list) >= 2:
        sig1_name = st.sidebar.selectbox("Signal 1", op_selected_list, key="conv_sig1")
        sig2_name = st.sidebar.selectbox(
            "Signal 2",
            [s for s in op_selected_list if s != sig1_name],
            key="conv_sig2",
        )
        if sig1_name and sig2_name:
            try:
                sig1_path = operation_time_signals[sig1_name]
                sig2_path = operation_time_signals[sig2_name]
                if sig1_path and sig2_path:
                    s1 = read_signal(sig1_path)
                    s2 = read_signal(sig2_path)
                    if s1 is not None and s2 is not None:
                        res = convolve_signals(s1, s2)
                        if res is not None:
                            res.draw(
                                ax,
                                label=f"Convolution ({sig1_name} * {sig2_name})",
                                is_cont=st.session_state["cont"],
                                scat=False,
                            )
            except Exception as e:
                st.sidebar.error(f"Error in convolution: {str(e)}")
    else:
        st.sidebar.info("Please select at least 2 signals for convolution")

elif option == "Normalized Cross-correlation":
    op_selected_list = [
        name
        for name, path in operation_time_signals.items()
    ]
    if len(op_selected_list) >= 2:
        sig1_name = st.sidebar.selectbox("Signal 1", op_selected_list, key="conv_sig1")
        sig2_name = st.sidebar.selectbox(
            "Signal 2",
            [s for s in op_selected_list if s != sig1_name],
            key="conv_sig2",
        )
        if sig1_name and sig2_name:
            try:
                sig1_path = operation_time_signals[sig1_name]
                sig2_path = operation_time_signals[sig2_name]
                if sig1_path and sig2_path:
                    s1 = read_signal(sig1_path)
                    s2 = read_signal(sig2_path)
                    if s1 is not None and s2 is not None:
                        corr = normalized_cross_correlation(s1, s2)
                        if corr is not None:
                            lags = np.arange(
                                -len(s1.matrix[:, 1]) + 1, len(s2.matrix[:, 1])
                            )
                            ax.plot(
                                lags,
                                corr,
                                label=f"NCC ({sig1_name}, {sig2_name})",
                            )
                        else:
                            ac = normalized_autocorrelation(s1)
                            if ac is not None:
                                lags = np.arange(
                                    -len(s1.matrix[:, 1]) + 1, len(s1.matrix[:, 1])
                                )
                                ax.plot(lags, ac, label=f"Autocorr ({sig1_name})")
            except Exception as e:
                st.sidebar.error(f"Error in correlation: {str(e)}")

elif option == "Normalized Autocorrelation":
    list_signals_selection()
    op_selected = {}
    for name in operation_time_signals.keys():
        op_selected[name] = st.session_state["selected-time"][name]
    print(op_selected)
    for name, selected in op_selected.items():
        if selected:
            try:
                sig_path = operation_time_signals[name]
                if sig_path:
                    signal = read_signal(sig_path)
                    if signal is not None:
                        ac = normalized_autocorrelation(signal)
                        if ac is not None:
                            lags = np.arange(
                                -len(signal.matrix[:, 1]) + 1,
                                len(signal.matrix[:, 1]),
                            )
                            ax.plot(lags, ac, label=f"Autocorr ({name})")
            except Exception as e:
                st.sidebar.error(f"Error processing {name}: {str(e)}")

elif option == "Periodic Cross-correlation":
    op_selected_list = [
        name
        for name, path in operation_time_signals.items()
    ]
    if len(op_selected_list) >= 2:
        sig1_name = st.sidebar.selectbox("Signal 1", op_selected_list, key="pcorr_sig1")
        sig2_name = st.sidebar.selectbox(
            "Signal 2",
            [s for s in op_selected_list if s != sig1_name],
            key="pcorr_sig2",
        )
        if sig1_name and sig2_name:
            try:
                sig1_path = operation_time_signals[sig1_name]
                sig2_path = operation_time_signals[sig2_name]
                if sig1_path and sig2_path:
                    s1 = read_signal(sig1_path)
                    s2 = read_signal(sig2_path)
                    if s1 is not None and s2 is not None:
                        corr = periodic_cross_correlation(s1, s2)
                        if corr is not None:
                            lags = np.arange(len(corr))
                            ax.plot(
                                lags,
                                corr,
                                label=f"Periodic CC ({sig1_name}, {sig2_name})",
                            )
            except Exception as e:
                st.sidebar.error(f"Error in periodic correlation: {str(e)}")
    else:
        st.sidebar.info("Please select at least 2 signals for periodic correlation")

elif option == "Time Delay Analysis (periodic)":
    op_selected_list = [
        name
        for name, path in operation_time_signals.items()
    ]
    if len(op_selected_list) >= 2:
        sig1_name = st.sidebar.selectbox("Periodic Signal 1", op_selected_list, key="delay_sig1")
        sig2_name = st.sidebar.selectbox(
            "Periodic Signal 2",
            [s for s in op_selected_list if s != sig1_name],
            key="delay_sig2",
        )
        Ts = st.sidebar.number_input(
            "Sampling period Ts (seconds)", value=1.0, min_value=1e-9
        )
        if sig1_name and sig2_name:
            try:
                sig1_path = operation_time_signals[sig1_name]
                sig2_path = operation_time_signals[sig2_name]
                if sig1_path and sig2_path:
                    s1 = read_signal(sig1_path)
                    s2 = read_signal(sig2_path)
                    if s1 is not None and s2 is not None:
                        res = time_delay_analysis(s1, s2, Ts)
                        if res is not None:
                            delay_seconds, lag, corr = res
                            st.sidebar.success(
                                f"Estimated delay: {delay_seconds:.4f} seconds (lag = {lag} samples)"
                            )
                            lags = np.arange(len(corr))
                            ax.plot(
                                lags,
                                corr,
                                label=f"Time Delay Corr ({sig1_name}, {sig2_name})",
                            )
            except Exception as e:
                st.sidebar.error(f"Error in time delay analysis: {str(e)}")
    else:
        st.sidebar.info("Please select at least 2 signals for time delay analysis")

# Auto-scale axes to fit the data
ax.relim()
ax.autoscale()

time_selected = any(st.session_state["selected-time"].values())
freq_selected = any(st.session_state["selected-freq"].values())

if time_selected or freq_selected:
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Amplitude (A)")
    ax.legend()

st.pyplot(fig)