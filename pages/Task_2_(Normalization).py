import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import accumulation, list_signals, normalize, read_signal, square, sub_all

st.set_page_config(page_title="Task 2 (Normalization)", page_icon="📶", layout="wide")


(st.session_state["time-signals"], st.session_state["freq-signals"]) = list_signals(2)

if "selected-time" not in st.session_state:
    st.session_state["selected-time"] = {}
    for signal in st.session_state["time-signals"].keys():
        st.session_state["selected-time"][signal] = False

if "selected-freq" not in st.session_state:
    st.session_state["selected-freq"] = {}
    for signal in st.session_state["freq-signals"].keys():
        st.session_state["selected-freq"][signal] = False

if "type" not in st.session_state:
    st.session_state["type"] = None

if "cont" not in st.session_state:
    st.session_state["cont"] = False

if "norm-1" not in st.session_state:
    st.session_state["norm-1"] = False
if "norm-2" not in st.session_state:
    st.session_state["norm-2"] = False

fig, ax = plt.subplots()

if st.session_state["type"] is True:
    for name in st.session_state["selected-time"].keys():
        st.session_state["selected-time"][name] = False
        st.session_state[f"time-{name}"] = False
elif st.session_state["type"] is False:
    for name in st.session_state["selected-freq"].keys():
        st.session_state["selected-freq"][name] = False
        st.session_state[f"freq-{name}"] = False

if len(st.session_state["time-signals"]):
    st.sidebar.title("Time Signals")
    for signal in st.session_state["time-signals"].keys():
        checked = st.sidebar.checkbox(
            signal,
            key=f"time-{signal}",
            disabled=st.session_state["type"] is True,
        )
        st.session_state["selected-time"][signal] = checked
    st.sidebar.divider()

if len(st.session_state["freq-signals"]):
    st.sidebar.title("Frequency Signals")
    for signal in st.session_state["freq-signals"].keys():
        checked = st.sidebar.checkbox(
            signal,
            key=f"freq-{signal}",
            disabled=st.session_state["type"] is False,
        )
        st.session_state["selected-freq"][signal] = checked
    st.sidebar.divider()

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
norm = st.sidebar.radio(
    "Normalization",
    options=["None", "Normalize (0 to 1)", "Normalize (-1 to 1)"],
)
if norm == "Normalize (0 to 1)":
    st.session_state["norm-1"] = True
    st.session_state["norm-2"] = False

elif norm == "Normalize (-1 to 1)":
    st.session_state["norm-1"] = False
    st.session_state["norm-2"] = True

elif norm == "None":
    st.session_state["norm-1"] = False
    st.session_state["norm-2"] = False

option = st.sidebar.radio(
    "Operations", ["None", "Subtract", "Squaring", "Accumulation"], index=0, key="op"
)

if option == "Subtract":
    signals = []
    label = ""
    for name, selected in st.session_state["selected-time"].items():
        if selected:
            signal = read_signal(st.session_state["time-signals"][name])
            signals.append(signal)
            label += f"{name}+"
    label = label[:-1]
    if len(signals) > 1:
        result = sub_all(signals)
        if st.session_state["norm-1"]:
            result = normalize(result, 0)
        elif st.session_state["norm-2"]:
            result = normalize(result, 1)
        result.draw(ax, type=0, label=label, is_cont=st.session_state["cont"])
elif option == "Squaring":
    if st.session_state["type"] == False:
        for name, selected in st.session_state["selected-time"].items():
            if selected:
                signal = read_signal(st.session_state["time-signals"][name])
                signal = square(signal)
                if st.session_state["norm-1"]:
                    signal = normalize(signal, 0)
                elif st.session_state["norm-2"]:
                    signal = normalize(signal, 1)
                label = f"{name}^2"
                signal.draw(ax, type=0, label=label, is_cont=st.session_state["cont"])
elif option == "Accumulation":
    if st.session_state["type"] == False:
        for name, selected in st.session_state["selected-time"].items():
            if selected:
                signal = read_signal(st.session_state["time-signals"][name])
                signal = accumulation(signal)
                if st.session_state["norm-1"]:
                    signal = normalize(signal, 0)
                elif st.session_state["norm-2"]:
                    signal = normalize(signal, 1)
                label = f"Σ {name}"
                signal.draw(ax, type=0, label=label, is_cont=st.session_state["cont"])

time_selected = any(st.session_state["selected-time"].values())
freq_selected = any(st.session_state["selected-freq"].values())
selected_signals = time_selected or freq_selected

if selected_signals:
    for name, selected in st.session_state["selected-time"].items():
        if selected:
            signal = read_signal(st.session_state["time-signals"][name])
            if st.session_state["norm-1"]:
                signal = normalize(signal, 0)
            elif st.session_state["norm-2"]:
                signal = normalize(signal, 1)
            signal.draw(ax, type=0, label=name, is_cont=st.session_state["cont"])

    for name, selected in st.session_state["selected-freq"].items():
        if selected:
            signal = read_signal(st.session_state["freq-signals"][name])
            if st.session_state["norm-1"]:
                signal = normalize(signal, 0)
            elif st.session_state["norm-2"]:
                signal = normalize(signal, 1)
            signal.draw(ax, type=0, label=name, is_cont=st.session_state["cont"])
else:
    st.session_state["type"] = None

time_selected = any(st.session_state["selected-time"].values())
freq_selected = any(st.session_state["selected-freq"].values())
if time_selected:
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Amplitude (A)")
    ax.legend()
st.pyplot(fig)
