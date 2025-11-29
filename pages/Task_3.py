import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    list_signals,
    quantization,
    read_signal,
)

st.set_page_config(page_title="Task 3", page_icon="📈", layout="wide")


(st.session_state["time-signals"], st.session_state["freq-signals"]) = list_signals("task3")

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
    for name in st.session_state["time-signals"].keys():
        checked = st.sidebar.checkbox(
            name,
            key=f"time-{name}",
            disabled=st.session_state["type"] is True,
        )
        st.session_state["selected-time"][name] = checked
    st.sidebar.divider()

if len(st.session_state["freq-signals"]):
    st.sidebar.title("Frequency Signals")
    for name in st.session_state["freq-signals"].keys():
        checked = st.sidebar.checkbox(
            name,
            key=f"freq-{name}",
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

for name, checked in st.session_state["selected-time"].items():
    if current_type == True:
        st.session_state["selected-time"][name] = False
    else:
        st.session_state["selected-time"][name] = checked

for name, checked in st.session_state["selected-freq"].items():
    if current_type == False:
        st.session_state["selected-freq"][name] = False
    else:
        st.session_state["selected-freq"][name] = checked

option = st.sidebar.radio("Quantization Levels (bits)", ["Levels", "Bits"])
inp = st.sidebar.number_input(option, min_value=1, max_value=100)
lvls = 0
if option == "Levels":
    lvls = inp
else:
    lvls = 2**inp

time_selected = any(st.session_state["selected-time"].values())
freq_selected = any(st.session_state["selected-freq"].values())
selected_signals = time_selected or freq_selected

if selected_signals:
    for name, selected in st.session_state["selected-time"].items():
        if selected:
            signal = read_signal(st.session_state["time-signals"][name])
            signal.draw(ax, type=0, label=name, is_cont=True)
            lvls_num, lvls_encoded, quantized, err = quantization(lvls, signal, plt=ax)
            quantized.draw(ax, type=0, label=f"Quantized {name}", is_cont=False)

    for name, selected in st.session_state["selected-freq"].items():
        if selected:
            signal = read_signal(st.session_state["freq-signals"][name])
            signal.draw(ax, type=0, label=name, is_cont=True)
            lvls_num, lvls_encoded, quantized, err = quantization(lvls, signal)
            quantized.draw(ax, type=0, label=f"Quantized {name}", is_cont=False)
else:
    st.session_state["type"] = None

time_selected = any(st.session_state["selected-time"].values())
freq_selected = any(st.session_state["selected-freq"].values())
if time_selected:
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Amplitude (A)")
    ax.legend()
st.pyplot(fig)
