import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    add_all,
    list_signals,
    mul,
    read_signal,
)

st.set_page_config(page_title="Task 1",page_icon='📈' , layout="wide")


(st.session_state["time-signals"], st.session_state["freq-signals"]) = list_signals(1)

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
        st.session_state["selected-freq"][name] = checked
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

option = st.sidebar.radio("Operations", ["None", "Add", "Multiply"], index=0, key="op")
if option == "Add":
    signals = []
    label = ""
    for name, selected in st.session_state["selected-time"].items():
        if selected:
            signal = read_signal(st.session_state["time-signals"][name])
            signals.append(signal)
            label += f"{name}+"
    label = label[:-1]
    if len(signals) > 1:
        result = add_all(signals)
        result.draw(ax, label=label, is_cont=st.session_state["cont"])
elif option == "Multiply":
    constant = st.sidebar.number_input(
        "Multiply Constant",
        value=1,
        min_value=-10000,
        max_value=10000,
        step=1,
    )
    if st.session_state["type"] == False:
        for name, selected in st.session_state["selected-time"].items():
            if selected:
                signal = read_signal(st.session_state["time-signals"][name])
                signal = mul(signal, constant)
                label = f"{name}*{constant}"
                signal.draw(ax, label=label, is_cont=st.session_state["cont"])

time_selected = any(st.session_state["selected-time"].values())
freq_selected = any(st.session_state["selected-freq"].values())
selected_signals = time_selected or freq_selected

if selected_signals:
    for name, selected in st.session_state["selected-time"].items():
        if selected:
            signal = read_signal(st.session_state["time-signals"][name])
            signal.draw(ax, label=name, is_cont=st.session_state["cont"])

    for name, selected in st.session_state["selected-freq"].items():
        if selected:
            signal = read_signal(st.session_state["freq-signals"][name])
            signal.draw(ax, label=name, is_cont=st.session_state["cont"])
else:
    st.session_state["type"] = None

time_selected = any(st.session_state["selected-time"].values())
freq_selected = any(st.session_state["selected-freq"].values())
if time_selected:
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Amplitude (A)")
    ax.legend()
st.pyplot(fig)
