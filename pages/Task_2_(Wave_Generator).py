import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import accumulation, add_all, list_signals, mul, normalize, read_signal, sin_wave, sin_wave_equation, square, sub_all, time_plot

st.set_page_config(page_title="Task 2 (Wave Generator)",page_icon='🌊', layout="wide")

fig, ax = plt.subplots()

A = st.sidebar.slider('Amplitude', min_value=0, max_value=20, value= 10)
f = st.sidebar.slider('Frequency', min_value=1, max_value=10, value=1)
deg = st.sidebar.slider('Phase (in degrees)', min_value=0, max_value=360, value=0)
theta = np.deg2rad(deg)
B = st.sidebar.slider('Bias', min_value=-20, max_value=20, value=0)
signal_type = st.sidebar.radio('Choose Signal Type', ["Sine", "Cosine"])
t = np.linspace(0, 2, 500)
fs = st.sidebar.number_input('Frequency Sampling', min_value=2, max_value=100, value=2*f)
signal = 0.0
eq = ''
if signal_type == "Sine":
    signal = sin_wave(A, f, t, theta, B, True)
    eq = sin_wave_equation(A, f, t, theta, B, True)
else:
    signal = sin_wave(A, f, t, theta, B, False)
    eq = sin_wave_equation(A, f, t, theta, B, False)
ax.plot(t, signal)
ax.set_xlabel('Time (t)')
ax.set_ylabel('Amplitude (A)')
if fs >= 2 * f:
    n = np.arange(0, 2, 1/fs)
    if signal_type == "Sine":
        signal_discrete = sin_wave(A, f, n, theta, B, True)
    else:
        signal_discrete = sin_wave(A, f, n, theta, B, False)
    ax.stem(n, signal_discrete)
st.pyplot(fig)