import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import accumulation, add_all, list_signals, mul, normalize, read_signal, sin_wave, sin_wave_equation, square, sub_all, time_plot

st.set_page_config(page_title="DSP Project", page_icon="📡", layout="wide")

st.title("DSP Project Dashboard")
st.write("""
Welcome to the DSP Project!  
Use the sidebar on the left to navigate between tasks.
""")
