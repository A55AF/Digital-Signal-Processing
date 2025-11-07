# Digital Signal Processing (DSP) — Python

Beginner-friendly repository for small digital signal processing utilities.

## Overview

This repository contains a collection of small DSP utilities and exercises implemented in Python. It is structured as a set of task scripts (Task_1..Task_4), helper modules, example input signals, and simple tests. A Streamlit-based GUI launcher is provided in `Home.py` so you can explore the functionality interactively through a browser.

Use this repo to learn and experiment with basic DSP building blocks: signal arithmetic, normalization, waveform generation, quantization, DFT/IDFT, DC removal, and small test harnesses for each task.

## Key features

- Multiply, add and combine signals
- Normalization utilities (range transforms)
- Wave generator (sine/cosine helpers)
- Signal quantization examples (uniform quantization)
- DFT & IDFT helper scripts and example inputs/outputs
- DC component removal
- Streamlit GUI entrypoint (`Home.py`) to run the demo interactively

## Prerequisites

- Python 3.8+ (3.10 or 3.11 recommended)
- Git
- `requirements.txt` is included for installing dependencies used by the tasks and the GUI (Streamlit and any other libs listed there).

## Quickstart — clone, venv, install, run GUI

1. Clone the repository

```bash
git clone https://github.com/<your-username>/Digital-Signal-Processing.git
cd Digital-Signal-Processing
```

2. Create and activate a virtual environment

    - (Linux/macOS)

    ```bash
    python3 -m .venv venv
    source .venv/bin/activate
    ```

    - On Windows (PowerShell):

    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Launch the Streamlit GUI

```bash
streamlit run Home.py
```

This will open a browser window (or show a local URL) where you can interact with the available tasks and visualizations.

## Project layout (high level)

- `Home.py` — Streamlit GUI entry point
- `signal_framework.py` — the main class of signals
- `utils.py` — utility functions (normalize, add, DFT, etc...)
- `pages/` — GUI seperated by task number with `Home.py` as GUI main entry
- `signals/` — input signal examples for each task
- `requirements.txt` — dependencies


## License

This project does not currently include a license file. To make the code usable by others, add a `LICENSE` (for example, MIT) in the repository root.

## Contact / Questions

If you want help extending this project (for example, adding more DFT visualizations, spectrograms, or a packaging step), open an issue describing what you want and I can help implement it.
