import math
import numpy as np
from signal_framework import Signal
import os
import cmath


def read_signal(path):
    signal_type = False
    is_periodic = False
    matrix = np.empty((0, 2))

    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    signal_type = line.strip() == "1"
                    continue
                elif i == 1:
                    is_periodic = line.strip() == "1"
                    continue
                elif i == 2:
                    continue
                data = [float(d.rstrip("f")) for d in line.split()]
                matrix = np.append(matrix, [data], axis=0)
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            for i, line in enumerate(f):
                if i == 0:
                    signal_type = line.strip() == "1"
                    continue
                elif i == 1:
                    is_periodic = line.strip() == "1"
                    continue
                elif i == 2:
                    continue
                data = [float(d.rstrip("f")) for d in line.split()]
                matrix = np.append(matrix, [data], axis=0)

    signal = Signal(signal_type, is_periodic, matrix)
    return signal


def list_signals(relpath):
    path = os.path.join("signals", relpath)
    time_signals = {}
    freq_signals = {}
    for file_name in os.listdir(path):
        full_path = os.path.join(path, file_name)
        if not os.path.isfile(full_path):
            continue
        if not file_name.lower().endswith(".txt"):
            continue
        try:
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    signal_type = f.readline().strip() == "1"
            except UnicodeDecodeError:
                with open(full_path, "r", encoding="latin-1") as f:
                    signal_type = f.readline().strip() == "1"
            name = file_name.split(".")[0]
            if signal_type == False:
                time_signals[name] = full_path
            else:
                freq_signals[name] = full_path
        except Exception:
            continue
    return time_signals, freq_signals


# Task 1
def add(signal1, signal2):
    if signal1.signal_type == signal2.signal_type and signal1.signal_type == False:
        (x1, a1) = signal1.split()
        (x2, a2) = signal2.split()

        T = min(x1[-1], x2[-1])

        N = max(len(a1), len(a2))

        t = np.linspace(0, T, N)

        signal1_resampled = np.interp(t, x1, a1)
        signal2_resampled = np.interp(t, x2, a2)

        a = signal1_resampled + signal2_resampled
        result_matrix = np.column_stack((t, a))

        result_signal = Signal(signal1.signal_type, signal1.is_periodic, result_matrix)
        return result_signal


# Task 2
def sub(signal1, signal2):
    if signal1.signal_type == signal2.signal_type and signal1.signal_type == False:
        (x1, a1) = signal1.split()
        (x2, a2) = signal2.split()

        T = min(x1[-1], x2[-1])

        N = max(len(a1), len(a2))

        t = np.linspace(0, T, N)

        signal1_resampled = np.interp(t, x1, a1)
        signal2_resampled = np.interp(t, x2, a2)

        a = signal1_resampled - signal2_resampled
        result_matrix = np.column_stack((t, a))

        result_signal = Signal(signal1.signal_type, signal1.is_periodic, result_matrix)
        return result_signal


# Task 1
def add_all(signals):
    result = signals[0]
    for i in range(1, len(signals)):
        result = add(result, signals[i])
    return result


# Task 2
def sub_all(signals):
    result = signals[0]
    for i in range(1, len(signals)):
        result = sub(result, signals[i])
    return result


# Task 1
def mul(signal, const):
    if signal.signal_type == False:
        result = Signal(signal.signal_type, signal.is_periodic, signal.matrix)
        result.matrix[:, 1] = result.matrix[:, 1] * const
        return result


# Task 2
def square(signal):
    if signal.signal_type == False:
        result = Signal(signal.signal_type, signal.is_periodic, signal.matrix)
        result.matrix[:, 1] = result.matrix[:, 1] * result.matrix[:, 1]
        return result


# Task 2
def accumulation(signal):
    if signal.signal_type == False:
        result = Signal(signal.signal_type, signal.is_periodic, signal.matrix)
        for i, value in enumerate(result.matrix[:, 1]):
            if i > 0:
                result.matrix[i, 1] += result.matrix[i - 1, 1]
        return result


# Testing
def time_plot(plt):
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()


# Testing
def freq_plot(plt):
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()


# Task 2
def sin_wave(A, f, t, theta, B, is_sin=True):
    signal = 0.0
    if is_sin:
        signal = A * np.sin(2 * np.pi * f * t + theta) + B
    else:
        signal = A * np.cos(2 * np.pi * f * t + theta) + B
    return signal


# Task 2
def sin_wave_equation(A, f, t, theta, B, is_sin=True):
    eq = ""
    if is_sin:
        eq = f"{A} * sin( 2π * {f} * {t} + {theta}) + {B}"
    else:
        eq = f"{A} * cos( 2π * {f} * {t} + {theta}) + {B}"
    return eq


# 0 => 0_to_1
# 1 => -1_to_1
# Task 2
def normalize(signal, range_type=0):
    t = None
    a = None
    p = None
    if signal.signal_type == False:
        t, a = signal.split()
    else:
        a, p = signal.split()
    result = Signal(signal.signal_type, signal.is_periodic, signal.matrix)
    if range_type == 0:
        # [0, 1] normalization
        signal_min = np.min(a)
        signal_max = np.max(a)
        if signal_max == signal_min:
            a_normalized = np.full_like(a, 0.5)
        else:
            a_normalized = (a - signal_min) / (signal_max - signal_min)

    elif range_type == 1:
        # [-1, 1] normalization
        signal_min = np.min(a)
        signal_max = np.max(a)
        if signal_max == signal_min:
            a_normalized = np.zeros_like(a)
        else:
            a_normalized = 2 * (a - signal_min) / (signal_max - signal_min) - 1

    if signal.signal_type == False:
        result.matrix = np.column_stack((t, a_normalized))
    else:
        result.matrix = np.column_stack((a_normalized, p))
    return result


def itob(num, bits):
    result = ""
    while num > 0:
        result = str(num % 2) + result
        num //= 2
    shift = "0" * int(bits - len(result))
    result = shift + result
    if len(result) > bits:
        return None
    return result


def quantization(lvls_num, signal, plt=None):
    x_min = np.min(signal.matrix[:, 1])
    x_max = np.max(signal.matrix[:, 1])
    delta = (x_max - x_min) / lvls_num

    result = Signal(signal.signal_type, signal.is_periodic, signal.matrix.copy())
    sz = len(signal.matrix[:, 1])
    bits = int(np.log2(lvls_num))
    lvls_encoded = np.empty(sz, dtype=f"U{bits}")
    lvls = np.empty(sz, dtype=int)
    err = np.empty(sz)
    for i, amp in enumerate(result.matrix[:, 1]):
        interval_index = (amp - x_min) / delta

        if interval_index >= lvls_num:
            interval_index = lvls_num - 1e-10

        cur_lvl = int(interval_index)
        lvls_encoded[i] = itob(cur_lvl, bits)
        lvls[i] = cur_lvl + 1
        low = x_min + cur_lvl * delta
        high = x_min + (cur_lvl + 1) * delta
        mid = (low + high) / 2
        err[i] = mid - result.matrix[i, 1]
        result.matrix[i, 1] = mid

    if plt is not None:
        for lvl in range(0, lvls_num):
            line = x_min + (lvl) * delta
            plt.axhline(y=line, color="grey")
    return lvls, lvls_encoded, result, err


def dft(signal, inv=False):
    N = len(signal.matrix[:, 0])
    result = None

    if inv == True:
        t = range(N)
        a = []

        for n in range(N):
            real = 0.0
            imag = 0.0

            for k in range(N):
                angle = 2 * math.pi * n * k / N + signal.matrix[k, 1]

                real += signal.matrix[k, 0] * math.cos(angle)
                imag += signal.matrix[k, 0] * math.sin(angle)

            real /= N
            a.append(real)

        result_matrix = np.column_stack((t, a))
        result = Signal(False, signal.is_periodic, result_matrix)

    else:
        a = []
        p = []
        for k in range(N):
            real = 0.0
            imag = 0.0

            for n in range(N):
                angle = -2 * math.pi * k * n / N

                real += signal.matrix[n, 1] * math.cos(angle)
                imag += signal.matrix[n, 1] * math.sin(angle)

            amp = math.sqrt(real**2 + imag**2)
            phase = math.atan2(imag, real)
            a.append(amp)
            p.append(phase)

        result_matrix = np.column_stack((a, p))
        result = Signal(True, signal.is_periodic, result_matrix)

    return result


def remove_dc_component(freq_signal):
    result = Signal(
        freq_signal.signal_type, freq_signal.is_periodic, freq_signal.matrix.copy()
    )
    if len(result.matrix) > 0:
        result.matrix[0, 0] = 0.0
    return result


def FFT_IFFT(signal, inverse=0):

    if inverse == 0:
        _, a = signal.split()
        x = a.astype(complex)
        N = len(x)
    else:
        a, p = signal.split()
        x = a * np.exp(1j * p)
        N = len(x)

    def recurse(x):
        n = len(x)
        if n == 1:
            return x.copy()

        even = recurse(x[0::2])
        odd = recurse(x[1::2])

        result = np.zeros(n, dtype=complex)
        sign = 1 if inverse else -1
        for k in range(n // 2):
            W = cmath.exp(sign * 2j * cmath.pi * k / n)
            result[k] = even[k] + W * odd[k]
            result[k + n // 2] = even[k] - W * odd[k]
            if inverse:
                result[k] /= 2
                result[k + n // 2] /= 2
        return result

    X = recurse(x)

    if inverse == 0:
        amplitudes = np.abs(X)
        phases = np.angle(X)
        matrix = np.column_stack((amplitudes, phases))
        return Signal(True, signal.is_periodic, matrix)
    else:
        time_amplitude = np.real(X)
        time_indices = np.arange(N)
        matrix = np.column_stack((time_indices, time_amplitude))
        return Signal(False, signal.is_periodic, matrix)


def _estimate_dt(signal):
    t, a = signal.split()
    if len(t) < 2:
        return 1.0
    return float(np.mean(np.diff(t)))


def moving_average(signal, window_size=3):
    if signal.signal_type != False:
        return None
    t, a = signal.split()
    if window_size <= 1:
        return Signal(False, signal.is_periodic, signal.matrix.copy())
    new_sz = len(a) - window_size + 1
    a_avg = np.zeros(new_sz)
    for i in range(0, window_size):
        a_avg[0] += a[i]
    a_avg[0] /= window_size
    for i in range(1, new_sz):
        if i + window_size - 1 < len(a):
            a_avg[i] = (
                a_avg[i - 1] * window_size - a[i - 1] + a[i + window_size - 1]
            ) / window_size
    result = Signal(False, signal.is_periodic, np.column_stack((t[0:new_sz], a_avg)))
    return result


def first_derivative(signal):
    if signal.signal_type != False:
        return None
    t, a = signal.split()
    # y[n] = x[n] - x[n-1]
    y = np.zeros_like(a)
    if len(a) > 0:
        y[0] = 0.0
    for i in range(0, len(a)):
        y[i] = a[i]
        if i > 0:
            y[i] -= a[i - 1]
    return Signal(False, signal.is_periodic, np.column_stack((t, y)))


def second_derivative(signal):
    if signal.signal_type != False:
        return None
    t, a = signal.split()
    # y[n] = x[n+1] - 2 x[n] + x[n-1]
    y = np.zeros_like(a)
    for i in range(0, len(a)):
        if 1 < i < len(a) - 1:
            y[i] = a[i + 1] - 2 * a[i] + a[i - 1]
        else:
            y[i] = 0.0
    return Signal(False, signal.is_periodic, np.column_stack((t, y)))


def delay_advance(signal, k=0):
    if signal.signal_type != False:
        return None
    t, a = signal.split()
    dt = _estimate_dt(signal)
    new_t = t + k * dt
    return Signal(False, signal.is_periodic, np.column_stack((new_t, a.copy())))


def fold_signal(signal):
    if signal.signal_type != False:
        return None
    t, a = signal.split()
    new_t = -t
    indices = np.argsort(new_t)
    return Signal(
        False, signal.is_periodic, np.column_stack((new_t[indices], a[indices].copy()))
    )


def fold_and_shift(signal, k=0):
    folded = fold_signal(signal)
    return delay_advance(folded, k)


def remove_dc_time(signal):
    if signal.signal_type != False:
        return None
    t, a = signal.split()
    a_no_dc = a - np.mean(a)
    return Signal(False, signal.is_periodic, np.column_stack((t, a_no_dc)))


def convolve_signals(signal1, signal2):
    if signal1.signal_type != False or signal2.signal_type != False:
        return None
    t1, a1 = signal1.split()
    t2, a2 = signal2.split()
    if len(a1) == 0 or len(a2) == 0:
        return None
    dt1 = _estimate_dt(signal1)
    dt2 = _estimate_dt(signal2)
    dt = min(dt1, dt2)
    N1 = len(a1)
    N2 = len(a2)
    N = N1 + N2 - 1
    conv = np.zeros(N, dtype=float)
    for n in range(N):
        k_min = max(0, n - (N2 - 1))
        k_max = min(n, N1 - 1)
        s = 0.0
        for k in range(k_min, k_max + 1):
            s += a1[k] * a2[n - k]
        conv[n] = s

    t_start = t1[0] + t2[0]
    t = t_start + np.arange(N) * dt
    return Signal(
        False, signal1.is_periodic or signal2.is_periodic, np.column_stack((t, conv))
    )


def normalized_cross_correlation(signal1, signal2):
    if signal1.signal_type != False or signal2.signal_type != False:
        return None
    t1, a1 = signal1.split()
    t2, a2 = signal2.split()
    if len(a1) == len(a2):
        x = a1.copy()
        y = a2.copy()
    else:
        x = a1 - np.mean(a1)
        y = a2 - np.mean(a2)

    N1 = len(x)
    N2 = len(y)
    if N1 == 0 or N2 == 0:
        return Signal(
            False, signal1.is_periodic or signal2.is_periodic, np.empty((0, 2))
        )

    if N1 == N2:
        denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
        corr_circ = np.zeros(N1, dtype=float)
        for k in range(N1):
            s = 0.0
            for n in range(N1):
                m = (n + k) % N1
                s += x[n] * y[m]
            corr_circ[k] = 0.0 if denom == 0 else s / denom
        matrix = np.column_stack((t1.astype(float), corr_circ))
        return Signal(False, signal1.is_periodic or signal2.is_periodic, matrix)

    L = N1 + N2 - 1
    lags = np.arange(-(N2 - 1), N1)
    corr = np.zeros(L, dtype=float)

    for idx, lag in enumerate(lags):
        n_start = max(0, -lag)
        n_end = min(N1 - 1, N2 - 1 - lag)
        s = 0.0
        for n in range(n_start, n_end + 1):
            m = n + lag
            s += x[n] * y[m]
        corr[idx] = s

    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if denom == 0:
        corr = corr * 0.0
    else:
        corr = corr / denom

    matrix = np.column_stack((lags.astype(float), corr))
    return Signal(False, signal1.is_periodic or signal2.is_periodic, matrix)


def normalized_autocorrelation(signal):
    if signal.signal_type != False:
        return None
    t, a = signal.split()
    N = len(a)
    if N == 0:
        return Signal(False, signal.is_periodic, np.empty((0, 2)))

    x = a.copy()
    denom = np.sum(x**2)
    L = 2 * N - 1
    corr = np.zeros(L, dtype=float)
    if denom != 0:
        for k in range(L):
            lag = k - (N - 1)
            s = 0.0
            n_start = max(0, -lag)
            n_end = min(N - 1, N - 1 - lag)
            for n in range(n_start, n_end + 1):
                m = n + lag
                s += x[n] * x[m]
            corr[k] = s / denom

    lags = np.arange(-(N - 1), N)
    matrix = np.column_stack((lags.astype(float), corr))
    return Signal(False, signal.is_periodic, matrix)


def periodic_cross_correlation(signal1, signal2):
    if signal1.signal_type != False or signal2.signal_type != False:
        return None
    t1, a1 = signal1.split()
    t2, a2 = signal2.split()
    N1 = len(a1)
    N2 = len(a2)
    if N1 == 0 or N2 == 0:
        return None
    L = N1 + N2 - 1
    x = np.zeros(L, dtype=float)
    y = np.zeros(L, dtype=float)
    x[:N1] = a1
    y[:N2] = a2

    denom = math.sqrt(np.sum(a1**2) * np.sum(a2**2))
    corr = np.zeros(L, dtype=float)
    if denom != 0:
        for k in range(L):
            s = 0.0
            for n in range(L):
                m = (n + k) % L
                s += x[n] * y[m]
            corr[k] = s / denom
    else:
        corr[:] = 0.0

    lags = np.arange(L)
    matrix = np.column_stack((lags.astype(float), corr))
    return Signal(False, signal1.is_periodic or signal2.is_periodic, matrix)


def time_delay_analysis(signal1, signal2, Ts=1.0):
    corr = periodic_cross_correlation(signal1, signal2)
    if corr is None:
        return None
    if hasattr(corr, "split"):
        _, corr_arr = corr.split()
    else:
        corr_arr = corr
    L = len(corr_arr)
    idx = int(np.argmax(corr_arr))
    if idx > L // 2:
        lag = idx - L
    else:
        lag = idx
    delay_seconds = lag * Ts
    return delay_seconds, lag, corr


def compute_N(delta_f, fs, delta_s):
    delta_f_norm = delta_f / fs
    
    # Choose A based on required attenuation (delta_s) and window type
    if delta_s <= 21:
        A = 0.9  # Rectangular window
    elif delta_s <= 44:
        A = 3.1  # Hanning window
    elif delta_s <= 53:
        A = 3.3  # Hamming window
    else:
        A = 5.5  # Blackman window
    
    N_float = A / delta_f_norm
    N = int(math.ceil(N_float))

    if N < 3:
        N = 3

    if N % 2 == 0:
        N += 1

    return N

def choose_window(delta_s):
    if delta_s <= 21:
        window_type = "rectangular"
    elif delta_s <= 44:
        window_type = "hanning"
    elif delta_s <= 53:
        window_type = "hamming"
    else:
        window_type = "blackman"

    return window_type


def normalize_freq(filter_type, fs, fc=None, f1=None, f2=None):
    if filter_type in ["Low Pass", "High Pass"]:
        fc_norm = fc / fs
        return fc_norm
    elif filter_type in ["Band Pass", "Band Stop"]:
        f1_norm = f1 / fs
        f2_norm = f2 / fs
        return f1_norm, f2_norm


def shift_freq(filter_type, half_delta_f, fc=None, f1=None, f2=None):
    if filter_type == "Low Pass":
        shifted_fc = fc + half_delta_f
        return shifted_fc
    elif filter_type == "High Pass":
        shifted_fc = fc - half_delta_f
        return shifted_fc
    elif filter_type == "Band Pass":
        shifted_f1 = f1 - half_delta_f
        shifted_f2 = f2 + half_delta_f
        return shifted_f1, shifted_f2
    elif filter_type == "Band Stop":
        shifted_f1 = f1 + half_delta_f
        shifted_f2 = f2 - half_delta_f
        return shifted_f1, shifted_f2


def make_window(window_type, N):
    if window_type == "rectangular":
        return np.ones(N)
    elif window_type == "hanning":
        return np.hanning(N)
    elif window_type == "hamming":
        return np.hamming(N)
    elif window_type == "blackman":
        return np.blackman(N)


def compute_fir_coff(filter_type,fc=None,f1=None,f2=None,N=51,window_type="hamming"):
    m = (N-1) // 2
    n = np.arange(N)
    k = n - m
    
    h_ideal = np.zeros(N, dtype=float)
    
    delta = np.zeros(N)
    delta[m] = 1
    
    if filter_type == "Low Pass":
        x = 2.0 * fc * k
        h_ideal = 2.0 * fc * np.sinc(x)
    elif filter_type == "High Pass":
        x = 2.0 * fc * k
        h_ideal = delta - 2.0 * fc * np.sinc(x)
    elif filter_type == "Band Pass":
        x1 = 2.0 * f1 * k
        x2 = 2.0 * f2 * k
        h_ideal = 2 * f2 * np.sinc(x2) - 2 * f1 * np.sinc(x1)
    elif filter_type == "Band Stop":
        x1 = 2.0 * f1 * k
        x2 = 2.0 * f2 * k
        h_ideal = 2.0 * f1 * np.sinc(x1) + (delta - 2.0 * f2 * np.sinc(x2)) 
    
    w = make_window(window_type, N)
    h = h_ideal * w
    
    return h, h_ideal, w

def apply_filter(x, h):
    y = x * h
    return y

def apply_conv(x, h):
    t = np.zeros(len(x))
    x_signal = Signal(False,False,np.column_stack((t, x)))
    h_signal = Signal(False,False,np.column_stack((t[:len(h)], h)))
    _,y = convolve_signals(x_signal,h_signal).split()
    
    return y

def apply_conv_same(x, h):
    y_full = np.convolve(x, h, mode='full')
    return y_full

def save_txt(h, filepath="signals/task7/coff/coffecients.txt"):
    np.savetxt(filepath, h, fmt="%.8f")

def upsample(signal, L):
    if signal.signal_type != False:
        return None
    t, a = signal.split()
    N = len(a)
    a_up = np.zeros(N * L)
    a_up[::L] = a
    
    dt = _estimate_dt(signal)
    t_up = t[0] + np.arange(N * L) * (dt / L)
    
    return Signal(False, signal.is_periodic, np.column_stack((t_up, a_up)))

def downsample(signal, M):
    if signal.signal_type != False:
        return None
    t, a = signal.split()
    a_down = a[::M]
    t_down = t[::M]
    
    return Signal(False, signal.is_periodic, np.column_stack((t_down, a_down)))

def resample_signal(signal, M, L, filter_specs=None):
    if M == 0 and L == 0:
        raise ValueError("Both M and L cannot be zero")
    
    result = signal
    h = None
    
    if M == 0 and L != 0:
        result = upsample(result, L)
        if filter_specs:
            h, _, _ = compute_fir_coff(
                filter_type="Low Pass",
                fc=filter_specs['fc_norm'],
                N=filter_specs['N'],
                window_type=filter_specs['window_type']
            )
            t_before, a = result.split()
            a_filtered = apply_conv_same(a, h)
            dt = _estimate_dt(result)
            t_new = t_before[0] + np.arange(len(a_filtered)) * dt
            result = Signal(False, result.is_periodic, np.column_stack((t_new, a_filtered)))
    
    elif M != 0 and L == 0:
        if filter_specs:
            h, _, _ = compute_fir_coff(
                filter_type="Low Pass",
                fc=filter_specs['fc_norm'],
                N=filter_specs['N'],
                window_type=filter_specs['window_type']
            )
            t_before, a = result.split()
            a_filtered = apply_conv_same(a, h)
            dt = _estimate_dt(result)
            t_new = t_before[0] + np.arange(len(a_filtered)) * dt
            result = Signal(False, result.is_periodic, np.column_stack((t_new, a_filtered)))
        result = downsample(result, M)
    
    elif M != 0 and L != 0:
        result = upsample(result, L)
        if filter_specs:
            h, _, _ = compute_fir_coff(
                filter_type="Low Pass",
                fc=filter_specs['fc_norm'],
                N=filter_specs['N'],
                window_type=filter_specs['window_type']
            )
            t_before, a = result.split()
            a_filtered = apply_conv_same(a, h)
            dt = _estimate_dt(result)
            t_new = t_before[0] + np.arange(len(a_filtered)) * dt
            result = Signal(False, result.is_periodic, np.column_stack((t_new, a_filtered)))
        result = downsample(result, M)
    
    return result, h