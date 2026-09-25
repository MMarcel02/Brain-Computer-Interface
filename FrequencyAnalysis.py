#!/usr/bin/env python3
"""
FrequencyAnalysis.py
Lightweight frequency-domain & SSVEP helpers for live EEG dashboards.

Used by BrainWavesVisualizer.py to turn raw EEG into a predicted flicker
frequency. The target frequencies it classifies against (freq1..freq4) and
the window lengths it uses are read from Config.py, so changing an arrow's
frequency in Config.DIRECTIONS is picked up here automatically — you
shouldn't need to edit this file to retune the arrows.

Dependencies:
    numpy, scipy, scikit-learn

All functions are pure (no state) so they can be called from any UI thread/timer.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.signal as sp
from sklearn.cross_decomposition import CCA

from Config import welch_window_s, cca_window_s, psd_window_s, freq1, freq2, freq3, freq4


# --------------------------- Basic DSP ---------------------------

def welch_spectrum(
    data: np.ndarray,
    fs: float,
    window_sec: float = welch_window_s,
    fmin: float = 1.0,
    fmax: float = 30.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute Welch spectrum (median-averaged) channel-wise.

    Parameters
    ----------
    data : (samples, channels)
    fs   : sampling rate (Hz)
    window_sec : window length for nperseg
    fmin, fmax : frequency range to keep

    Returns
    -------
    freqs : (K,) frequencies in Hz
    spec  : (K, channels) spectrum (linear units, 'spectrum' scaling)
    """
    if data is None or data.size == 0:
        return None

    n = data.shape[0]
    nperseg = max(8, int(window_sec * fs))
    nperseg = min(n, nperseg)
    if nperseg < 8:
        return None

    noverlap = max(0, nperseg - 1)

    freqs, spec = sp.welch(
        data,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        average="median",
        axis=0,
        detrend=False,
        return_onesided=True,
    )
    # keep band of interest
    sel = (freqs >= fmin) & (freqs <= fmax)
    return freqs[sel], spec[sel, :]


def bandpass_filter(data: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass (applies along axis=0 for (samples, channels))."""
    if data.size == 0:
        return data
    nyq = 0.5 * fs
    b, a = sp.butter(order, [low / nyq, high / nyq], btype="band")
    return sp.filtfilt(b, a, data, axis=0)


def bandpower(freqs: np.ndarray, spec: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    """
    Integrate power within 'band' for each channel.
    freqs : (K,)
    spec  : (K, C)
    band  : (f_lo, f_hi)
    Returns (C,)
    """
    if freqs is None or spec is None:
        return np.zeros(spec.shape[1] if spec is not None else 0)
    lo, hi = band
    sel = (freqs >= lo) & (freqs <= hi)
    if not np.any(sel):
        return np.zeros(spec.shape[1])
    # trapezoidal integrate along freq axis
    return np.trapz(spec[sel, :], freqs[sel], axis=0)


# --------------------------- SSVEP (CCA) ---------------------------

def generate_reference_signals(freq: float, n_samples: int, fs: float, n_harmonics: int = 2) -> np.ndarray:
    """
    Generate sine/cosine reference signals for given freq and harmonics.
    Returns array shape (n_samples, 2*n_harmonics).
    """
    t = np.arange(n_samples, dtype=np.float64) / fs
    refs = []
    for h in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * h * freq * t))
        refs.append(np.cos(2 * np.pi * h * freq * t))
    return np.stack(refs, axis=1)


def classify_ssvep_cca(
    eeg_segment: np.ndarray,
    fs: float,
    target_freqs: Iterable[float],
    n_harmonics: int = 2,
    bandpass: Optional[Tuple[float, float]] = (6.0, 30.0),
) -> Tuple[float, Dict[float, float]]:
    """
    Classify SSVEP frequency using CCA on a single segment.

    Parameters
    ----------
    eeg_segment : (samples, channels)
    fs : sampling frequency
    target_freqs : iterable of candidate frequencies (Hz)
    n_harmonics : number of harmonics in refs
    bandpass : (low, high) or None

    Returns
    -------
    pred_freq : argmax frequency
    scores    : dict mapping freq -> correlation
    """
    X = eeg_segment
    if bandpass is not None:
        X = bandpass_filter(X, bandpass[0], bandpass[1], fs)

    # ensure float64 to keep sklearn happy
    X = np.asarray(X, dtype=np.float64)
    n_samples, _ = X.shape

    scores: Dict[float, float] = {}
    for f in target_freqs:
        R = generate_reference_signals(f, n_samples, fs, n_harmonics=n_harmonics)  # (2H, n) -> we want (n, 2H)
        R = R.T  # (2H, n)
        R = R.T  # back to (n, 2H) – explicit to show intent

        cca = CCA(n_components=1)
        cca.fit(X, R)
        Xc, Rc = cca.transform(X, R)
        # corr of the first canonical variates
        num = np.cov(Xc[:, 0], Rc[:, 0], bias=True)[0, 1]
        den = np.std(Xc[:, 0]) * np.std(Rc[:, 0]) + 1e-12
        corr = float(np.clip(num / den, -1.0, 1.0))
        scores[f] = corr

    pred = max(scores, key=scores.get)
    return pred, scores


# --------------------------- Sliding Window ---------------------------

def sliding_window(data: np.ndarray, window_size: int, step_size: int):
    """Yield (start_idx, window) for a (samples, channels) array."""
    n = data.shape[0]
    for start in range(0, max(0, n - window_size + 1), step_size):
        yield start, data[start:start + window_size, :]


def evaluate_ssvep_cca(
    eeg_data: np.ndarray,
    fs: float,
    true_freq: float,
    target_freqs: Iterable[float],
    window_sec: float = 2.0,
    step_sec: float = 0.5,
    n_harmonics: int = 2,
) -> Tuple[np.ndarray, float]:
    """
    Run CCA-based SSVEP classification in sliding windows (offline-style).

    Returns predictions per window and accuracy vs. true_freq.
    """
    w = int(window_sec * fs)
    s = int(step_sec * fs)
    preds: List[float] = []
    for _, seg in sliding_window(eeg_data, w, s):
        if seg.shape[0] < w:
            continue
        pred, _ = classify_ssvep_cca(seg, fs, target_freqs, n_harmonics=n_harmonics)
        preds.append(pred)
    if len(preds) == 0:
        return np.array([]), np.nan
    preds = np.array(preds)
    acc = float(np.mean(preds == true_freq))
    return preds, acc


# --------------------------- Convenience ---------------------------

@dataclass(frozen=True)
class SSVEPResult:
    pred_freq: float
    scores: Dict[float, float]


def live_psd_and_ssvep(
    eeg_window_uv: np.ndarray,
    fs: float,
    target_freqs: Iterable[float] = (freq1, freq2, freq3, freq4),  # from Config.py — change arrows there
    psd_window_sec: float = psd_window_s,
    cca_window_sec: float = cca_window_s,
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[SSVEPResult]]:
    """
    Convenience: compute PSD (Welch) on the whole window and SSVEP-CCA
    on the last cca_window_sec of data.

    Inputs are in microvolts; scaling doesn't affect CCA or PSD shape.

    Returns
    -------
    (freqs, psd) or None
    SSVEPResult or None
    """
    if eeg_window_uv is None or eeg_window_uv.size == 0:
        return None, None

    # PSD on the full window
    psd = welch_spectrum(eeg_window_uv, fs, window_sec=psd_window_sec, fmin=1.0, fmax=30.0)

    # CCA on last cca_window_sec
    n = eeg_window_uv.shape[0]
    n_cca = max(1, int(cca_window_sec * fs))
    seg = eeg_window_uv[-n_cca:, :]
    pred, scores = classify_ssvep_cca(seg, fs, target_freqs, n_harmonics=3, bandpass=(6, 30))
    return psd, SSVEPResult(pred, scores)