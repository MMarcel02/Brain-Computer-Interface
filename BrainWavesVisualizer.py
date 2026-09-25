#!/usr/bin/env python3
"""
BrainWavesVisualizer.py
Pygame live dashboard for Unicorn -> LSL (8 EEG channels)

Left:  8 live traces (µV), fixed scale ±5 µV (adjust ±1 µV with ↑/↓), centered by robust baseline.
Right: Live analysis using FrequencyAnalysis.py
       - Welch PSD (avg across channels) 1–30 Hz (log power)
       - SSVEP CCA bars (target freqs), predicted freq label

This is the file that turns real EEG into the 'BCI_FREQ' LSL stream that
Main.py listens to. The four target frequencies and their detection
thresholds come from Config.py (Config.DIRECTIONS) — change an arrow's
frequency or threshold there, not in this file.

Controls:
  ↑ / ↓     : increase/decrease amplitude range by exactly 1 µV
  R         : recenter baselines to current medians
  ← / →     : shorten/lengthen rolling window (±1 s, 2–60 s)
  Space     : pause/resume
  Q / Esc   : quit

Dependencies:
  numpy pygame pylsl scipy scikit-learn
"""

import argparse
import time
import threading
from typing import List, Tuple

import numpy as np
import pygame
from pylsl import StreamInlet, resolve_byprop, cf_float32

#Stream classified signals to the game
from pylsl import StreamInfo, StreamOutlet

# Local analysis helpers
from FrequencyAnalysis import live_psd_and_ssvep
from Config import freq1, freq2, freq3, freq4, FREQ_THRESHOLDS, CONSECUTIVE_REQUIRED_DEFAULT

# ---- Stream / display defaults ----
DEFAULT_NAME = "Unicorn"
DEFAULT_TYPE = "EEG"
DEFAULT_FS   = 250
N_CH         = 8

WINDOW_SEC   = 1.0       # rolling window seconds
FPS          = 60         # UI FPS
WIN_W, WIN_H = 1200, 620  # window size (wider for sidebar)

# Scale: fixed, start at ±5 µV, step 1
YLIM_UV_START = 5.0
YLIM_STEP_UV  = 1.0
YLIM_MIN_UV   = 1.0
YLIM_MAX_UV   = 500.0

# Colors
BG_COLOR      = (18, 18, 18)
GRID_COLOR    = (40, 40, 40)
TEXT_COLOR    = (210, 210, 210)
TRACE_COLOR   = (120, 200, 255)
ZERO_COLOR    = (90, 90, 90)
BORDER_COLOR  = (60, 60, 60)
ACCENT_COLOR  = (255, 180, 80)
BAR_COLOR     = (100, 210, 140)

# nV -> µV
NV_TO_UV = 1.0 / 1000.0

# Baseline smoothing (for centering)
BASELINE_ALPHA = 0.05  # EMA

# Plot mapping: occupy ~90% of pane height with ±ylim
HEIGHT_FRACTION = 0.90

# Analysis cadence (do heavy computations less often than drawing)
ANALYSIS_HZ = 4.0        # compute PSD/CCA this many times per second
ANALYSIS_EVERY_FRAMES = max(1, int(FPS / ANALYSIS_HZ))

# Target freqs — read from Config.py so they always match the arrows shown
# in the game (Main.py / UI.py).
TARGET_FREQS = (freq1, freq2, freq3, freq4)  # ~8.57, 10, 12, 15 Hz by default


# --------------------- LSL Reader --------------------- #
class LSLReader:
    """Background reader that fills a rolling buffer from an LSL inlet."""
    def __init__(self, name: str, stype: str, win_sec: float, fallback_fs: float):
        self.inlet = self._connect(name, stype, timeout=30)
        info = self.inlet.info()
        self.fs = int(info.nominal_srate()) or int(fallback_fs)
        self.n = max(100, int(win_sec * self.fs))
        self.buf = np.zeros((N_CH, self.n), dtype=np.float32)
        self.idx = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @staticmethod
    def _connect(name, stype, timeout=30):
        print(f"[lsl] resolving stream name='{name}', type='{stype}'...")
        t0 = time.time()
        stream = None
        while (time.time() - t0) < timeout and stream is None:
            cand = resolve_byprop('type', stype, timeout=2)
            exact = [s for s in cand if s.name() == name]
            stream = exact[0] if exact else (cand[0] if cand else None)
        if stream is None:
            raise RuntimeError(f"No LSL stream found (type='{stype}', name='{name}').")
        print(f"[lsl] connected to '{stream.name()}', chans={stream.channel_count()}, fs={int(stream.nominal_srate())}")
        return StreamInlet(stream, max_buflen=60, processing_flags=0)

    def _run(self):
        self.inlet.flush()
        while not self.stop.is_set():
            samples, _ = self.inlet.pull_chunk(timeout=0.05, max_samples=1024)
            if not samples:
                continue
            arr = np.asarray(samples, dtype=np.float32).T  # [ch x m]
            if arr.shape[0] >= N_CH:
                arr = arr[:N_CH, :]
            else:
                pad = np.zeros((N_CH - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                arr = np.vstack((arr, pad))

            m = arr.shape[1]
            with self.lock:
                j = self.idx % self.n
                end = (j + m) % self.n
                if j < end:
                    self.buf[:, j:end] = arr
                else:
                    k = self.n - j
                    self.buf[:, j:] = arr[:, :k]
                    self.buf[:, :end] = arr[:, k:]
                self.idx = (self.idx + m) % (2 * self.n)

    def snapshot(self) -> np.ndarray:
        with self.lock:
            j = self.idx % self.n
            if j == 0:
                return self.buf.copy()
            return np.hstack((self.buf[:, j:], self.buf[:, :j]))

    def resize_window(self, new_n: int):
        with self.lock:
            new_n = max(100, int(new_n))
            old = self.snapshot()
            if old.shape[1] >= new_n:
                self.buf = old[:, -new_n:]
            else:
                pad = np.zeros((N_CH, new_n - old.shape[1]), dtype=old.dtype)
                self.buf = np.hstack((pad, old))
            self.n = new_n
            self.idx = 0

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.0)


# --------------------- Drawing helpers --------------------- #
def draw_grid(surface: pygame.Surface, rect: pygame.Rect, x_ticks: int = 10, y_ticks: int = 4):
    pygame.draw.rect(surface, BORDER_COLOR, rect, width=1)
    w, h = rect.width, rect.height
    for i in range(1, x_ticks):
        x = rect.left + int(i * w / x_ticks)
        pygame.draw.line(surface, GRID_COLOR, (x, rect.top), (x, rect.bottom), 1)
    for j in range(1, y_ticks):
        y = rect.top + int(j * h / y_ticks)
        pygame.draw.line(surface, GRID_COLOR, (rect.left, y), (rect.right, y), 1)

def signal_to_polyline(sig_uv: np.ndarray, rect: pygame.Rect, ylim_uv: float) -> List[Tuple[int, int]]:
    n = sig_uv.size
    if n <= 1:
        return []
    xs = np.linspace(rect.left, rect.right - 1, n, dtype=np.int32)
    px_per_uV = (rect.height * 0.90) / max(1e-6, 2 * ylim_uv) * 2  # map ±ylim to ~90% pane height
    zero_y = rect.centery
    ys = (zero_y - sig_uv * px_per_uV).astype(np.int32)
    return list(zip(xs.tolist(), ys.tolist()))


# --------------------- Sidebar charts --------------------- #
def draw_psd(surface: pygame.Surface, rect: pygame.Rect, freqs: np.ndarray, psd: np.ndarray, font, title="PSD (avg, 6–30 Hz)"):
    """
    Plot log10(avg power) in 6–30 Hz with x-axis ticks (vertical text)
    and dashed vertical guide lines at 8.6, 10, 12, 15 Hz.
    """
    # Frame & title
    pygame.draw.rect(surface, BORDER_COLOR, rect, width=1)
    surface.blit(font.render(title, True, TEXT_COLOR), (rect.left + 6, rect.top + 6))

    if freqs is None or psd is None or psd.size == 0:
        surface.blit(font.render("PSD: waiting...", True, TEXT_COLOR), (rect.left + 6, rect.top + 28))
        return

    # ---- crop to 6–30 Hz
    f0, f1 = 6.0, 30.0
    sel = (freqs >= f0) & (freqs <= f1)
    if not np.any(sel):
        surface.blit(font.render("PSD: no bins in 6–30 Hz", True, TEXT_COLOR), (rect.left + 6, rect.top + 28))
        return
    f = freqs[sel]
    P = psd[sel, :]  # (K, C)

    # Average across channels and log
    p = np.mean(P, axis=1)
    p = np.maximum(p, 1e-20)
    p_log = np.log10(p)

    # Plot area (leave space for labels)
    left   = rect.left + 44
    right  = rect.right - 20
    top    = rect.top + 26
    bottom = rect.bottom - 40
    w = max(2, right - left)
    h = max(2, bottom - top)

    # Horizontal grid lines
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(top + (1.0 - frac) * h)
        pygame.draw.line(surface, GRID_COLOR, (left, y), (right, y), 1)

    # Normalize vertical to panel
    ymin, ymax = float(np.min(p_log)), float(np.max(p_log))
    if ymax - ymin < 1e-6:
        ymax = ymin + 1e-6
    xs = (left + (f - f0) / (f1 - f0) * (w - 1)).astype(np.int32)
    ys = (bottom - 1 - (p_log - ymin) / (ymax - ymin) * (h - 1)).astype(np.int32)

    pts = list(zip(xs.tolist(), ys.tolist()))
    if len(pts) >= 2:
        pygame.draw.lines(surface, ACCENT_COLOR, False, pts, 2)

    # ---- x-axis ticks, vertical guide lines, vertical labels
    tick_freqs = [6, 8.6, 10, 12, 15, 20, 25, 30]
    interest_freqs = [8.6, 10, 12, 15]

    label_y = bottom + 4
    for tf in tick_freqs:
        if tf < f0 or tf > f1:
            continue
        x = int(left + (tf - f0) / (f1 - f0) * (w - 1))

        # If frequency is one of the targets, draw a dashed vertical line
        if tf in interest_freqs:
            dash_y = top
            while dash_y < bottom:
                pygame.draw.line(surface, (120, 120, 120), (x, dash_y), (x, dash_y + 4), 1)
                dash_y += 8

        # small tick mark
        pygame.draw.line(surface, TEXT_COLOR, (x, bottom - 4), (x, bottom), 1)

        # vertical label
        lbl = font.render(f"{tf}", True, TEXT_COLOR)
        lbl_rot = pygame.transform.rotate(lbl, 90)
        surface.blit(lbl_rot, (x - lbl_rot.get_width() // 2, label_y))


def draw_ssvep_bars(surface: pygame.Surface, rect: pygame.Rect, scores: dict, pred: float, font, title="SSVEP CCA (corr)"):
    """
    Draw SSVEP CCA correlation bars with a y-axis scale from 0 to 1.
    Bars are scaled directly by their correlation (clamped to [0,1]),
    not normalized to the max value.
    """
    # Panel title
    pygame.draw.rect(surface, BORDER_COLOR, rect, width=1)
    surface.blit(font.render(title, True, TEXT_COLOR), (rect.left + 6, rect.top + 6))

    # Layout with space on the left for y-axis labels
    left_margin = 44  # px for "1.00" labels
    top_margin = 28
    right_margin = 10
    bottom_margin = 12

    plot_left   = rect.left + left_margin
    plot_top    = rect.top + top_margin
    plot_width  = max(10, rect.width - left_margin - right_margin)
    plot_height = max(10, rect.height - top_margin - bottom_margin)

    # Background grid (horizontal lines at 0..1)
    pygame.draw.rect(surface, (35, 35, 35), pygame.Rect(plot_left, plot_top, plot_width, plot_height), width=1)
    for val in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(plot_top + (1.0 - val) * plot_height)
        pygame.draw.line(surface, GRID_COLOR, (plot_left, y), (plot_left + plot_width, y), 1)
        # y-axis tick + label
        tick_x = plot_left - 6
        pygame.draw.line(surface, TEXT_COLOR, (tick_x, y), (tick_x + 4, y), 1)
        lbl = font.render(f"{val:.2f}", True, TEXT_COLOR)
        surface.blit(lbl, (plot_left - left_margin + 2, y - lbl.get_height() // 2))

    if not scores:
        surface.blit(font.render("waiting...", True, TEXT_COLOR), (rect.left + 6, rect.top + 28))
        return

    # Bars
    freqs = list(scores.keys())
    vals = [max(0.0, min(1.0, float(scores[f]))) for f in freqs]  # clamp to [0,1]

    n = max(1, len(freqs))
    gap_ratio = 0.25  # fraction gap inside each slot
    slot_w = plot_width / n
    bar_w = max(2, int(slot_w * (1.0 - gap_ratio)))

    for i, f in enumerate(freqs):
        v = vals[i]  # already 0..1
        x_center = plot_left + int(i * slot_w + slot_w / 2)
        x0 = x_center - bar_w // 2
        x0 = max(plot_left + 1, min(x0, plot_left + plot_width - bar_w - 1))

        y_bottom = plot_top + plot_height - 1
        y_top = int(y_bottom - v * (plot_height - 2))
        color = BAR_COLOR if f != pred else ACCENT_COLOR
        pygame.draw.rect(surface, color, pygame.Rect(x0, y_top, bar_w, y_bottom - y_top))

        # Frequency label under bar
        flbl = font.render(f"{f:.2f} Hz", True, TEXT_COLOR)
        surface.blit(flbl, (x_center - flbl.get_width() // 2, y_bottom - flbl.get_height()))

        # Optional: numeric value above bar (small)
        vlbl = font.render(f"{v:.2f}", True, TEXT_COLOR)
        surface.blit(vlbl, (x_center - vlbl.get_width() // 2, y_top - vlbl.get_height() - 2))

    # Predicted label at the bottom-left of the panel
    pred_text = font.render(f"Pred: {pred:.2f} Hz", True, ACCENT_COLOR)
    surface.blit(pred_text, (rect.left + 6, rect.bottom - pred_text.get_height() - 6))


def score_for_pred(scores: dict, pred_freq: float) -> float:
    """Return the score for the predicted target freq, robust to float-key mismatch."""
    if not scores or pred_freq is None or not np.isfinite(pred_freq):
        return float("-inf")
    # Find the nearest key in `scores` to the predicted freq
    keys = np.array(list(scores.keys()), dtype=float)
    k = keys[int(np.argmin(np.abs(keys - float(pred_freq))))]
    return float(scores.get(k, float("-inf")))

def nearest_freq_key(keys: List[float], f: float) -> float:
    """Return the key in 'keys' that is closest to f."""
    if not keys:
        return f
    arr = np.array(keys, dtype=float)
    return float(arr[int(np.argmin(np.abs(arr - float(f))))])

def threshold_for_pred_freq(pred_freq: float, default_min: float) -> float:
    """
    Look up the per-frequency threshold for 'pred_freq' using nearest key in FREQ_THRESHOLDS.
    Falls back to default_min (from --min-score) if not found or map is empty.

    FREQ_THRESHOLDS comes from Config.DIRECTIONS — edit the per-arrow
    "threshold" there to make an arrow easier or harder to trigger.
    """
    if not np.isfinite(pred_freq) or not FREQ_THRESHOLDS:
        return float(default_min)
    k = nearest_freq_key(list(FREQ_THRESHOLDS.keys()), pred_freq)
    return float(FREQ_THRESHOLDS.get(k, default_min))


def best_candidate_above_threshold(scores: dict, default_min: float):
    """
    Pick which frequency (if any) is confident enough to publish.

    NOTE: this is NOT the same as "the frequency with the highest score".
    `ssvep.pred_freq` (from FrequencyAnalysis.classify_ssvep_cca) is just the
    argmax over ALL four frequencies' scores, with no threshold involved. If
    you only ever check *that* single winner against its own threshold, a
    frequency with an easy (low) threshold can score perfectly well but never
    get published, simply because some other frequency with a harder (high)
    threshold happened to score a bit higher that frame while still failing
    its own bar. Concretely: North (threshold 0.80) scores 0.68 and "wins"
    the argmax, while East (threshold 0.60) scores 0.62 — a genuinely valid
    East detection — but gets silently dropped because it wasn't the winner.

    So instead we check every candidate frequency against its OWN threshold,
    and, among the ones that pass, publish the highest-scoring one — the
    candidate the classifier is most confident about this frame, out of the
    ones that were actually eligible to be published at all.

    (This project also tried breaking ties by whichever candidate had to
    clear the strictest threshold instead of by score. That has a sharp
    edge worth knowing about: if two arrows share the same threshold value
    — which happens the moment you tune two arrows to the same number —
    "strictest threshold" can no longer tell them apart, and the tie-break
    silently falls back to whichever one happens to come first in `scores`
    (the order target_freqs are listed in), regardless of which one the
    classifier was actually more confident about. Comparing by score
    doesn't have that failure mode, which is why we're using it here.)

    Returns (freq, score) for the best passing candidate, or (None, None) if
    nothing clears its threshold this frame.
    """
    if not scores:
        return None, None
    best_freq, best_score = None, float("-inf")
    for f, s in scores.items():
        f = float(f)
        s = float(s)
        thresh = threshold_for_pred_freq(f, default_min)
        if s >= thresh and s > best_score:
            best_freq, best_score = f, s
    return (best_freq, best_score) if best_freq is not None else (None, None)

# --------------------- Main app --------------------- #
def main():
    ap = argparse.ArgumentParser(description="Unicorn LSL live dashboard (EEG + freq analysis).")
    ap.add_argument("--name", type=str, default=DEFAULT_NAME, help="LSL stream name.")
    ap.add_argument("--type", type=str, default=DEFAULT_TYPE, help="LSL stream type.")
    ap.add_argument( "--consecutive", type=int, default=CONSECUTIVE_REQUIRED_DEFAULT, help=f"Number of consecutive frames above the per-frequency threshold before publishing (default: {CONSECUTIVE_REQUIRED_DEFAULT}).")
    args = ap.parse_args()

    proc = None
    reader = None
    try:
        reader = LSLReader(args.name, args.type, WINDOW_SEC, DEFAULT_FS)

        # --- LSL outlet for predicted frequency (Hz) ---
        info_freq = StreamInfo(
            name="BCI_FREQ",  # stream name
            type="BCI",  # your choice; "Markers" also fine
            channel_count=1,
            nominal_srate=0,  # irregular / event-like
            channel_format=cf_float32,
            source_id="bci_freq_1"
        )
        outlet_freq = StreamOutlet(info_freq)

        pygame.init()
        pygame.display.set_caption("Unicorn EEG – Live (dashboard)")
        screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Consolas,Menlo,Monaco,DejaVu Sans Mono", 14)

        paused = False
        ylim_uv = float(YLIM_UV_START)
        baseline_uv = np.zeros(N_CH, dtype=np.float32)
        last_data_uv = np.zeros((N_CH, 2), dtype=np.float32)

        # Analysis buffers
        freqs = None
        psd = None
        ssvep_scores = {}
        ssvep_pred = None
        consec_ok = 0
        previous_key = None

        frame = 0
        running = True
        while running:
            # ---- events ----
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_UP:
                        ylim_uv = min(YLIM_MAX_UV, ylim_uv + YLIM_STEP_UV)
                    elif event.key == pygame.K_DOWN:
                        ylim_uv = max(YLIM_MIN_UV, ylim_uv - YLIM_STEP_UV)
                    elif event.key == pygame.K_r:
                        snap_uv = reader.snapshot() * NV_TO_UV
                        for ch in range(N_CH):
                            baseline_uv[ch] = float(np.median(snap_uv[ch]))
                        print("[viz] baselines reset.")
                    elif event.key == pygame.K_RIGHT:
                        new_win = min(60.0, reader.n / reader.fs + 1.0)
                        reader.resize_window(int(new_win * reader.fs))
                    elif event.key == pygame.K_LEFT:
                        new_win = max(2.0, reader.n / reader.fs - 1.0)
                        reader.resize_window(int(new_win * reader.fs))
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            # ---- background ----
            screen.fill(BG_COLOR)
            width, height = screen.get_size()

            # Layout: left wave area (70%), right sidebar (30%)
            sidebar_w = max(360, int(width * 0.32))
            waves_w = width - sidebar_w - 6
            left_rect = pygame.Rect(4, 4, waves_w - 8, height - 8)
            right_rect = pygame.Rect(width - sidebar_w + 2, 4, sidebar_w - 6, height - 8)

            # ---- data ----
            if not paused:
                data_uv = reader.snapshot() * NV_TO_UV
                last_data_uv = data_uv
            else:
                data_uv = last_data_uv

            # ---- left: channel traces ----
            gap = 4
            pane_h = (left_rect.height - gap * (N_CH + 1)) // N_CH
            for ch in range(N_CH):
                # update baseline (median EMA)
                med = float(np.median(data_uv[ch]))
                baseline_uv[ch] = (1.0 - BASELINE_ALPHA) * baseline_uv[ch] + BASELINE_ALPHA * med

                sig_centered = data_uv[ch] - baseline_uv[ch]

                # pane rect
                top = left_rect.top + gap + ch * (pane_h + gap)
                rect = pygame.Rect(left_rect.left + gap, top, left_rect.width - 2 * gap, pane_h)

                draw_grid(screen, rect)
                pygame.draw.line(screen, ZERO_COLOR, (rect.left, rect.centery), (rect.right, rect.centery), 1)

                pts = signal_to_polyline(sig_centered, rect, ylim_uv)
                if len(pts) >= 2:
                    pygame.draw.lines(screen, TRACE_COLOR, False, pts, 2)

                label = f"Ch {ch+1} | ±{int(ylim_uv)} µV | offset {baseline_uv[ch]:.1f} µV"
                screen.blit(font.render(label, True, TEXT_COLOR), (rect.left + 6, rect.top + 4))

            # ---- right: analysis ----
            sb_gap = 8
            sb_top = right_rect.top + sb_gap
            sb_left = right_rect.left + sb_gap
            sb_width = right_rect.width - 2 * sb_gap
            sb_height = right_rect.height - 2 * sb_gap

            # two stacked panels
            psd_rect = pygame.Rect(sb_left, sb_top, sb_width, int(sb_height * 0.55))
            cca_rect = pygame.Rect(sb_left, psd_rect.bottom + sb_gap, sb_width, right_rect.bottom - sb_gap - (psd_rect.bottom + sb_gap))

            # compute analysis at lower rate
            if frame % ANALYSIS_EVERY_FRAMES == 0:
                try:
                    # data for analysis expects (samples, channels)
                    eeg_window = data_uv.T    # (n, ch)
                    (freqs, psd), ssvep = live_psd_and_ssvep(eeg_window, reader.fs, target_freqs=TARGET_FREQS, psd_window_sec=4.0, cca_window_sec=4.0)
                    if ssvep is not None:
                        # ssvep_pred/ssvep_scores are only used for the on-screen
                        # CCA bars (draw_ssvep_bars) below — showing the raw
                        # top-scoring candidate is fine for that diagnostic view.
                        ssvep_pred = ssvep.pred_freq
                        ssvep_scores = ssvep.scores

                        # Publish gating: check EVERY candidate frequency against
                        # its OWN threshold and take the best one that passes,
                        # instead of only checking the single argmax winner
                        # (ssvep_pred) against its threshold — see the docstring
                        # of best_candidate_above_threshold() for why that matters.
                        publish_freq, publish_score = best_candidate_above_threshold(
                            ssvep_scores, FREQ_THRESHOLDS[ssvep_pred]
                        )
                        if publish_freq is not None:
                            freq_threshold = threshold_for_pred_freq(publish_freq, FREQ_THRESHOLDS[ssvep_pred])
                            print(f"[lsl] Predicted SSVEP: {publish_freq:.3f} Hz | score {publish_score:.3f} ≥ thresh {freq_threshold:.3f}")
                            if previous_key == publish_freq:
                                consec_ok += 1
                            else:
                                consec_ok = 1
                                previous_key = publish_freq

                            print(f"[lsl] Counter: {consec_ok}")
                            if consec_ok==args.consecutive:
                                outlet_freq.push_sample([publish_freq])
                                print(f"[lsl] ----------------- Sent SSVEP: {publish_freq:.3f} Hz")
                                consec_ok = 0
                                previous_key = None

                except Exception as e:
                    # Keep last good analysis if something hiccups
                    print("[analysis error]", e)

            draw_psd(screen, psd_rect, freqs, psd, font)
            draw_ssvep_bars(screen, cca_rect, ssvep_scores, ssvep_pred if ssvep_pred else 0.0, font)

            # ---- HUD ----
            win_sec = reader.n / reader.fs
            hud = (
                f"win={win_sec:.1f}s  fs≈{reader.fs}Hz  units=µV  "
                f"scale=±{ylim_uv:.0f}µV  (↑/↓)±1µV  [R]recenter  (←/→)window  [Space]pause  [Q/Esc]quit"
            )
            hudsurf = font.render(hud, True, TEXT_COLOR)
            screen.blit(hudsurf, (8, height - hudsurf.get_height() - 6))

            pygame.display.flip()
            frame += 1
            clock.tick(FPS)

    except Exception as e:
        print(f"[error] {e}")
        raise
    finally:
        try:
            if reader is not None:
                reader.close()
        except Exception:
            pass
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        pygame.quit()


if __name__ == "__main__":
    main()