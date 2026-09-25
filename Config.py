"""
Config.py — all the tunable numbers and shared "vocabulary" for the game,
in one file, so you don't have to go hunting through Main.py / Controller.py /
UI.py to find the value you want to change.

This file has TWO zones, clearly marked:

  1. GENERAL SETTINGS      — colors, sizes, speeds, the maze file. Change
                              these freely while experimenting with
                              usability fixes.

  2. ARROWS & BCI TUNING   — everything about the four arrows: which key
                              they are, which way they move the player,
                              what frequency they flicker at, and how
                              confident the BCI classifier needs to be
                              before it acts on a detection.

"""
import math
import numpy as np


# =====================================================================
# 1. GENERAL SETTINGS — gameplay, layout, look & feel
# =====================================================================

# --- Maze ---
MAZE_PATH = "mazes/level0.txt"  # which ASCII maze file to load (see Maze.py)

# --- Window & layout ---
WINDOW_W, WINDOW_H = 1280, 720   # starting window size (the game maximizes on top of this)
MIN_SIDEBAR_PX = 200             # minimum width (px) reserved for the arrow sidebar in UI.py
                                  # -> too small and the arrows/labels get cramped.
ARROW_SIZE_PX = 36               # fixed size (px) of each sidebar arrow. Unlike MIN_SIDEBAR_PX,
                                  # this does NOT adjust itself to the window — if you make the
                                  # sidebar much narrower/shorter than the default, arrows this
                                  # size may not all fit

# --- Movement feel ---
WALK_SPEED_PX_S = 150     # currently unused
MOVE_COOLDOWN_S = 0.25    # seconds to wait after a successful move before accepting the next
                           # one. Lower = more responsive but easier to overshoot; higher =
                           # safer but feels laggy.

# --- Arrow order in the sidebar (top -> bottom) ---
# This is PURELY visual — it does not change what each arrow does, only where
# it is drawn. Reorder this list to test different arrow layouts.
SIDEBAR_ORDER = ["N", "W", "S", "E"]

# --- Arrow flicker style ---
# "square" (default) - hard on/off flicker, the classic SSVEP, at
#                       EXACTLY the frequency each arrow is given below.
#                       Only the frequencies in SQUARE_MODE_ALLOWED_HZ
#                       (just below) are allowed in this mode.
# "cosine"            - smooth, continuously-fading flicker. Reproduces
#                       ANY frequency exactly, with no restriction on which
#                       values you can use. See FlashableIcon.py for the
#                       full explanation, and re-check FREQ_THRESHOLDS
#                       below if you switch, since they were tuned for
#                       the square-wave flicker.
ICON_FLICKER_MODE = "square"

# In "square" mode, an arrow's "hz" (below) must be one of these exact
# values. Each one was chosen so that REFRESH_HZ / hz comes out to a whole,
# EVEN number of screen frames: "whole" so the flicker isn't slightly off
# the frequency it's named after, and "even" so its on/off halves split
# exactly in two (an uneven split makes it a weaker, less "square"
# stimulus). This list is REFRESH_HZ / {2, 4, 6, 8, 10, 12, 14}:
#   30, 15, 10, 7.5, 6, 5, 4.29 Hz  ->  2, 4, 6, 8, 10, 12, 14 frames
# ("4.29" is a rounded display of 60/14 ≈ 4.285714... Hz,  Config still
# checks/rounds to the correct 14 frames for it; see FlashableIcon.py.)
# Want a frequency that isn't on this list? Switch that arrow's mode to
# "cosine" instead of trying to add a new value here: cosine mode has no
# such restriction, at the cost of a weaker signal. Also note that not
# every pair of values on this list is safe to use together: two arrows
# whose frequencies are an exact 2x/3x multiple of each other (e.g. 30 & 15,
# or 15 & 5) can be confused by the classifier due to harmonics.
SQUARE_MODE_ALLOWED_HZ = (30, 15, 10, 7.5, 6, 5, 4.29)

# --- Game colors ---
PANEL    = (0, 0, 0)
START    = (170, 215, 255)  # starting square
GOAL     = (180, 255, 190)  # end square
TEXT     = (200, 200, 200)
BG       = (0, 0, 0)
WALL     = (105, 154, 104)
PATH     = (153, 209, 1)
ARROW_ARMED_TINT = (150, 140, 220)  # highlight color when an arrow is "armed" (selected)
CHECKER1         = (255, 255, 255)  # arrow checker-pattern color 1
CHECKER2         = (0, 0, 0)        # arrow checker-pattern color 2


# =====================================================================
# 2. ARROWS & BCI TUNING — one entry per arrow, everything about it
# =====================================================================
#
# The maze game shows 4 flickering arrows. Each one flickers at a specific
# frequency; the EEG cap + BrainWavesVisualizer.py try to detect which
# frequency the player is looking at, and publish that frequency over LSL
# (stream 'BCI_FREQ'). Main.py listens for it and moves the player.
#
# DIRECTIONS is the single source of truth for each arrow:
#   "vector"    -> (row_delta, col_delta) applied to the player's position
#                  when this arrow fires. Must match the maze's row/col
#                  convention (row increases downward, col increases
#                  rightward) — see Maze.py.
#   "hz"        -> the flicker frequency for that arrow, directly in Hz.
#   "threshold" -> how confident (0.0-1.0) the CCA classifier in
#                  FrequencyAnalysis.py needs to be about this frequency
#                  before BrainWavesVisualizer.py will publish it as a
#                  detected move.
#
# Both "hz" and "threshold" are meant to be tuned as you experiment:
#
#   - Changing "hz" changes the flicker frequency, and in "square" mode
#     (the default — see ICON_FLICKER_MODE above) that frequency is exactly
#     what appears on screen. For it to work well:
#       1. In "square" mode, it MUST be one of the values listed in
#          SQUARE_MODE_ALLOWED_HZ above — Config checks this for you at
#          startup and will refuse to run with a clear error if not. (In
#          "cosine" mode, any value is fine — see ICON_FLICKER_MODE above.)
#       2. Don't pick an hz that is an exact 2x/3x multiple of another
#          arrow's hz (e.g. 30 & 15, or 15 & 5). If it is, this arrow's
#          flicker frequency lands exactly on that other arrow's harmonic
#          (see FlashableIcon.py's module docstring for why that matters),
#          which can make the classifier confuse the two arrows.
#       3. Keep the four resulting frequencies reasonably far apart in
#          general, for the same reason.
#   - Changing a "threshold" changes how easily that arrow triggers: lower
#     = easier to trigger but more false positives; higher = more
#     reliable but needs a stronger/cleaner signal. This is one of your
#     main levers for tuning per-arrow usability. Note: the values below
#     were tuned before "square" mode flickered at its exact named
#     frequency (see FlashableIcon.py) — re-check them against a real
#     headset if detections feel off.
#
# Whatever you change here is automatically picked up everywhere else —
# BrainWavesEmulator.py and BrainWavesVisualizer.py import these values
# from this file rather than hardcoding their own, so you only ever need
# to edit this one dictionary.

REFRESH_HZ = 60  # assumed display refresh rate (see FlashableIcon.py)

DIRECTIONS = {
    "N": {"vector": (-1, 0), "hz": 4.29,   "threshold": 0.80},  # -> 14  frames
    "W": {"vector": (0, -1), "hz": 6,    "threshold": 0.80},  # -> 10 frames
    "S": {"vector": (1, 0),  "hz": 10, "threshold": 0.80},  # -> 6 frames
    "E": {"vector": (0, 1),  "hz": 15,   "threshold": 0.60},  # -> 4  frames
}

# --- Everything below this line is DERIVED from DIRECTIONS above. ---
# You should not normally need to edit any of it directly — edit
# DIRECTIONS instead and these will update themselves.

DIRS = list(DIRECTIONS.keys())

# Direction -> movement vector, e.g. VEC["N"] == (-1, 0). Used by Maze.py
# (to find walkable neighbours) and Controller.py (to actually move the
# player) so both always agree on what each direction means.
VEC = {d: cfg["vector"] for d, cfg in DIRECTIONS.items()}

# Direction -> flicker frequency in Hz, e.g. FREQUENCIES["N"] == 10.0
FREQUENCIES = {d: float(cfg["hz"]) for d, cfg in DIRECTIONS.items()}


def _check_no_duplicate_frequencies():
    """Two arrows sharing the exact same Hz would be indistinguishable to
    the classifier (and would silently collapse FREQ_TO_DIR below to just
    one of them)."""
    seen = {}
    for d, hz in FREQUENCIES.items():
        if hz in seen:
            raise ValueError(
                f"Config.DIRECTIONS: arrows '{seen[hz]}' and '{d}' both use "
                f"{hz} Hz. Every arrow needs its own frequency, pick a "
                f"different 'hz' for one of them."
            )
        seen[hz] = d


def _check_square_mode_frequencies():
    """In 'square' mode, every arrow's 'hz' must be one of
    SQUARE_MODE_ALLOWED_HZ: see the comment above that list for why."""
    if ICON_FLICKER_MODE != "square":
        return
    bad = [
        (d, hz) for d, hz in FREQUENCIES.items()
        if not any(math.isclose(hz, allowed, abs_tol=1e-9) for allowed in SQUARE_MODE_ALLOWED_HZ)
    ]
    if bad:
        bad_str = ", ".join(f"{d}={hz}" for d, hz in bad)
        allowed_str = ", ".join(str(h) for h in SQUARE_MODE_ALLOWED_HZ)
        raise ValueError(
            f"Config.ICON_FLICKER_MODE is 'square', but these arrows have an "
            f"'hz' that isn't in SQUARE_MODE_ALLOWED_HZ ({allowed_str}): "
            f"{bad_str}. Change their 'hz' to one of the allowed values, or "
            f"switch Config.ICON_FLICKER_MODE to 'cosine' (which allows any "
            f"frequency)."
        )


_check_no_duplicate_frequencies()
_check_square_mode_frequencies()

# Frequency -> direction (the inverse of FREQUENCIES), used by Main.py to
# turn an incoming BCI_FREQ reading back into 'N'/'E'/'S'/'W'.
FREQ_TO_DIR = {freq: d for d, freq in FREQUENCIES.items()}

# Same frequencies as a numpy array, for fast "nearest frequency" lookups
# in Main.py (np.argmin(np.abs(TARGET_FREQS - measured_freq))).
TARGET_FREQS = np.array(list(FREQUENCIES.values()), dtype=float)

# Per-frequency confidence threshold, used by BrainWavesVisualizer.py to
# decide whether a detection is trustworthy enough to publish.
FREQ_THRESHOLDS = {FREQUENCIES[d]: cfg["threshold"] for d, cfg in DIRECTIONS.items()}
CONSECUTIVE_REQUIRED_DEFAULT = 1  # default number of consecutive detections required before streaming

# --- Legacy per-frequency names -------------------------------------
# BrainWavesEmulator.py, BrainWavesVisualizer.py and FrequencyAnalysis.py
# import these four names directly. Keep them in sync with
# DIRECTIONS/FREQUENCIES above rather than editing these values by hand.
freq1 = FREQUENCIES["S"]  # ≈ 4.2857 Hz
freq2 = FREQUENCIES["N"]  # 10.0 Hz
freq3 = FREQUENCIES["W"]  # 6.0 Hz
freq4 = FREQUENCIES["E"]  # 15.0 Hz

# --- Frequency analysis window lengths (used by FrequencyAnalysis.py / BrainWavesVisualizer.py) ---
welch_window_s = 4.0
psd_window_s = 4.0
cca_window_s = 4.0