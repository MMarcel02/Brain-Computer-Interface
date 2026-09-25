"""
FlashableIcon.py:  makes an arrow "flicker" at a target frequency by
telling the renderer, for a given frame number, how bright the icon
should be right now (0.0 = fully off/dark, 1.0 = fully on/light).

Two flicker styles ("modes") are available, set which one the game uses
with Config.ICON_FLICKER_MODE:

  "square" (default): the icon snaps between fully on and fully off,
      like a classic SSVEP stimulus, at EXACTLY the frequency named in
      Config.DIRECTIONS. Only frequencies in Config.SQUARE_MODE_ALLOWED_HZ
      are allowed here,Config.py checks this at startup, because each
      one was picked so REFRESH_HZ / hz comes out to a whole, EVEN number
      of frames, which is what lets the on/off halves split evenly.

  "cosine" — brightness follows a smooth cosine wave instead of snapping
      on/off. Because this is computed straight from time
      (frame_idx / refresh_hz) rather than a whole-frame period, it
      reproduces ANY frequency exactly, with no rounding error at all and
      no restriction on which values you can use. Good choice if you want
      a frequency that isn't in Config.SQUARE_MODE_ALLOWED_HZ, and it's
      gentler to look at. Trade-off: FREQ_THRESHOLDS in Config.py were
      tuned assuming a square-wave flicker, so switching to cosine may
      need re-tuning those thresholds for reliable detection.

# --- DO NOT EDIT the "square" mode math -----------------------------------
# The flicker must be locked in phase to the display's actual refresh rate
# for the SSVEP/BCI detection to work at all (see the "Arrows & BCI tuning"
# section of Config.py for why). `period_frames` must be a whole number of
# frames. A fractional flicker period is not something a renderer that draws
# frame by frame can reproduce accurately in "square" mode. Because "square"
# mode only ever receives an hz value from Config.SQUARE_MODE_ALLOWED_HZ
# (Config checks this at startup and refuses to run otherwise), `refresh_hz
# / freq_hz` always comes out extremely close to a whole, even number of
# frames. The round() call below is just a safety net for floating point
# noise (for example, 60/4.29 = 13.986013986... rounds correctly to 14),
# not something papering over a real fraction. If you want a different
# flicker frequency, that's expected. Add it to SQUARE_MODE_ALLOWED_HZ in
# Config.py (only if REFRESH_HZ divided by your new value is a whole, even
# number of frames) and use it in Config.DIRECTIONS, rather than editing
# the math here. If you want a frequency that doesn't divide evenly, use
# "cosine" mode instead. It sidesteps frame quantization entirely.
# ---------------------------------------------------------------------------
"""
import math


class FlashableIcon:
    """
    Frame-locked flicker generator for one on-screen cue (arrow).

    Call `luminance(frame_idx)` once per rendered frame; it returns a
    brightness from 0.0 ("off"/dark) to 1.0 ("on"/light), flickering at
    exactly `freq_hz`, assuming frames are drawn at `refresh_hz` frames
    per second.

    `mode`:
      "square" (default) - hard on/off, matches classic SSVEP stimuli.
                            `luminance` only ever returns exactly 0.0 or 1.0.
      "cosine"           - smooth brightness, exact at any frequency.
                            `luminance` returns every value in between too.
    """

    def __init__(self, freq_hz, refresh_hz=60, mode="square"):
        if mode not in ("square", "cosine"):
            raise ValueError(f"Unknown FlashableIcon mode: {mode!r} (expected 'square' or 'cosine')")
        self.f = float(freq_hz)
        self.rf = float(refresh_hz)
        self.mode = mode

        if mode == "square":
            # Number of frames in one full on/off cycle, i.e. how many
            # frames it takes to complete exactly 1/freq_hz seconds. See
            # the DO NOT EDIT note above for why this comes out to (very
            # nearly) a whole, even number automatically as long as
            # freq_hz is one of Config.SQUARE_MODE_ALLOWED_HZ.
            period_float = self.rf / self.f
            self.period_frames = int(round(period_float))

    def luminance(self, frame_idx: int) -> float:
        """Return a brightness from 0.0 ("off") to 1.0 ("on") for the given frame number."""
        if self.mode == "cosine":
            t = frame_idx / self.rf
            return 0.5 * (1.0 + math.cos(2 * math.pi * self.f * t))

        # "square" mode
        k = frame_idx % self.period_frames
        return 1.0 if k < (self.period_frames // 2) else 0.0