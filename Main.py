"""
Main.py — the game's entry point.

What it does, in order:
  1. Waits for the 'BCI_FREQ' LSL stream (from BrainWavesVisualizer.py, or
     from BrainWavesEmulator.py if you're testing with the keyboard).
  2. Loads the maze from Config.MAZE_PATH.
  3. Opens a pygame window sized to fit the maze + a sidebar for the arrows.
  4. Runs the main loop: read input -> update player position -> redraw.

Run it with:  python Main.py
(after starting BrainWavesEmulator.py or the real BCI pipeline — see README.md)
"""
import pygame as pg
import numpy as np
import sys, time

from Maze import read_ascii_maze, Maze
from UI import UI
from Controller import Controller

from pylsl import resolve_byprop as lsl_resolve_byprop, StreamInlet as LSLInlet

from Config import TARGET_FREQS, FREQ_TO_DIR, MAZE_PATH, WINDOW_W, WINDOW_H, MIN_SIDEBAR_PX


def nearest_dir_from_freq(f):
    """Snap to nearest target frequency and return 'N','W','S','E' ('' if invalid)."""
    if f is None or not np.isfinite(f):
        return ""
    i = int(np.argmin(np.abs(TARGET_FREQS - float(f))))
    return FREQ_TO_DIR[TARGET_FREQS[i]]


# ---------------------------------------------------------------------------
# DO NOT EDIT: BCIListener talks to the LSL network stream that carries the
# detected frequency from the EEG pipeline (or the keyboard emulator). Its
# job is just "give me the latest direction, or '' if nothing new arrived" —
# the connection/retry/debounce details below are unrelated to game
# usability and easy to break by accident. If the game isn't picking up
# your input, check that BrainWavesVisualizer.py / BrainWavesEmulator.py is
# actually running first, before changing anything here.
# ---------------------------------------------------------------------------
class BCIListener:
    """
    Listens for LSL stream 'BCI_FREQ' (type 'BCI') with one float channel (Hz).
    Debounces with a short hold to avoid jitter.
    Exits the program if no LSL stream is found within 5 seconds.
    """
    def __init__(self, name="BCI_FREQ", stype="BCI", timeout=5.0):
        self.inlet = None
        self.name = name
        self.stype = stype
        self.last_freq = None
        self.last_dir  = ""
        # debounce state
        self._last_snap = ""
        self._stable_count = 0
        self._need = 2  # require 2 consecutive matches

        if LSLInlet is None or lsl_resolve_byprop is None:
            print("[BCI] pylsl not available; cannot continue.")
            sys.exit(1)

        print(f"[BCI] Waiting for LSL stream '{name}' (type '{stype}') ...")
        start = time.time()
        while self.inlet is None and time.time() - start < timeout:
            self._connect_once()
            if self.inlet is None:
                time.sleep(0.25)

        if self.inlet is None:
            print(f"[BCI] No LSL stream found after {timeout:.1f}s. Exiting.")
            sys.exit(1)

    def _connect_once(self):
        try:
            # IMPORTANT: use positional args only: (prop, value, minimum, timeout)
            cands = lsl_resolve_byprop('name', self.name, 1, 0.3)
            if not cands:
                cands = lsl_resolve_byprop('type', self.stype, 1, 0.3)
            if cands:
                self.inlet = LSLInlet(cands[0], max_buflen=3, max_chunklen=0, recover=True)
                self.inlet.flush()
                try:
                    nm = cands[0].name()
                except Exception:
                    nm = "<unknown>"
                print("[BCI] Connected to LSL stream:", nm)
        except Exception as e:
            print("[BCI] connect error:", e)

    def poll_direction(self) -> str:
        """Return snapped direction when stable; else ''. Non-blocking."""
        if self.inlet is None:
            return ""
        try:
            # Non-blocking read; drain to newest
            sample, ts = self.inlet.pull_sample(timeout=0.0)
            got = None
            while sample is not None:
                if sample and len(sample) >= 1:
                    got = float(sample[0])
                sample, ts = self.inlet.pull_sample(timeout=0.0)

            if got is None or not np.isfinite(got):
                return ""

            self.last_freq = got
            snap = nearest_dir_from_freq(got)
            return snap

        except Exception as e:
            print("[BCI] poll error:", e)
        return ""
# ---------------------------------------------------------------------------
# END DO NOT EDIT
# ---------------------------------------------------------------------------


def main():
    bci = BCIListener(name="BCI_FREQ", stype="BCI")  # listens for float Hz

    clock = pg.time.Clock()

    # --- load maze from file (change which maze loads via Config.MAZE_PATH) ---
    lines = read_ascii_maze(MAZE_PATH)
    maze = Maze(lines)

    # --- pygame / window ---
    pg.init()
    pg.display.set_caption("BCI Maze")

    # Create a normal resizable window (no SCALED to avoid logical scaling)
    flags = pg.RESIZABLE | pg.DOUBLEBUF
    surf = pg.display.set_mode((WINDOW_W, WINDOW_H), flags)

    # Maximize (windowed), if SDL2 helper is available
    try:
        from pygame._sdl2 import Window
        Window.from_display_module().maximize()
    except Exception:
        # Fallback: nearly full screen size without going fullscreen
        info = pg.display.Info()
        w = max(800, info.current_w - 80)
        h = max(600, info.current_h - 120)
        surf = pg.display.set_mode((w, h), flags)

    # Use the *actual drawable surface* size for all layout math
    screen_w, screen_h = surf.get_size()

    # Compute cell size to maximize maze height, while keeping at least
    # Config.MIN_SIDEBAR_PX of width for the arrow sidebar.
    cell_px_h = screen_h // maze.rows
    cell_px_w = max(1, (screen_w - MIN_SIDEBAR_PX) // maze.cols)
    cell_px = max(1, min(cell_px_h, cell_px_w))

    # recompute actual sidebar to fill remaining width exactly
    maze_w = maze.cols * cell_px
    sidebar_px = max(MIN_SIDEBAR_PX, screen_w - maze_w)

    # ---- setup UI and Controller objects ----
    ui = UI(surf, cell_px=cell_px, sidebar_px=sidebar_px)
    ctrl = Controller(maze, cell_px=cell_px, bci=bci)

    ui.frame_idx = 0
    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE:
                running = False

        ctrl.update(dt)

        # draw frame
        ui.draw(
            maze,
            ctrl.pos_rc,
            ctrl.armed_dir,
            steps=ctrl.step_count,
            elapsed_s=ctrl.elapsed_time
        )

        pg.display.flip()
        ui.frame_idx += 1

    # teardown
    pg.quit()


if __name__ == "__main__":
    main()