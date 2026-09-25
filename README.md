# Lab 5 — Brain-Computer Interfaces (IUI/HCI, Maastricht University)

This project implements a **maze game** controlled by **SSVEP-style BCI input** delivered over **Lab Streaming Layer (LSL)**. The goal is to study **usability principles** for BCI user interfaces in a controlled maze-navigation task.

The system consists of three parts:

1) **EEG acquisition** from a Unicorn Hybrid Black headset → `unicorn2lsl.py` publishes an 8‑channel EEG LSL stream named `Unicorn`.
2) **SSVEP analysis & visualization** → `BrainWavesVisualizer.py` subscribes to `Unicorn`, computes a best‑matching flicker frequency (CCA + PSD) and publishes a **single‑channel LSL stream** `BCI_FREQ` (type `BCI`) with the predicted frequency in Hz.
3) **Maze game** → `Main.py` subscribes to `BCI_FREQ`, snaps to the nearest target frequency and moves in the corresponding direction.

---

## How control works

The game UI shows four flickering cues (square‑wave, phase‑locked to the display refresh). For a 60 Hz display the targets are:

- `60/14 = 4.3 Hz` → **North**
- `60/10 = 6.0 Hz` → **West**
- `60/6 ≈ 10 Hz` → **South**
- `60/4 = 15.0 Hz` → **East**

`Main.py` reads the current frequency from the **`BCI_FREQ`** LSL stream and maps the nearest target to a direction. The maze is loaded from an ASCII file (default: `src/mazes/level2.txt`).

> If you use a non‑60 Hz monitor, adjust the target set in the config file so that each target has an **integer frame period** at your refresh rate with no harmonics.

---

## Installation

Use the provided requirements file:

```bash
python -m pip install -r src/requirements.txt
```

This installs: `numpy`, `pygame`, `pylsl`, `pyserial`, `scikit-learn`, `scipy` (versions pinned in the file).

---

## Running

### A) Demo without EEG (keyboard → LSL emulator)

In one terminal, start the frequency emitter (arrow keys → `BCI_FREQ` LSL stream):

```bash
python src/BrainWavesEmulator.py
```

Then start the game in a second terminal:

```bash
python src/Main.py
```

**Emulator controls** (shown in its window): Up/Left/Down/Right send 10/12/≈8.57/15 Hz respectively; `Esc` / `Q` to quit.

---

### B) With Unicorn Hybrid Black EEG (real BCI)

Run the following **in order**:

1) **Unicorn → LSL (EEG):**

```bash
# Edit the serial device in the script if needed (default is COM5 on Windows), see setup instructions for more details
python src/unicorn2lsl.py
```

- Publishes 8‑channel EEG at 250 Hz, LSL stream **name** `Unicorn`, **type** `EEG`.

2) **SSVEP analysis & dashboard:**

```bash
python src/BrainWavesVisualizer.py
```

- Subscribes to LSL **name** `Unicorn` (type `EEG`).
- Live plots: 8 EEG traces, Welch PSD, and CCA bars for the target frequencies.
- Publishes the predicted frequency to **`BCI_FREQ`** (type `BCI`).

**Visualizer controls**
- `arrow up` / `arrow down` : increase/decrease amplitude range by **1 µV**
- `R` : recenter baselines
- `←` / `→` : shorten/lengthen rolling window (**±1 s**, 2–60 s)
- `Space` : pause/resume
- `Q` / `Esc` : quit

3) **Maze game (LSL consumer):**

```bash
python src/Main.py
```

- Subscribes to **`BCI_FREQ`** (type `BCI`, 1 channel float32). If no stream is found shortly after startup, the game exits with a message.

---

## Configuration notes

- **Interesting parameters:** `src/Config.py` defines most constants and parameters used in this project. Some more are defined locally. These should in general not be modified to ensure the smooth running of this lab, so proceed with caution!
- **Serial port (Unicorn):** `src/Config.py` defines the port used by `src/unicorn2lsl.py` and defaults to `COM5`. Update to your OS device path.
- **Maze file:** change the `path = "mazes/..."` line in `src/Main.py`. Provided mazes: `level0.txt`, `level1.txt`, `level2.txt`, `level3.txt`.
- **Refresh rate:** ensure display runs at 60 Hz (or update target frequencies to match your refresh).

---

## Code hierarchy

```
src/
    Controller.py          # Player movement, timing, integration with UI
    FlashableIcon.py       # Frame-locked flicker generator for cues
    Maze.py                # ASCII maze loader + traversal helpers
    UI.py                  # Sidebar, cues, HUD, on-screen text and metrics
    FrequencyAnalysis.py   # Welch PSD, bandpower, reference signals, CCA classifier helpers
    unicorn2lsl.py         # Unicorn Hybrid Black → LSL (EEG), via pyserial + pylsl
    Config.py                # Global and configuration constants
    BrainWavesEmulator.py    # Arrow keys → LSL frequency emitter (stream 'BCI_FREQ', type 'BCI')
    BrainWavesVisualizer.py  # EEG dashboard + SSVEP → publishes 'BCI_FREQ' for the game
    Main.py                  # Game entry point; subscribes to 'BCI_FREQ' and runs the maze
    requirements.txt         # Project dependencies
    
    mazes/
        level0.txt
        level1.txt
        level2.txt
        level3.txt
```

---

## License

- `src/unicorn2lsl.py` contains a GPLv3 header (Robert Oostenveld; adapted for this course). Ensure license compatibility if you redistribute. Other files are provided for educational use within the IUI/HCI course.

## Authors

This project was developed as part of **Lab 5 – Brain-Computer Interfaces** for the **IUI and HCI courses at Maastricht University**.

**Contributors**
- Adrien Bersia — Maastricht University
- Meike Thijsen — Maastricht University
- Course Instructor: Konstantia Zarkogianni, Yusuf Can Semerci
