"""
Maze.py: loads an ASCII maze from a text file and answers simple questions
about it ("is this cell a wall?", "which neighbours can I walk to?").

The maze file uses these characters:
    '#'  wall
    ' '  open path
    'S'  start cell (exactly one)
    'G'  goal cell (exactly one)

This file does not know anything about the BCI/EEG side of the project,
it is plain grid logic and safe to read, extend, or restyle (e.g. to add
new tile types) as long as you keep `is_path` in sync with any new symbols
you introduce.
"""
from Config import DIRS, VEC


def read_ascii_maze(path: str):
    """
    Read a maze file and return a list of equal-length strings (one per row).

    Blank lines are dropped and every row is padded with spaces on the
    right so all rows have the same width (this keeps row/col indexing
    simple everywhere else in the code).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = [line.rstrip("\n") for line in f]
    lines = [ln for ln in raw if ln.strip() != ""]
    width = max(len(ln) for ln in lines)
    lines = [ln.ljust(width) for ln in lines]
    return lines


class Maze:
    """
    A minimal grid maze model.

    Positions are (row, col) tuples, with row 0 at the top and col 0 on the
    left, matching how the ASCII file reads on screen. `Config.VEC` maps
    each direction letter to the (row_delta, col_delta) used to move in
    that direction, so this class and Controller.py always agree on what
    "moving north" means.
    """

    def __init__(self, lines):
        self.lines = [list(row) for row in lines]
        self.rows = len(self.lines)
        self.cols = len(self.lines[0]) if self.rows else 0
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, ch):
        """Return the (row, col) of the first cell containing `ch`, or None."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.lines[r][c] == ch:
                    return (r, c)
        return None

    def in_bounds(self, rc):
        """True if (row, col) is inside the grid."""
        r, c = rc
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_path(self, rc):
        """True if (row, col) is a cell the player can stand on (not a wall)."""
        if not self.in_bounds(rc):
            return False
        ch = self.lines[rc[0]][rc[1]]
        return ch in (" ", "S", "G")

    def neighbors(self, rc):
        """List of (neighbour_rc, direction) pairs for every walkable neighbour of rc."""
        r, c = rc
        out = []
        for d in DIRS:
            dr, dc = VEC[d]
            rc2 = (r + dr, c + dc)
            if self.is_path(rc2):
                out.append((rc2, d))
        return out