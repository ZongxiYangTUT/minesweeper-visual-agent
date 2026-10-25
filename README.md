# Minesweeper Visual Agent

A screen-only classic Minesweeper agent for Windows. This core provides board calibration, template annotation, read-only inspection and explicitly confirmed single-step mouse control.

## Setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Keep your game visible on the primary monitor. Use `minesweeper-agent calibrate --rows 9 --cols 9 --mines 10` to select its grid, then `annotate` to collect cell templates. Use `inspect` to verify recognition, `step --dry-run` to inspect a proposed move, and `step` to execute one move after typing `yes`.

The solver combines clue propagation, subset subtraction, bounded grouped enumeration and global mine-count weighting. Unknown cells, inconsistent clues and visible mines prevent actions. Incomplete enumeration counts are discarded. Local environment, screenshots and templates are ignored by Git.
