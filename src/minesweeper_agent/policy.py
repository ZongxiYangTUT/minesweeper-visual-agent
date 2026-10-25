"""Decision policy: preserve proof boundaries while reducing guesses and inputs."""

from __future__ import annotations

from math import prod

from .model import Board, Settings, Move, COVERED, FLAG, UNKNOWN, MINE
from .solver import analyze
from .endgame import exact_endgame
from .lookahead import progress_choice


def _blank_score(board, cell, probabilities, known_mines):
    neighbors = board.neighbors(*cell)
    if any(n in known_mines or board.cells[n[0]][n[1]] == FLAG for n in neighbors):
        return 0.0
    return prod(
        1 - probabilities.get(n, 0.5) for n in neighbors if board.cells[n[0]][n[1]] == COVERED
    )


def _safe_action(board, result):
    best = None
    gain = 0.0
    for cell in sorted(result.safe):
        hidden = sum(board.cells[r][c] == COVERED for r, c in board.neighbors(*cell))
        expected = 1 + hidden * _blank_score(board, cell, result.probabilities, result.mines)
        if expected > gain:
            best = Move("reveal", *cell, "Proven safe; prioritize opening a blank region")
            gain = expected
    # A flag is useful only as preparation for a chord that saves inputs.
    # The budget includes every required flag; a chord is not counted as free.
    for row in range(board.rows):
        for col in range(board.cols):
            clue = board.cells[row][col]
            if not 1 <= clue <= 8:
                continue
            neighbors = board.neighbors(row, col)
            covered = {n for n in neighbors if board.cells[n[0]][n[1]] == COVERED}
            safe = covered & result.safe
            mines = covered & result.mines
            flags = sum(board.cells[r][c] == FLAG for r, c in neighbors)
            if not safe or covered != safe | mines or flags + len(mines) != clue:
                continue
            efficiency = len(safe) / (len(mines) + 1)
            if efficiency > 1 and efficiency > gain:
                gain = efficiency
                if mines:
                    best = Move(
                        "flag",
                        *min(mines),
                        f"Proven mine; prepare chord opening {len(safe)} safe cells",
                    )
                else:
                    best = Move(
                        "chord",
                        row,
                        col,
                        f"Verified flags; open {len(safe)} proven safe neighbours together",
                    )
    return best


def choose_move(board: Board, settings: Settings, allow_guess: bool):
    if board.locations(UNKNOWN):
        return None, None, "unknown"
    if board.locations(MINE):
        return None, None, "lost"
    revealed = sum(v >= 0 for row in board.cells for v in row)
    if revealed == board.rows * board.cols - settings.total_mines:
        return None, None, "won"
    covered = board.locations(COVERED)
    if len(covered) == board.rows * board.cols:
        return (
            Move("reveal", 0, 0, "First-click-safe opening in classic XP Minesweeper"),
            None,
            "opening",
        )
    result = analyze(board, settings.total_mines, max_component_cells=96)
    if result.errors:
        raise ValueError("; ".join(result.errors))
    if result.safe:
        move = _safe_action(board, result)
        return move, 1.0 if move.kind == "flag" else 0.0, "proven"
    if allow_guess and covered <= result.probabilities.keys():
        endgame = exact_endgame(board, settings.total_mines, result)
        if endgame:
            return (
                Move(
                    "reveal",
                    *endgame.cell,
                    f"Exact endgame: {endgame.layouts} layouts, completion chance {endgame.win_probability:.1%}",
                ),
                endgame.mine_probability,
                "guess",
            )

        def rank(c):
            return (
                result.probabilities[c],
                -_blank_score(board, c, result.probabilities, result.mines),
                c,
            )

        progress = progress_choice(board, settings.total_mines, result, rank)
        if progress:
            return (
                Move(
                    "reveal",
                    *progress.cell,
                    f"One-step lookahead: {progress.progress_probability:.1%} chance of a forced follow-up",
                ),
                progress.mine_probability,
                "guess",
            )
        cell = min(covered, key=rank)
        return (
            Move("reveal", *cell, "Lowest complete mine probability; no cosmetic flags"),
            result.probabilities[cell],
            "guess",
        )
    # Marking an already inferred mine does not create information. In a capped
    # search or no-guess run, stop rather than spending inputs on cosmetic flags.
    return None, None, "stalled"
