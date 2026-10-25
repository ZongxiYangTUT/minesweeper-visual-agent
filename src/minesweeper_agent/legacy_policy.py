"""Frozen v0.3 policy from commit 5ac0693, used only as an evaluation baseline."""
from math import prod
from .model import Board, Settings, Move, UNKNOWN, MINE, COVERED, FLAG
from .solver import analyze

def choose_move(board: Board, settings: Settings, allow_guess: bool) -> tuple[Move | None, float | None, str]:
    if board.locations(UNKNOWN):
        return None, None, "unknown"
    if board.locations(MINE):
        return None, None, "lost"
    revealed = sum(0 <= v <= 8 for row in board.cells for v in row)
    if revealed == board.rows * board.cols - settings.total_mines:
        return None, None, "won"
    if len(board.locations(COVERED)) == board.rows * board.cols:
        # XP moves a mine away from the first clicked cell. Later guesses are opt-in.
        return Move("reveal", 0, 0, "Opening click on a fresh classic XP board"), None, "opening"
    result = analyze(board, settings.total_mines, max_component_cells=96)
    if result.errors:
        raise ValueError("; ".join(result.errors))
    forced = result.next_move()
    if forced:
        return forced, 0.0 if forced.kind == "reveal" else 1.0, "proven"
    if allow_guess:
        covered = board.locations(COVERED)
        # Do not rank incomplete probabilities from a skipped frontier.
        if covered and covered <= result.probabilities.keys():
            def rank(cell):
                # An independence estimate is only a tie-breaker among equal
                # exact risks. It is never used as proof or reported as mine risk.
                neighbors = board.neighbors(*cell)
                blank_score = prod(0.0 if board.cells[r][c] == FLAG else
                                   1 - result.probabilities.get((r, c), 0.0) for r, c in neighbors)
                return result.probabilities[cell], -blank_score, cell
            cell = min(covered, key=rank)
            p = result.probabilities[cell]
            return Move("reveal", *cell, "Lowest mine probability among all covered cells"), p, "guess"
    return None, None, "stalled"
