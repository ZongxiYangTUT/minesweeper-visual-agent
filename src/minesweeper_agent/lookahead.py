"""One-observation lookahead for non-endgames, with exact outcome weights.

The progress bonus is a heuristic, not a claim of optimal win probability.
Only fully counted hypothetical outcomes contribute to a candidate's score.
"""

from dataclasses import dataclass

from .model import Board, COVERED, FLAG
from .solver import analyze


@dataclass(frozen=True)
class ProgressChoice:
    cell: tuple[int, int]
    mine_probability: float
    progress_probability: float  # joint probability of survival and a forced next reveal
    score: float


def progress_choice(board, total_mines, analysis, rank, max_candidates=6):
    if not analysis.layout_count:
        return None
    covered = board.locations(COVERED)
    ordered = sorted(covered, key=rank)
    minimum = analysis.probabilities[ordered[0]]
    candidates = []
    signatures = set()
    for cell in ordered:
        if analysis.probabilities[cell] > minimum + 0.03 + 1e-12:
            continue
        r, c = cell
        # Translation-equivalent immediate neighbourhoods are a shortlist
        # heuristic only; probabilities still come from the whole board.
        signature = tuple(
            board.cells[nr][nc] if 0 <= nr < board.rows and 0 <= nc < board.cols else -9
            for nr in range(r - 1, r + 2)
            for nc in range(c - 1, c + 2)
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        candidates.append(cell)
        if len(candidates) >= max_candidates:
            break
    if len(candidates) < 2:
        return None
    best = None
    fallback_evaluated = False
    for cell in candidates:
        fixed = sum(
            n in analysis.mines or board.cells[n[0]][n[1]] == FLAG for n in board.neighbors(*cell)
        )
        uncertain = sum(
            board.cells[n[0]][n[1]] == COVERED and n not in analysis.mines
            for n in board.neighbors(*cell)
        )
        survivor = progress = zero = 0
        complete = True
        for clue in range(fixed, fixed + uncertain + 1):
            raw = [list(row) for row in board.cells]
            raw[cell[0]][cell[1]] = clue
            outcome = analyze(
                Board(tuple(map(tuple, raw))),
                total_mines,
                max_component_cells=96,
                max_search_nodes=10_000,
            )
            if outcome.errors:
                continue
            if outcome.layout_count is None:
                complete = False
                break
            ways = outcome.layout_count
            survivor += ways
            if outcome.safe:
                progress += ways
            if clue == 0:
                zero += ways
        if not complete or not survivor:
            continue
        total = analysis.layout_count
        risk = analysis.probabilities[cell]
        if abs(survivor / total - (1 - risk)) > 1e-10:
            continue
        fallback_evaluated |= cell == ordered[0]
        score = (1 - risk) + 0.08 * progress / total + 0.02 * zero / total
        candidate = ProgressChoice(cell, risk, progress / total, score)
        if best is None or (candidate.score, -candidate.mine_probability) > (
            best.score,
            -best.mine_probability,
        ):
            best = candidate
    # An incomplete shortlist cannot justify switching away from the minimum
    # risk fallback unless that fallback was fully evaluated too.
    if best is None or not fallback_evaluated:
        return None
    return best
