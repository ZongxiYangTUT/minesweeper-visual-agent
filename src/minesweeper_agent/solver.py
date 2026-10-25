"""Exact Minesweeper constraints, subset propagation and grouped enumeration.

Proof uses integer layout counts. Bounded searches discard partial counts;
the caller may separately opt into guesses from complete probabilities.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import comb
import time

from .model import Board, Cell, COVERED, FLAG, MINE, UNKNOWN, Move


@dataclass
class Analysis:
    safe: set[Cell] = field(default_factory=set)
    mines: set[Cell] = field(default_factory=set)
    probabilities: dict[Cell, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def next_move(self) -> Move | None:
        if self.errors:
            return None
        if self.safe:
            r, c = sorted(self.safe)[0]
            return Move("reveal", r, c, "All valid mine layouts prove this cell is safe")
        if self.mines:
            r, c = sorted(self.mines)[0]
            return Move("flag", r, c, "All valid mine layouts place a mine here")
        return None


@dataclass
class Component:
    cells: list[Cell]
    # Number of valid assignments indexed by number of mines in this component.
    counts: dict[int, int]
    # For each cell, valid assignments where it is a mine, indexed by mine count.
    mine_counts: dict[Cell, dict[int, int]]


class SearchLimit(Exception):
    pass


@dataclass
class SearchBudget:
    remaining: int
    deadline: float

    def visit(self):
        self.remaining -= 1
        if self.remaining < 0 or (self.remaining % 256 == 0 and time.monotonic() > self.deadline):
            raise SearchLimit("Exact search reached its node/time budget; incomplete counts discarded.")


def _enumerate_component(cells: list[Cell],
                         constraints: list[tuple[frozenset[Cell], int]],
                         budget: SearchBudget | None = None) -> Component:
    # Cells belonging to exactly the same clues are exchangeable. Enumerate a
    # group's mine count rather than all 2**n labelled placements, retaining
    # binomial weights so every individual layout is still counted exactly.
    equivalent: dict[tuple[int, ...], list[Cell]] = defaultdict(list)
    for cell in sorted(cells):
        equivalent[tuple(i for i, (group, _) in enumerate(constraints) if cell in group)].append(cell)
    ordered = sorted(equivalent.items(), key=lambda item: (-len(item[0]), item[1][0]))
    groups = [set(group) for group, _ in constraints]
    targets = [target for _, target in constraints]
    used = [0] * len(groups)
    remaining = [len(group) for group in groups]
    memberships = [member for member, _ in ordered]
    sizes = [len(group) for _, group in ordered]
    assignment = [0] * len(ordered)
    counts: dict[int, int] = defaultdict(int)
    mine_counts: dict[Cell, dict[int, int]] = {cell: defaultdict(int) for cell in cells}

    def walk(index: int, mines: int, weight: int) -> None:
        if budget:
            budget.visit()
        if index == len(ordered):
            counts[mines] += weight
            for position, number in enumerate(assignment):
                if number:
                    per_cell = weight * number // sizes[position]
                    for cell in ordered[position][1]:
                        mine_counts[cell][mines] += per_cell
            return
        size = sizes[index]
        low, high = 0, size
        for ci in memberships[index]:
            low = max(low, targets[ci] - used[ci] - remaining[ci] + size)
            high = min(high, targets[ci] - used[ci])
        for value in range(low, high + 1):
            assignment[index] = value
            valid = True
            for ci in memberships[index]:
                used[ci] += value
                remaining[ci] -= size
                if used[ci] > targets[ci] or used[ci] + remaining[ci] < targets[ci]:
                    valid = False
            if valid:
                walk(index + 1, mines + value, weight * comb(size, value))
            for ci in memberships[index]:
                used[ci] -= value
                remaining[ci] += size

    walk(0, 0, 1)
    return Component(sorted(cells), dict(counts), {cell: dict(v) for cell, v in mine_counts.items()})


def _propagate(constraints, covered, remaining_mines):
    """Direct equalities + subset subtraction, including global count extremes."""
    known: dict[Cell, int] = {}
    while True:
        reduced = {}
        for cells, target in constraints:
            pending = frozenset(cells.difference(known))
            target -= sum(known[cell] for cell in cells if cell in known)
            if not 0 <= target <= len(pending):
                raise ValueError("Clues conflict after propagating forced moves.")
            if pending:
                if pending in reduced and reduced[pending] != target:
                    raise ValueError("The same covered cells have conflicting mine counts.")
                reduced[pending] = target
        before = len(known)
        for cells, target in reduced.items():
            if target in (0, len(cells)):
                value = int(target != 0)
                for cell in cells:
                    if cell in known and known[cell] != value:
                        raise ValueError("Clues force a cell to be both safe and mined.")
                    known[cell] = value
        if remaining_mines is not None:
            pending = covered.difference(known)
            needed = remaining_mines - sum(known.values())
            if not 0 <= needed <= len(pending):
                raise ValueError("Total mine count conflicts with forced mine positions.")
            if pending and needed in (0, len(pending)):
                known.update({cell: int(needed != 0) for cell in pending})
        if len(known) != before:
            constraints = list(reduced.items())
            continue

        # Index possible supersets by shared cells instead of comparing all pairs.
        memberships = defaultdict(set)
        for cells in reduced:
            for cell in cells:
                memberships[cell].add(cells)
        additions = {}
        for small, target in reduced.items():
            candidates = set.intersection(*(memberships[cell] for cell in small))
            for large in candidates:
                if small == large:
                    continue
                difference = large - small
                mines = reduced[large] - target
                if not 0 <= mines <= len(difference):
                    raise ValueError("Subset clues are contradictory.")
                existing = reduced.get(difference, additions.get(difference))
                if existing is not None and existing != mines:
                    raise ValueError("Subset deductions give inconsistent mine counts.")
                # Bound redundant derived constraints; original clues are retained.
                if existing is None and len(reduced) + len(additions) < 2048:
                    additions[difference] = mines
        if not additions:
            return known, list(reduced.items())
        constraints = list(reduced.items()) + list(additions.items())


def _convolve(left: dict[int, int], right: dict[int, int]) -> dict[int, int]:
    result: dict[int, int] = defaultdict(int)
    for a, count_a in left.items():
        for b, count_b in right.items():
            result[a + b] += count_a * count_b
    return dict(result)


def _ways_with_outside(counts: dict[int, int], required: int, free_count: int) -> int:
    return sum(count * comb(free_count, required - k)
               for k, count in counts.items() if 0 <= required - k <= free_count)


def analyze(board: Board, total_mines: int | None = None,
            max_component_cells: int = 18, max_search_nodes: int = 500_000,
            max_search_seconds: float = 1.5) -> Analysis:
    result = Analysis()
    if board.locations(UNKNOWN):
        result.errors.append("Unrecognized cells exist. Teach templates before executing any click.")
        return result
    if board.locations(MINE):
        result.errors.append("A visible mine/end screen was detected. No further action is safe.")
        return result

    covered = board.locations(COVERED)
    flags = board.locations(FLAG)
    if total_mines is not None and (total_mines < len(flags) or total_mines > len(flags) + len(covered)):
        result.errors.append("Total mine count is inconsistent with observed flags/covered cells.")
        return result

    constraints: list[tuple[frozenset[Cell], int]] = []
    for row in range(board.rows):
        for col in range(board.cols):
            clue = board.cells[row][col]
            if not 0 <= clue <= 8:
                continue
            neighbors = board.neighbors(row, col)
            unknown_neighbors = frozenset(cell for cell in neighbors if cell in covered)
            target = clue - sum(cell in flags for cell in neighbors)
            if not 0 <= target <= len(unknown_neighbors):
                result.errors.append(f"Contradictory clue {clue} at ({row},{col}); check flags/vision.")
                return result
            if unknown_neighbors:
                constraints.append((unknown_neighbors, target))

    try:
        known, reduced = _propagate(constraints, covered,
                                   None if total_mines is None else total_mines - len(flags))
    except ValueError as exc:
        result.errors.append(str(exc))
        return result

    result.safe.update(cell for cell, val in known.items() if val == 0)
    result.mines.update(cell for cell, val in known.items() if val == 1)
    result.probabilities.update({cell: float(val) for cell, val in known.items()})

    frontier = set().union(*(cells for cells, _ in reduced)) if reduced else set()
    unassigned = covered.difference(known)
    outside = unassigned.difference(frontier)

    adjacency: dict[Cell, set[Cell]] = {cell: set() for cell in frontier}
    for group, _ in reduced:
        for cell in group:
            adjacency[cell].update(group)
    components: list[Component] = []
    budget = SearchBudget(max_search_nodes, time.monotonic() + max_search_seconds)
    while adjacency:
        start = next(iter(adjacency))
        group = {start}
        stack = [start]
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor not in group:
                    group.add(neighbor)
                    stack.append(neighbor)
        for cell in group:
            del adjacency[cell]
        if len(group) > max_component_cells:
            result.notes.append(f"A {len(group)}-cell frontier exceeds enumeration cap "
                                f"({max_component_cells}); only direct proof is used.")
            return result
        relevant = [(cells, target) for cells, target in reduced if cells & group]
        try:
            component = _enumerate_component(sorted(group), relevant, budget)
        except SearchLimit as exc:
            result.notes.append(str(exc))
            return result
        if not component.counts:
            result.errors.append("No valid mine layouts; annotations or flags may be wrong.")
            result.safe.clear()
            result.mines.clear()
            result.probabilities.clear()
            return result
        components.append(component)

    if total_mines is None:
        for component in components:
            denominator = sum(component.counts.values())
            for cell in component.cells:
                numerator = sum(component.mine_counts[cell].values())
                result.probabilities[cell] = numerator / denominator
                if numerator == 0:
                    result.safe.add(cell)
                elif numerator == denominator:
                    result.mines.add(cell)
        if outside:
            result.notes.append("Unconstrained cells have no probability without total mine count.")
        return result

    needed = total_mines - len(flags) - len(result.mines)
    if not 0 <= needed <= len(unassigned):
        result.errors.append("Total mine count conflicts with forced mine positions.")
        result.safe.clear()
        result.mines.clear()
        result.probabilities.clear()
        return result

    prefix = [{0: 1}]
    for component in components:
        prefix.append(_convolve(prefix[-1], component.counts))
    suffix = [{0: 1} for _ in range(len(components) + 1)]
    for i in range(len(components) - 1, -1, -1):
        suffix[i] = _convolve(components[i].counts, suffix[i + 1])

    denominator = _ways_with_outside(prefix[-1], needed, len(outside))
    if denominator == 0:
        result.errors.append("No valid layouts satisfy the stated total mine count.")
        result.safe.clear()
        result.mines.clear()
        result.probabilities.clear()
        return result

    for i, component in enumerate(components):
        other = _convolve(prefix[i], suffix[i + 1])
        for cell in component.cells:
            numerator = sum(mine_solutions * _ways_with_outside(other, needed - k,
                                                                 len(outside))
                            for k, mine_solutions in component.mine_counts[cell].items())
            result.probabilities[cell] = numerator / denominator
            if numerator == 0:
                result.safe.add(cell)
            elif numerator == denominator:
                result.mines.add(cell)

    if outside:
        numerator = _ways_with_outside(prefix[-1], needed - 1, len(outside) - 1)
        for cell in outside:
            result.probabilities[cell] = numerator / denominator
            if numerator == 0:
                result.safe.add(cell)
            elif numerator == denominator:
                result.mines.add(cell)
    return result
