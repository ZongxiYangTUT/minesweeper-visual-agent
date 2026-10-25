"""Bounded exact game-tree search over layouts implied by visible clues only.

Maximize completion probability first; break ties by expected reveal inputs on
successful branches. Zero floods are modelled, but physical flag/chord setup is
handled separately by the policy. This is not a global minimum-click guarantee.
"""

from dataclasses import dataclass
from functools import lru_cache

from .model import COVERED, FLAG


class _Limit(Exception):
    pass


@dataclass(frozen=True)
class EndgameChoice:
    cell: tuple[int, int]
    mine_probability: float
    win_probability: float
    winning_reveals: float
    layouts: int


def exact_endgame(board, total_mines, analysis, max_cells=18, max_layouts=512, max_work=250_000):
    count = analysis.layout_count
    if not count or count > max_layouts or analysis.safe:
        return None
    fixed = board.locations(FLAG) | analysis.mines
    cells = sorted(board.locations(COVERED) - analysis.mines)
    if not 1 <= len(cells) <= max_cells:
        return None
    index = {cell: i for i, cell in enumerate(cells)}
    constraints = []
    for r in range(board.rows):
        for c in range(board.cols):
            clue = board.cells[r][c]
            if clue < 0:
                continue
            neighbors = board.neighbors(r, c)
            variables = tuple(index[n] for n in neighbors if n in index)
            target = clue - sum(n in fixed for n in neighbors)
            if variables:
                constraints.append((variables, target))
    constraints.append((tuple(range(len(cells))), total_mines - len(fixed)))
    members = [
        [j for j, (group, _) in enumerate(constraints) if i in group] for i in range(len(cells))
    ]
    order = sorted(range(len(cells)), key=lambda i: (-len(members[i]), i))
    targets = [t for _, t in constraints]
    remaining = [len(g) for g, _ in constraints]
    used = [0] * len(constraints)
    layouts = []
    work = [max_work]

    def spend(amount=1):
        work[0] -= amount
        if work[0] < 0:
            raise _Limit

    def enumerate_visible(depth, mask):
        spend()
        if depth == len(order):
            layouts.append(mask)
            if len(layouts) > max_layouts:
                raise _Limit
            return
        i = order[depth]
        for mine in (0, 1):
            valid = True
            for j in members[i]:
                used[j] += mine
                remaining[j] -= 1
                if not used[j] <= targets[j] <= used[j] + remaining[j]:
                    valid = False
            if valid:
                enumerate_visible(depth + 1, mask | ((1 << i) if mine else 0))
            for j in members[i]:
                used[j] -= mine
                remaining[j] += 1

    def bits(mask):
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    try:
        enumerate_visible(0, 0)
        if len(layouts) != count:
            return None  # incomplete/unsupported posterior is never called exact
        neighbor_masks = []
        fixed_clues = []
        for cell in cells:
            neighbors = board.neighbors(*cell)
            neighbor_masks.append(sum(1 << index[n] for n in neighbors if n in index))
            fixed_clues.append(sum(n in fixed for n in neighbors))
        clues = []
        floods = []
        for layout in layouts:
            values = tuple(
                (
                    -1
                    if layout & (1 << i)
                    else fixed_clues[i] + (layout & neighbor_masks[i]).bit_count()
                )
                for i in range(len(cells))
            )
            clues.append(values)
            expansion = []
            for i in range(len(cells)):
                seen = 1 << i
                pending = [i]
                while pending:
                    j = pending.pop()
                    if values[j] != 0:
                        continue
                    extra = neighbor_masks[j] & ~seen
                    seen |= extra
                    pending.extend(bits(extra))
                expansion.append(seen)
            floods.append(expansion)

        def deterministic_cost(mask, layout_id):
            safe = mask & ~layouts[layout_id]
            steps = 0
            for i in list(bits(safe)):
                if safe & (1 << i) and clues[layout_id][i] == 0:
                    steps += 1
                    safe &= ~floods[layout_id][i]
            return steps + safe.bit_count()

        @lru_cache(maxsize=None)
        def search(mask, ids):
            spend()
            if all((layouts[l] & mask) == (layouts[ids[0]] & mask) for l in ids):
                return len(ids), len(ids) * deterministic_cost(mask, ids[0]), None
            options = []
            for i in bits(mask):
                safe_ids = tuple(l for l in ids if not layouts[l] & (1 << i))
                if safe_ids:
                    options.append((-len(safe_ids), i, safe_ids))
            best_wins, best_cost, best_cell = -1, 0, None
            for negative_survival, i, safe_ids in sorted(options):
                if -negative_survival < best_wins:
                    continue
                groups = {}
                for l in safe_ids:
                    opened = floods[l][i] & mask
                    signature = tuple((j, clues[l][j]) for j in bits(opened))
                    spend(len(signature) + 1)
                    groups.setdefault((mask & ~opened, signature), []).append(l)
                wins = cost = 0
                for (new_mask, _), group in groups.items():
                    child_wins, child_cost, _ = search(new_mask, tuple(group))
                    wins += child_wins
                    cost += child_cost + child_wins
                if wins > best_wins or (wins == best_wins and cost < best_cost):
                    best_wins, best_cost, best_cell = wins, cost, i
            return best_wins, best_cost, best_cell

        wins, cost, i = search((1 << len(cells)) - 1, tuple(range(len(layouts))))
        if i is None:
            return None
        risk = sum(bool(mask & (1 << i)) for mask in layouts) / len(layouts)
        return EndgameChoice(
            cells[i], risk, wins / len(layouts), cost / wins if wins else 0.0, len(layouts)
        )
    except _Limit:
        return None
