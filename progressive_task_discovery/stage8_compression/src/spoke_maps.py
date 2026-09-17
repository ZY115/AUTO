"""Corridor layouts: more geometric diversity per walkable cell.

An open room grows as the square of its side, and under a terminal-only reward
the exploration cost grows with it, so the large rooms needed to force many
distinct optimal orderings stopped being learnable at all. Corridors buy the
same diversity far more cheaply: distance along a spoke varies sharply with
position, so nearby starts prefer different orderings while the cell count
stays small.

Layout: a hub with `spokes` radial corridors of length `arm`. Unordered labels
sit on the spokes, possibly several cells per label. A single corridor leaves
the hub to a suffix area holding the three suffix labels, and the cell where it
leaves is a cut vertex, so every history funnels through one place.
"""
from pathlib import Path
import random, sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def build_spokes(n, per_label, seed, spokes=3, arm=7, stem=3):
    """Hub with `spokes` labelled corridors, plus one unlabelled exit corridor.

    The exit corridor is the only route from the hub to the suffix labels, and
    its first cell is therefore a cut vertex. Label corridors never run beside
    it, so no history can slip past the funnel.
    """
    rng = random.Random(seed)
    hub = (0, 0)
    arms = [[(dx * i, dy * i) for i in range(1, arm + 1)]
            for dx, dy in DIRS[1:1 + spokes]]
    exit_line = [(0, -i) for i in range(1, stem + 4)]
    cells = {hub} | {c for line in arms for c in line} | set(exit_line)
    cells = sorted(cells)
    index = {p: i for i, p in enumerate(cells)}
    slots = [c for line in arms for c in line]
    if len(slots) < n * per_label:
        return None
    picks = rng.sample(slots, n * per_label)
    labels = {p: g for g in range(n) for p in picks[g * per_label:(g + 1) * per_label]}
    for j in range(3):
        labels[exit_line[stem + j]] = n + j
    moves = []
    for p in cells:
        row = []
        for dx, dy in DIRS:
            z = (p[0] + dx, p[1] + dy)
            if z not in index:
                z = p
            row.append([index[z], labels.get(z, -1) if z != p else -1])
        moves.append(row)
    room = [index[p] for p in cells if p not in exit_line]
    starts = [index[p] for p in cells if p not in labels and p not in exit_line]
    return dict(n=n, instances=per_label, seed=seed, cells=[list(p) for p in cells],
                labels=[[index[p], g] for p, g in sorted(labels.items())], moves=moves,
                junction=index[exit_line[0]], room=room, candidate_starts=starts,
                suffix_cells=[index[p] for p in exit_line])


def funnel_ok(layout):
    """Every route from the room to a suffix label passes the junction cell."""
    J = layout['junction']
    suffix = {s for s, g in layout['labels'] if g >= layout['n']}
    root = layout['room'][0]
    seen, stack = {root}, [root]
    while stack:
        s = stack.pop()
        for ns, _ in layout['moves'][s]:
            if ns != J and ns not in seen:
                seen.add(ns)
                stack.append(ns)
    return not (seen & suffix)


def build_forked(n, per_label, seed, spokes=3, arm=7, stem=3, bridge=None, tails=None):
    """Spoke map whose exit ends in a fork, one arm per branch.

    With the suffix labels strung along a single corridor, reaching one branch's
    first label means walking over the other's. That is fine while wrong events
    are ignored and fatal once they are not: it removes a whole branch rather
    than punishing a memory mistake. Here the exit corridor carries the shared
    bridge and then forks, so each branch's first label sits one step off the
    fork in a different direction and neither blocks the other.

    The same geometry is used for the reversible and the irreversible version of
    a task, so the two differ only in what a wrong commit does.
    """
    rng = random.Random(seed)
    hub = (0, 0)
    arms = [[(dx * i, dy * i) for i in range(1, arm + 1)]
            for dx, dy in DIRS[1:1 + spokes]]
    bridge = list(bridge or [])
    tails = [list(t) for t in (tails or [[], []])]
    exit_line = [(0, -i) for i in range(1, stem + len(bridge) + 1)]
    fork = exit_line[-1] if exit_line else hub
    left = [(fork[0] - i, fork[1]) for i in range(1, len(tails[0]) + 1)]
    right = [(fork[0] + i, fork[1]) for i in range(1, len(tails[1]) + 1)]
    cells = {hub} | {c for line in arms for c in line} | set(exit_line) | set(left) | set(right)
    cells = sorted(cells)
    index = {p: i for i, p in enumerate(cells)}
    slots = [c for line in arms for c in line]
    if len(slots) < n * per_label:
        return None
    picks = rng.sample(slots, n * per_label)
    labels = {p: g for g in range(n) for p in picks[g * per_label:(g + 1) * per_label]}
    for j, e in enumerate(bridge):
        labels[exit_line[stem + j]] = e
    for j, e in enumerate(tails[0]):
        labels[left[j]] = e
    for j, e in enumerate(tails[1]):
        labels[right[j]] = e
    moves = []
    for p in cells:
        row = []
        for dx, dy in DIRS:
            z = (p[0] + dx, p[1] + dy)
            if z not in index:
                z = p
            row.append([index[z], labels.get(z, -1) if z != p else -1])
        moves.append(row)
    suffix_cells = exit_line + left + right
    room = [index[p] for p in cells if p not in set(suffix_cells)]
    starts = [index[p] for p in cells if p not in labels and p not in set(suffix_cells)]
    return dict(n=n, instances=per_label, seed=seed, cells=[list(p) for p in cells],
                labels=[[index[p], g] for p, g in sorted(labels.items())], moves=moves,
                junction=index[exit_line[0]], room=room, candidate_starts=starts,
                suffix_cells=[index[p] for p in suffix_cells])
