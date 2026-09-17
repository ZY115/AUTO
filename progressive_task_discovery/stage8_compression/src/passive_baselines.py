"""Two standard passive automaton learners, run on this project's own evidence.

The plan required a comparison against a passive-inference baseline before the
merge rule could be reported as a contribution, and REPORT_STEP9 records that it
was never run. AALpy is not installed in this environment, so RPNI and
blue-fringe EDSM are written here directly, on top of the same `Classes`
machinery the project's own rule uses. That machinery supplies the consistency
test and the merge closure with rollback, so the only thing that differs between
the three learners is the search: which candidate pairs are considered, in what
order, and how one is chosen among them.

That is the right thing to isolate. The project's rule sweeps every eligible pair
in index order and merges whenever the pair does not directly contradict. RPNI
keeps a red core and a blue fringe and takes the first consistent merge in
lexicographic order. EDSM does the same but scores every consistent candidate by
how much evidence the merge would explain, and takes the best one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import Classes


class Scored(Classes):
    """Classes plus the count of evidence agreements a merge closure produced."""

    def attempt_scored(self, i, j):
        mark = len(self.log)
        stack, gained = [(i, j)], 0
        while stack:
            x, y = stack.pop()
            a, b = self.find(x), self.find(y)
            if a == b:
                continue
            if self.conflict(a, b):
                self.rollback(mark)
                return None
            if self.strict and not (self.quota_met(a) and self.quota_met(b)):
                self.rollback(mark)
                return None
            # EDSM's score: every event on which the two sides already agree is
            # one piece of evidence the merge explains rather than assumes.
            gained += len((self.adv[a] & self.adv[b]) | (self.ign[a] & self.ign[b])
                          | (self.fat[a] & self.fat[b]))
            children = [(self.succ[a][e], self.succ[b][e]) for e in range(self.alphabet)
                        if self.succ[a][e] >= 0 and self.succ[b][e] >= 0]
            self.unite(a, b)
            stack.extend(children)
        return gained


def _fringe(cls, nodes, red):
    """States one transition out of the red core that are not themselves red."""
    blue = []
    for r in red:
        for e in range(cls.alphabet):
            t = cls.succ[r][e]
            if t < 0:
                continue
            t = cls.find(t)
            if t not in red and t not in blue:
                blue.append(t)
    return sorted(blue)


def _blue_fringe(nodes, alphabet, quota, strict, pick):
    cls = Scored(nodes, alphabet, quota, strict)
    red = [cls.find(0)]
    guard = 0
    while guard < 10 * len(nodes) + 100:
        guard += 1
        blue = _fringe(cls, nodes, red)
        if not blue:
            break
        b = blue[0]
        candidates = []
        for r in red:
            mark = len(cls.log)
            s = cls.attempt_scored(r, b)
            if s is None:
                continue
            cls.rollback(mark)
            candidates.append((s, r))
        if not candidates:
            red.append(b)
            continue
        cls.attempt_scored(pick(candidates)[1], b)
        red = sorted({cls.find(r) for r in red})
    label, out = {}, []
    for i in range(len(nodes)):
        out.append(label.setdefault(cls.find(i), len(label)))
    return out


def rpni(nodes, alphabet, quota=0, strict=False):
    """First consistent merge in lexicographic order over the red core."""
    return _blue_fringe(nodes, alphabet, quota, strict, lambda c: min(c, key=lambda x: x[1]))


def edsm(nodes, alphabet, quota=0, strict=False):
    """Highest-scoring consistent merge, ties broken lexicographically."""
    return _blue_fringe(nodes, alphabet, quota, strict,
                        lambda c: max(c, key=lambda x: (x[0], -x[1])))
