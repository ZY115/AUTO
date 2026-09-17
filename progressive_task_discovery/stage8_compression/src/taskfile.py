"""Read a .task file in Python, so an offline audit can use the same ground truth
the learner ran against without re-deriving the task from its generator.

Layout is exactly what compress.cpp's Task::load consumes, in order.
"""
from pathlib import Path


class TaskFile:
    def __init__(self, path):
        t = iter(Path(path).read_text().split())
        nxt = lambda: int(next(t))
        self.n, self.d, self.m, self.pa, self.pb = (nxt() for _ in range(5))
        self.alphabet, self.nstates = nxt(), nxt()
        self.minimal = [nxt() for _ in range(self.nstates)]
        self.accepting = [nxt() for _ in range(self.nstates)]
        self.failing = [nxt() for _ in range(self.nstates)]
        self.trans = [[nxt() for _ in range(self.alphabet)] for _ in range(self.nstates)]
        self.N = nxt()
        self.moves = [[0] * 4 for _ in range(self.N)]
        self.events = [[0] * 4 for _ in range(self.N)]
        for s in range(self.N):
            for a in range(4):
                self.moves[s][a] = nxt()
                self.events[s][a] = nxt()
        S = nxt()
        self.starts = [nxt() for _ in range(S)]
        self.lengths = [nxt() for _ in range(S)]
        self.horizon, self.maxlen, self.minwindow = nxt(), nxt(), nxt()

    def step(self, q, e):
        return q if e < 0 else self.trans[q][e]

    def run(self, history):
        q = 0
        for e in history:
            q = self.step(q, e)
        return q

    def verdict(self, q, e):
        """What the task does with event e in state q: the ground truth labels."""
        nq = self.trans[q][e]
        if self.failing[nq]:
            return 'fatal'
        return 'advance' if nq != q else 'ignore'

    def reachable_histories(self):
        """Progress-event histories physically achievable on this map.

        A history is the sequence of events that changed the task state, so two
        routes differing only in wandering give the same history.

        Breadth first, and that matters: a stack closed each (cell, history) at
        whatever depth it was first reached, so arriving late by a long route
        blocked a later, shorter arrival from being expanded within the horizon.
        An independent check found 14 of 176 task files short of histories that
        way. Equal step costs make the first dequeue minimal-depth, so closing a
        key on dequeue is sound here.
        """
        from collections import deque
        start_keys = [(s, ()) for s in self.starts]
        seen = set(start_keys)
        frontier = deque((s, h, 0) for s, h in start_keys)
        out = {()}
        while frontier:
            s, h, depth = frontier.popleft()
            if depth >= self.horizon:
                continue
            q = self.run(h)
            if self.accepting[q] or self.failing[q]:
                continue
            for a in range(4):
                ns, ev = self.moves[s][a], self.events[s][a]
                nh = h
                if ev >= 0:
                    nq = self.trans[q][ev]
                    if nq != q and not self.failing[nq]:
                        nh = h + (ev,)
                    elif self.failing[nq]:
                        continue          # a dead history is not a task history
                key = (ns, nh)
                if key in seen:
                    continue
                seen.add(key)
                out.add(nh)
                frontier.append((ns, nh, depth + 1))
        return out
