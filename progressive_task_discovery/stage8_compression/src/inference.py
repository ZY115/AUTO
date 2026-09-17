"""The merge rule, as an independent implementation of what compress.cpp does.

Kept separate on purpose. The C++ learner can dump the evidence it collected and
the partition it produced; this module re-derives the partition from the same
evidence, and a test asserts the two agree. Counterexamples are then written
against this implementation, where they are cheap to state.

The rule: two classes may merge when no event has been seen advancing the task
in one and being ignored in the other, when both classes have tried every event
at least `quota` times, and when the same holds recursively for every successor
pair the merge implies. Evidence is aggregated over class members, never read
off a representative. A merge is tentative until its whole closure passes.
"""
from copy import deepcopy
import json, sys


class Node:
    __slots__ = ('visits', 'advanced', 'ignored', 'fatal', 'succ', 'truth')

    def __init__(self, alphabet, visits=0, truth=None):
        self.visits = visits
        self.advanced = [0] * alphabet
        self.ignored = [0] * alphabet
        # An event that ends the task is a third verdict, never a kind of
        # "ignored": a node that saw one must not merge with a node that saw the
        # other.
        self.fatal = [0] * alphabet
        self.succ = [-1] * alphabet
        self.truth = truth

    def observations(self):
        return sum(self.advanced) + sum(self.ignored) + sum(self.fatal)


class Classes:
    """Union-find over nodes carrying the union of its members' evidence."""

    def __init__(self, nodes, alphabet, quota, strict_children):
        n = len(nodes)
        self.alphabet, self.quota, self.strict = alphabet, quota, strict_children
        self.parent = list(range(n))
        self.adv = [set(e for e in range(alphabet) if nd.advanced[e]) for nd in nodes]
        self.ign = [set(e for e in range(alphabet) if nd.ignored[e]) for nd in nodes]
        self.fat = [set(e for e in range(alphabet) if nd.fatal[e]) for nd in nodes]
        self.tried = [[nd.advanced[e] + nd.ignored[e] + nd.fatal[e]
                       for e in range(alphabet)] for nd in nodes]
        self.succ = [list(nd.succ) for nd in nodes]
        self.log = []

    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def conflict(self, a, b):
        return bool((self.adv[a] & (self.ign[b] | self.fat[b]))
                    or (self.ign[a] & (self.adv[b] | self.fat[b]))
                    or (self.fat[a] & (self.adv[b] | self.ign[b])))

    def quota_met(self, a):
        return self.quota <= 0 or all(t >= self.quota for t in self.tried[a])

    def unite(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        w, l = min(a, b), max(a, b)
        self.log.append((l, w, set(self.adv[w]), set(self.ign[w]), set(self.fat[w]),
                         list(self.tried[w]), list(self.succ[w])))
        self.parent[l] = w
        self.adv[w] |= self.adv[l]
        self.ign[w] |= self.ign[l]
        self.fat[w] |= self.fat[l]
        for e in range(self.alphabet):
            self.tried[w][e] += self.tried[l][e]
            if self.succ[w][e] < 0:
                self.succ[w][e] = self.succ[l][e]

    def rollback(self, mark):
        while len(self.log) > mark:
            l, w, adv, ign, fat, tried, succ = self.log.pop()
            self.parent[l] = l
            self.adv[w], self.ign[w], self.fat[w] = adv, ign, fat
            self.tried[w], self.succ[w] = tried, succ

    def attempt(self, i, j):
        mark = len(self.log)
        stack = [(i, j)]
        while stack:
            x, y = stack.pop()
            a, b = self.find(x), self.find(y)
            if a == b:
                continue
            if self.conflict(a, b):
                self.rollback(mark)
                return False
            if self.strict and not (self.quota_met(a) and self.quota_met(b)):
                self.rollback(mark)
                return False
            children = [(self.succ[a][e], self.succ[b][e]) for e in range(self.alphabet)
                        if self.succ[a][e] >= 0 and self.succ[b][e] >= 0]
            self.unite(a, b)
            stack.extend(children)
        return True


def rebuild(nodes, alphabet, theta=0, quota=0, strict_children=True, order=None):
    cls = Classes(nodes, alphabet, quota, strict_children)
    eligible = [i for i, nd in enumerate(nodes)
                if nd.visits >= theta and nd.observations() >= theta]
    pairs = [(i, j) for ii, i in enumerate(eligible) for j in eligible[ii + 1:]]
    if order is not None:
        pairs = order(pairs)
    for i, j in pairs:
        a, b = cls.find(i), cls.find(j)
        if a == b or not cls.quota_met(a) or not cls.quota_met(b) or cls.conflict(a, b):
            continue
        cls.attempt(i, j)
    label, out = {}, []
    for i in range(len(nodes)):
        r = cls.find(i)
        out.append(label.setdefault(r, len(label)))
    return out


def score(nodes, blocks):
    agree = learned = truth = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            same_true = nodes[i].truth == nodes[j].truth
            same_learned = blocks[i] == blocks[j]
            learned += same_learned
            truth += same_true
            agree += same_true and same_learned
    return (agree / learned if learned else 1.0,
            agree / truth if truth else 1.0,
            len(set(blocks)),
            len({nd.truth for nd in nodes}))


def from_dump(dump):
    alphabet = dump['alphabet']
    nodes = []
    for raw in dump['nodes']:
        nd = Node(alphabet, raw['visits'], raw['truth'])
        nd.advanced = list(raw['advanced'])
        nd.ignored = list(raw['ignored'])
        nd.fatal = list(raw.get('fatal', [0] * alphabet))
        nd.succ = list(raw['succ'])
        nodes.append(nd)
    return nodes, alphabet


if __name__ == '__main__':
    dump = json.load(open(sys.argv[1]))['tree']
    nodes, alphabet = from_dump(dump)
    blocks = rebuild(nodes, alphabet, dump['theta'], dump['quota'],
                     bool(dump['strict_children']))
    print('agrees with C++:', blocks == dump['blocks'])
