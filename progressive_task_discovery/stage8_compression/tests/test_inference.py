"""Counterexamples the first merger passed and should not have.

Every fixture here is small enough to reason about by hand. Three of the four
are regression tests for defects found by review after Step 5 was reported; the
fourth is a diagnostic, not a requirement, because a greedy merger is
order-dependent in general.
"""
import itertools, json, subprocess, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inference import Node, Classes, rebuild, score, from_dump


def node(alphabet, advanced=(), ignored=(), succ=None, visits=100, truth=None):
    nd = Node(alphabet, visits, truth)
    for e in advanced:
        nd.advanced[e] = 1
    for e in ignored:
        nd.ignored[e] = 1
    if succ:
        for e, s in succ.items():
            nd.succ[e] = s
    return nd


def legacy_rebuild(nodes, alphabet, theta=0, quota=0):
    """The rule as first implemented: conflict read off the representative only,
    quota checked only on the top-level pair, union applied as it went."""
    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def conflict(a, b):
        return any((nodes[a].advanced[e] and nodes[b].ignored[e]) or
                   (nodes[a].ignored[e] and nodes[b].advanced[e]) for e in range(alphabet))

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = find(i), find(j)
            if a == b or conflict(a, b):
                continue
            stack, touched, seen = [(a, b)], [], set()
            ok = True
            while stack:
                x, y = stack.pop()
                x, y = find(x), find(y)
                if x == y or (x, y) in seen:
                    continue
                seen.add((x, y))
                touched.append((x, y))
                if conflict(x, y):
                    ok = False
                    break
                for e in range(alphabet):
                    if nodes[x].succ[e] >= 0 and nodes[y].succ[e] >= 0:
                        stack.append((nodes[x].succ[e], nodes[y].succ[e]))
            if ok:
                for x, y in touched:
                    x, y = find(x), find(y)
                    if x != y:
                        parent[max(x, y)] = min(x, y)
    label, out = {}, []
    for i in range(len(nodes)):
        out.append(label.setdefault(find(i), len(label)))
    return out


class MergeRuleTests(unittest.TestCase):
    def fixture_member_evidence(self):
        """A knows nothing about event 0; B says it advances, C says it is
        ignored. Once A and B are one class, A must inherit B's evidence."""
        return [node(2, ignored=[1]), node(2, advanced=[0]), node(2, ignored=[0])]

    def test_A_member_evidence_blocks_the_second_merge(self):
        nodes = self.fixture_member_evidence()
        blocks = rebuild(nodes, 2, quota=0, strict_children=False)
        self.assertEqual(blocks[0], blocks[1], 'A and B are compatible and should merge')
        self.assertNotEqual(blocks[1], blocks[2], 'B and C contradict and must stay apart')

    def test_A_is_a_real_regression_the_first_rule_fails_it(self):
        nodes = self.fixture_member_evidence()
        legacy = legacy_rebuild(nodes, 2)
        self.assertEqual(legacy[1], legacy[2],
                         'the first rule merged a directly contradicting pair')

    def test_B_propagated_children_are_held_to_the_quota(self):
        """Parents have tried everything; their successors have tried nothing."""
        nodes = [node(2, advanced=[0], ignored=[1], succ={0: 2}),
                 node(2, advanced=[0], ignored=[1], succ={0: 3}),
                 node(2), node(2)]
        strict = rebuild(nodes, 2, quota=1, strict_children=True)
        self.assertNotEqual(strict[2], strict[3], 'children have no evidence at all')
        self.assertNotEqual(strict[0], strict[1], 'so the parent merge must fail too')
        loose = rebuild(nodes, 2, quota=1, strict_children=False)
        self.assertEqual(loose[0], loose[1])
        self.assertEqual(loose[2], loose[3], 'without the flag children merge for free')

    def test_C_a_contradiction_two_levels_down_fails_the_whole_merge(self):
        nodes = [node(2, advanced=[0], succ={0: 2}), node(2, advanced=[0], succ={0: 3}),
                 node(2, advanced=[0], succ={0: 4}), node(2, advanced=[0], succ={0: 5}),
                 node(2, advanced=[1]), node(2, ignored=[1])]
        blocks = rebuild(nodes, 2, quota=0, strict_children=False)
        self.assertNotEqual(blocks[4], blocks[5], 'the deep pair contradicts')
        self.assertNotEqual(blocks[0], blocks[1],
                            'and the merge that implies it must be discarded whole')

    def test_D_order_dependence_is_measured_not_assumed(self):
        """A greedy merger is order-dependent in general, so this reports rather
        than requires. A failure here is information about the rule."""
        nodes = [node(2, advanced=[0]), node(2, ignored=[1]), node(2, advanced=[1]),
                 node(2, ignored=[0])]
        seen = set()
        for perm in itertools.permutations(range(4)):
            blocks = rebuild(nodes, 2, quota=0, strict_children=False,
                             order=lambda ps, p=perm: sorted(ps, key=lambda z: (p[z[0]], p[z[1]])))
            seen.add(tuple(sorted(map(tuple, _classes(blocks)))))
        self.assertLessEqual(len(seen), 24)
        if len(seen) > 1:
            print(f'\n  note: this rule is order-dependent, {len(seen)} distinct partitions')

    def test_rollback_leaves_no_trace_of_a_failed_attempt(self):
        nodes = [node(2, advanced=[0], succ={0: 2}), node(2, advanced=[0], succ={0: 3}),
                 node(2, advanced=[1]), node(2, ignored=[1])]
        cls = Classes(nodes, 2, quota=0, strict_children=False)
        before = (list(cls.parent), [set(x) for x in cls.adv], [list(x) for x in cls.tried])
        self.assertFalse(cls.attempt(0, 1))
        self.assertEqual(cls.parent, before[0])
        self.assertEqual([set(x) for x in cls.adv], before[1])
        self.assertEqual([list(x) for x in cls.tried], before[2])


class PortAgreementTests(unittest.TestCase):
    def test_python_rule_reproduces_the_learner_partition(self):
        metas = json.loads((ROOT / 'results/family_maps/rungs.json').read_text())
        for meta, quota, strict, theta in [(metas[0], 0, 0, 50), (metas[2], 1, 1, 100),
                                           (metas[4], 1, 0, 100)]:
            out = subprocess.check_output(
                [str(ROOT / 'compress'), '--task', str(ROOT / f"results/family_maps/{meta['tag']}.task"),
                 '--method', 'merged', '--seed', '1', '--budget', '60000', '--every', '60000',
                 '--theta', str(theta), '--quota', str(quota),
                 '--strict-children', str(strict), '--dump-tree', '1'], text=True)
            dump = json.loads(out)['tree']
            nodes, alphabet = from_dump(dump)
            blocks = rebuild(nodes, alphabet, dump['theta'], dump['quota'],
                             bool(dump['strict_children']))
            self.assertEqual(blocks, dump['blocks'], meta['tag'])


def _classes(blocks):
    groups = {}
    for i, b in enumerate(blocks):
        groups.setdefault(b, []).append(i)
    return [tuple(v) for v in groups.values()]


if __name__ == '__main__':
    unittest.main(verbosity=2)
