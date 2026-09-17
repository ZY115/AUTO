"""The offline predictors, which carry the whole of Step 19's claim."""
import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inference import Node
from conversion_audit import Folded, histories_of
from taskfile import TaskFile


def tree(alphabet=2):
    #   root --0--> a --1--> b
    # root saw 0 advance; a saw 1 advance; b saw nothing.
    root, a, b = (Node(alphabet, visits=9, truth=0) for _ in range(3))
    root.advanced[0] = 1
    root.succ[0] = 1
    a.advanced[1] = 1
    a.succ[1] = 2
    return [root, a, b]


class FoldedTests(unittest.TestCase):
    def test_the_two_causes_of_unknown_are_kept_apart(self):
        """A history the tree cannot fold at all, and one it can but has nothing
        to say about, call for different fixes and are reported separately."""
        nodes = tree()
        f = Folded(nodes, list(range(3)), 2)
        self.assertEqual(f.ask((0, 1), 1, False)[0], 'no_evidence')   # folds, silent
        self.assertEqual(f.ask((0, 1, 1), 0, False)[0], 'unreachable')  # cannot fold

    def test_merging_is_what_makes_an_unseen_history_reachable(self):
        """Putting b in the same class as a gives the class a's successor."""
        nodes = tree()
        merged = Folded(nodes, [0, 1, 1], 2)
        self.assertEqual(merged.ask((0, 1, 1), 1, False)[1], 'advance')

    def test_open_world_says_unknown_where_closed_world_asserts(self):
        nodes = tree()
        f = Folded(nodes, list(range(3)), 2)
        self.assertEqual(f.ask((0,), 0, False), ('no_evidence', None))
        self.assertEqual(f.ask((0,), 0, True), ('answer', 'ignore'))

    def test_closed_world_never_overrides_observed_evidence(self):
        nodes = tree()
        f = Folded(nodes, list(range(3)), 2)
        self.assertEqual(f.ask((), 0, True)[1], 'advance')

    def test_histories_are_recovered_from_the_successor_edges(self):
        self.assertEqual(set(histories_of(tree()).values()), {(), (0,), (0, 1)})


class ReachabilityTests(unittest.TestCase):
    def test_reachable_histories_exclude_dead_and_include_the_empty_one(self):
        t = TaskFile(ROOT / 'results/irreversible/taskR_00_fatal.task')
        hs = t.reachable_histories()
        self.assertIn((), hs)
        for h in hs:
            self.assertFalse(t.failing[t.run(h)])

    def test_time_bounded_reachability_is_complete(self):
        """A stack closed each (cell, history) at whatever depth it first got
        there, so a long route arriving early blocked a short route arriving
        later from being expanded inside the horizon. Counts below were computed
        independently by an external review before the fix.
        """
        self.assertEqual(
            len(TaskFile(ROOT / 'results/overlap/taskQ_high0B.task').reachable_histories()), 109)
        self.assertEqual(
            len(TaskFile(ROOT / 'results/overlap/taskP_high0A.task').reachable_histories()), 43)

    def test_every_prefix_of_a_reachable_history_is_reachable(self):
        hs = TaskFile(ROOT / 'results/overlap/taskQ_high0B.task').reachable_histories()
        for h in hs:
            for k in range(len(h)):
                self.assertIn(h[:k], hs)

    def test_two_maps_of_one_task_reach_different_history_sets(self):
        """Without this the generalisation audit has no novel questions."""
        a = TaskFile(ROOT / 'results/overlap/taskP_high0A.task').reachable_histories()
        b = TaskFile(ROOT / 'results/overlap/taskP_high0B.task').reachable_histories()
        self.assertTrue(b - a)


if __name__ == '__main__':
    unittest.main(verbosity=2)
