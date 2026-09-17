"""RPNI and EDSM, on the same evidence and through the same consistency test."""
import json, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from family_maps import Task
from inference import Node, rebuild, score
from inference_audit import reachable, build_nodes
from passive_baselines import rpni, edsm


def rungs():
    return json.loads((ROOT / 'results/family_maps/rungs.json').read_text())


class PassiveTests(unittest.TestCase):
    def test_complete_evidence_recovers_the_minimal_partition(self):
        """The claim this kills: the merge rule is not a contribution.

        With the whole reachable task language and every event tried once, the
        project's rule, RPNI and EDSM all return exactly the minimal partition.
        """
        for meta in rungs()[:3]:
            task = Task(meta['n'], tuple(meta['pair']), meta['d'], meta['m'])
            nodes = build_nodes(task, reachable(task))
            for name, rule in (('sweep', lambda n, a: rebuild(n, a, quota=0, strict_children=False)),
                               ('rpni', rpni), ('edsm', edsm)):
                p, r, k, t = score(nodes, rule(nodes, task.alphabet))
                self.assertEqual((p, r, k), (1.0, 1.0, t), (meta['tag'], name))

    def test_a_blue_fringe_merge_never_unites_a_contradicting_pair(self):
        """Both learners go through the project's own consistency test."""
        alphabet = 3
        a = Node(alphabet, visits=10, truth=0)
        a.advanced[0] = 1
        b = Node(alphabet, visits=10, truth=1)
        b.ignored[0] = 1
        root = Node(alphabet, visits=10, truth=2)
        root.advanced[1] = 1
        root.succ[1] = 1
        root.advanced[2] = 1
        root.succ[2] = 2
        nodes = [root, a, b]
        for rule in (rpni, edsm):
            blocks = rule(nodes, alphabet)
            self.assertNotEqual(blocks[1], blocks[2], rule.__name__)

    def test_partition_is_a_partition(self):
        for meta in rungs()[:2]:
            task = Task(meta['n'], tuple(meta['pair']), meta['d'], meta['m'])
            nodes = build_nodes(task, reachable(task))
            for rule in (rpni, edsm):
                blocks = rule(nodes, task.alphabet)
                self.assertEqual(len(blocks), len(nodes))
                self.assertEqual(sorted(set(blocks)), list(range(len(set(blocks)))))


if __name__ == '__main__':
    unittest.main(verbosity=2)
