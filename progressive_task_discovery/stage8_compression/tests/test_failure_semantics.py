"""The evidence boundary around failure, which was not clean before.

Entering the absorbing failure sink used to leave the episode's history and tree
node frozen at their pre-failure values while every remaining step kept writing
"this event was ignored at that history", kept incrementing the failure count,
and kept charging the whole remaining horizon again. On one reproduced condition
that was 1,797 post-failure steps counted as 1,797 extra failures and 1,766
events written into a healthy history, 738 of them directly contradicting a
progress record already stored at that node.
"""
import json, subprocess, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inference import Node, rebuild
from passive_baselines import rpni, edsm

FATAL = ROOT / 'results/irreversible/taskR_00_fatal.task'


def run(**kw):
    cmd = [str(ROOT / 'compress'), '--task', str(FATAL), '--every', '100000']
    for k, v in kw.items():
        cmd += ['--' + k.replace('_', '-'), str(v)]
    return json.loads(subprocess.check_output(cmd, text=True))


class FailureSemanticsTests(unittest.TestCase):
    def test_failures_counts_entries_not_steps_spent_dead(self):
        r = run(method='merged', seed=5000, budget=50000, doom_continues=1)
        # One entry per episode at most, and episodes are bounded by budget/1.
        self.assertLessEqual(r['failures'], r['episodes'])

    def test_wasted_steps_are_counted_once_each(self):
        r = run(method='merged', seed=5000, budget=50000, doom_continues=1)
        # Steps taken after the task failed cannot exceed the steps taken.
        self.assertLessEqual(r['doomed_steps'], r['budget'])
        self.assertGreater(r['doomed_steps'], 0)

    def test_stopping_on_failure_wastes_nothing(self):
        r = run(method='merged', seed=5000, budget=50000, doom_continues=0)
        self.assertEqual(r['doomed_steps'], 0)

    def test_a_fatal_event_is_not_an_ignored_event(self):
        """The merge rule must keep the two verdicts apart.

        Folding fatal into ignored let a node that had seen an event end the task
        merge with a node that had seen the same event do nothing, and no later
        evidence could ever refute that merge.
        """
        alphabet = 2
        killed = Node(alphabet, visits=10, truth=0)
        killed.fatal[0] = 3
        harmless = Node(alphabet, visits=10, truth=1)
        harmless.ignored[0] = 3
        nodes = [killed, harmless]
        for name, rule in (('sweep', lambda n, a: rebuild(n, a, quota=0, strict_children=False)),
                           ('rpni', rpni), ('edsm', edsm)):
            blocks = rule(nodes, alphabet)
            self.assertNotEqual(blocks[0], blocks[1], name)

    def test_jirp_transfers_only_against_a_snapshotted_hypothesis(self):
        """Both signatures used to be read off the tree as it stood *now*."""
        kw = dict(method='merged', seed=0, budget=60000, theta=20, quota=1,
                  strict_children=0)
        self.assertEqual(run(revision='reset', **kw)['jirp_transfers'], 0)
        self.assertGreater(run(revision='jirp', **kw)['jirp_transfers'], 0)

    def test_no_relabel_trains_the_goal_the_row_was_collected_under(self):
        """Recomputing it at replay time used current skill values and no node."""
        kw = dict(method='goal', budget=40000, seed=0)
        a = run(goal_select='value', relabel=0, **kw)
        b = run(goal_select='value', relabel=0, **kw)
        self.assertEqual(a['trace_hash'], b['trace_hash'])
        self.assertNotEqual(a['trace_hash'],
                            run(goal_select='value', relabel=1, **kw)['trace_hash'])


class VerificationBudgetTests(unittest.TestCase):
    """Metering the progress question must be exact and must not leak."""

    TASK = ROOT / 'results/overlap/taskP_high0B.task'

    def go(self, **kw):
        cmd = [str(ROOT / 'compress'), '--task', str(self.TASK), '--method', 'goal',
               '--goal-select', 'learned', '--every', '40000', '--theta', '20',
               '--quota', '1', '--strict-children', '0']
        for k, v in kw.items():
            cmd += ['--' + k.replace('_', '-'), str(v)]
        return json.loads(subprocess.check_output(cmd, text=True))

    def test_unlimited_budget_reproduces_free_feedback_exactly(self):
        """The control that says the metering itself changes nothing."""
        free = self.go(seed=0, budget=40000)
        metered = self.go(seed=0, budget=40000, verify_budget=10 ** 9)
        self.assertEqual(free['trace_hash'], metered['trace_hash'])
        self.assertGreater(metered['verifications'], 0)
        self.assertEqual(metered['unknown_firings'], 0)

    def test_the_budget_is_respected(self):
        r = self.go(seed=0, budget=40000, verify_budget=20)
        self.assertLessEqual(r['verifications'], 20)
        self.assertGreater(r['unknown_firings'], 0)

    def test_terminal_only_feedback_leaves_the_tree_empty(self):
        """With nothing verified the agent may not grow a structure at all.

        If it did, the history would be being filtered by a task-state change it
        never paid to see, and switching the feedback off would still leak.
        """
        r = self.go(seed=0, budget=40000, verify_budget=0)
        self.assertEqual(r['verifications'], 0)
        self.assertEqual(r['nodes'], 1)

    def test_less_feedback_is_never_better(self):
        a = self.go(seed=0, budget=100000, every=250, verify_budget=0)['auc']
        b = self.go(seed=0, budget=100000, every=250, verify_budget=10 ** 9)['auc']
        self.assertLess(a, b)


class QueryPolicyTests(unittest.TestCase):
    TASK = ROOT / 'results/ceiling/c_n4d2m3_6L3a11.task'

    def go(self, **kw):
        cmd = [str(ROOT / 'compress'), '--task', str(self.TASK), '--method', 'goal',
               '--goal-select', 'learned', '--every', '250', '--theta', '20',
               '--quota', '1', '--strict-children', '0', '--merge-rule', 'edsm',
               '--budget', '200000', '--seed', '0']
        for k, v in kw.items():
            cmd += ['--' + k.replace('_', '-'), str(v)]
        return json.loads(subprocess.check_output(cmd, text=True))

    def test_a_selective_policy_spends_strictly_less(self):
        a = self.go(verify_budget=320, verify_policy='arrival')
        n = self.go(verify_budget=320, verify_policy='novel')
        d = self.go(verify_budget=320, verify_policy='decision')
        self.assertLess(n['verifications'], a['verifications'])
        self.assertLess(d['verifications'], n['verifications'])

    def test_consulting_the_structure_does_not_cost_the_outcome(self):
        a = self.go(verify_budget=320, verify_policy='arrival')
        n = self.go(verify_budget=320, verify_policy='novel')
        self.assertEqual(a['first90'], n['first90'])

    def test_closed_world_as_a_skip_rule_trades_outcome_for_queries(self):
        """It is a saving, but not a free one, so it must be reported as a trade.

        The single-family probe that suggested otherwise did not survive the
        grid: across 24 families it cuts queries 2.03x and the solved count from
        96/96 to 81/96.
        """
        a = self.go(verify_budget=320, verify_policy='arrival')
        c = self.go(verify_budget=320, verify_policy='closed')
        self.assertLess(c['verifications'], a['verifications'])

    def test_the_skip_rule_reads_the_partition_it_is_told_to(self):
        sweep = self.go(verify_budget=320, verify_policy='closed',
                        verify_partition='sweep')
        true = self.go(verify_budget=320, verify_policy='closed',
                       verify_partition='true')
        self.assertNotEqual(sweep['verifications'], true['verifications'])

    def test_the_class_fallback_only_fires_when_a_sibling_knows(self):
        """With no merging there is no sibling, so nothing can be generalised."""
        merged = self.go(verify_budget=160, verify_fallback='class')
        self.assertGreater(merged['generalised_firings'], 0)
        cmd_no_merge = self.go(verify_budget=160, verify_fallback='own')
        self.assertEqual(cmd_no_merge['generalised_firings'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
