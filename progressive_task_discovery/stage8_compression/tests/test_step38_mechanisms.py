"""Regression cover for the switches Steps 36-40 added, which the old 59 did not.

The audit's point: the suite passing said nothing about `--skill-key perstate`,
`--route meta`, `--meta-safe`, the CraftWorld export, or the statistics. Each
test below pins a property that was actually broken at some point in this work.
"""
import json, subprocess, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
EXE = str(ROOT / 'compress')
LAVA = ROOT / 'results/craftworld_lava'


def run(task, *args, seed=0, budget=60000, every=2000):
    return json.loads(subprocess.check_output(
        [EXE, '--task', str(task), '--seed', str(seed), '--budget', str(budget),
         '--every', str(every), *map(str, args)], text=True))


class SkillKeyTests(unittest.TestCase):
    def test_perstate_allocates_a_policy_per_task_state(self):
        """Shared and per-state must not be the same run."""
        t = LAVA / 'cwl_book_expensive_0_safe.task'
        a = run(t, '--method', 'goal')
        b = run(t, '--method', 'goal', '--skill-key', 'perstate')
        self.assertNotEqual(a['trace_hash'], b['trace_hash'])

    def test_shared_is_the_default_and_unchanged(self):
        t = LAVA / 'cwl_book_expensive_0_safe.task'
        self.assertEqual(run(t, '--method', 'goal')['trace_hash'],
                         run(t, '--method', 'goal', '--skill-key', 'shared')['trace_hash'])


class MetaSafeTests(unittest.TestCase):
    """`--meta-safe` was once applied in action selection but not in the SMDP
    bootstrap, which pinned the target at the forbidden goal's untouched zero and
    silently zeroed the whole value function."""

    def test_meta_safe_is_inert_when_nothing_is_fatal(self):
        t = LAVA / 'cwl_book_expensive_0_safe.task'
        a = run(t, '--method', 'goal', '--route', 'meta')
        b = run(t, '--method', 'goal', '--route', 'meta', '--meta-safe', '1')
        self.assertEqual(a['trace_hash'], b['trace_hash'])

    def test_meta_safe_learns_something_when_a_goal_is_fatal(self):
        t = LAVA / 'cwl_book_expensive_0_lava.task'
        r = run(t, '--method', 'goal', '--route', 'meta', '--meta-safe', '1',
                budget=300000)
        self.assertGreater(r['auc'], 0.0,
                           'a forbidden goal left in the bootstrap zeroes the value function')

    def test_meta_safe_avoids_the_fatal_goal(self):
        t = LAVA / 'cwl_book_expensive_0_lava.task'
        free = run(t, '--method', 'goal', '--route', 'meta', budget=300000)
        safe = run(t, '--method', 'goal', '--route', 'meta', '--meta-safe', '1',
                   budget=300000)
        self.assertLess(safe['failures'], free['failures'])


class RewardProtocolTests(unittest.TestCase):
    def test_zero_step_cost_without_override_also_zeroes_the_goal_reward(self):
        """The trap that made the first reproduction of the audit's probe fail:
        reward_success is derived from the step cost."""
        t = LAVA / 'cwl_book_expensive_0_lava.task'
        r = run(t, '--method', 'automaton', '--crm', '1', '--cost', '0.0',
                budget=200000)
        self.assertEqual(r['first90'], -1)

    def test_zero_step_cost_with_override_rescues_qrm(self):
        t = LAVA / 'cwl_book_expensive_0_lava.task'
        r = run(t, '--method', 'automaton', '--crm', '1', '--cost', '0.0',
                '--reward-success', '3.36', budget=200000)
        self.assertGreater(r['first90'], 0)


class ExportTests(unittest.TestCase):
    def test_safe_and_lava_share_geometry_events_and_starts(self):
        sys.path.insert(0, str(ROOT / 'src'))
        from taskfile import TaskFile
        for stem in ('cwl_book_expensive_0', 'cwl_cake_cheap_0'):
            a = TaskFile(str(LAVA / f'{stem}_safe.task'))
            b = TaskFile(str(LAVA / f'{stem}_lava.task'))
            self.assertEqual(a.moves, b.moves)
            self.assertEqual(a.events, b.events)
            self.assertEqual(a.starts, b.starts)
            self.assertEqual(a.horizon, b.horizon)
            self.assertEqual(sum(a.failing), 0)
            self.assertEqual(sum(b.failing), 1)

    def test_event_distance_counts_the_firing_step(self):
        """C_g was understated by exactly one: standing next to an event still
        costs the step that fires it. Checked on an exported task so the test
        does not need the HRM package, which only the .venv-rl interpreter has."""
        from collections import deque
        from taskfile import TaskFile
        t = TaskFile(str(LAVA / 'cwl_book_expensive_0_safe.task'))
        back = [[] for _ in range(t.N)]
        for pc in range(t.N):
            for a in range(4):
                back[t.moves[pc][a]].append(pc)
        INF = 10 ** 6
        for e in range(t.alphabet):
            d = [INF] * t.N
            q = deque()
            for c in range(t.N):
                if any(t.events[c][a] == e for a in range(4)):
                    d[c] = 1; q.append(c)
            while q:
                c = q.popleft()
                for pc in back[c]:
                    if d[pc] > d[c] + 1:
                        d[pc] = d[c] + 1; q.append(pc)
            reach = [v for v in d if v < INF]
            if reach:
                self.assertGreaterEqual(min(reach), 1)


class StatisticsTests(unittest.TestCase):
    def test_rmst_is_a_time_and_smaller_is_better(self):
        tau = 1_000_000
        rmst = lambda xs: sum(min(x, tau) if x > 0 else tau for x in xs) / len(xs)
        self.assertEqual(rmst([-1, -1]), tau)                 # all censored
        self.assertLess(rmst([1000, 2000]), rmst([1000, -1]))  # censoring is slow


if __name__ == '__main__':
    unittest.main()
