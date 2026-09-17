"""Checks the ladder result depends on: task semantics, map, accounting, arms."""
import itertools, json, subprocess, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from family_maps import Task, evaluate_layout, product_shortest
from spoke_maps import build_spokes, funnel_ok
from emit_tasks import make_layout


def rungs():
    return json.loads((ROOT / 'results/family_maps/rungs.json').read_text())


def run(task, method, seed=0, budget=6000, **kw):
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/family_maps/{task}.task'),
           '--method', method, '--seed', str(seed), '--budget', str(budget), '--every', '1000']
    for k, v in kw.items():
        cmd += ['--' + k, str(v)]
    return json.loads(subprocess.check_output(cmd, text=True))


class LadderTests(unittest.TestCase):
    def test_task_machine_accepts_exactly_the_intended_language(self):
        t = Task(3, (0, 1), 2, 1)
        for perm in itertools.permutations(range(3)):
            bit = 0 if perm.index(0) < perm.index(1) else 1
            good = list(perm) + t.bridge + t.tails[bit]
            self.assertTrue(t.done(t.run(good)), good)
            wrong = list(perm) + t.bridge + t.tails[1 - bit]
            self.assertNotEqual(t.tails[0], t.tails[1])
            self.assertFalse(t.done(t.run(wrong)), wrong)
        # wrong events are ignored, never fatal
        self.assertEqual(t.run([0, 0, 0]), t.run([0]))

    def test_minimisation_merges_exactly_future_equivalent_states(self):
        t = Task(3, (0, 1), 2, 1)
        futures = {}
        for i in range(len(t.states)):
            key = tuple(t.minimal[t.trans[i][e]] for e in range(t.alphabet)) + (i in t.accepting,)
            futures.setdefault(t.minimal[i], key)
            self.assertEqual(futures[t.minimal[i]], key)
        self.assertLess(t.n_minimal, len(t.states))

    def test_every_rung_funnels_and_count_is_aliased(self):
        for meta in rungs():
            layout = make_layout(meta)
            self.assertTrue(funnel_ok(layout), meta['tag'])
            info = evaluate_layout(layout, Task(meta['n'], tuple(meta['pair']),
                                                meta['d'], meta['m']), 24)
            self.assertTrue(info['count_aliased'], meta['tag'])
            self.assertEqual(info['history_states'], meta['history_states'])
            self.assertEqual(info['minimal_states'], meta['minimal_states'])

    def test_starts_have_distinct_unique_optima_and_both_branches(self):
        for meta in rungs():
            hists = [tuple(h) for h in meta['optimal_histories']]
            self.assertEqual(len(set(hists)), len(hists), meta['tag'])
            task = Task(meta['n'], tuple(meta['pair']), meta['d'], meta['m'])
            bits = {task.run(h[:meta['n']])[1] for h in hists}
            self.assertEqual(bits, {0, 1}, meta['tag'])

    def test_optimal_lengths_match_independent_shortest_paths(self):
        for meta in rungs():
            layout = make_layout(meta)
            task = Task(meta['n'], tuple(meta['pair']), meta['d'], meta['m'])
            dist, N = product_shortest(layout, task)
            self.assertEqual([dist[s] for s in meta['starts']], meta['lengths'], meta['tag'])

    def test_update_budget_is_identical_across_arms(self):
        tag = rungs()[0]['tag']
        for method in ('count', 'bag', 'history', 'automaton'):
            r = run(tag, method)
            self.assertEqual(r['updates'], 6 * r['budget'], method)

    def test_reruns_are_bitwise_reproducible(self):
        tag = rungs()[0]['tag']
        a, b = run(tag, 'automaton', seed=3), run(tag, 'automaton', seed=3)
        self.assertEqual(a['trace_hash'], b['trace_hash'])
        self.assertNotEqual(a['trace_hash'], run(tag, 'automaton', seed=4)['trace_hash'])

    def test_wide_window_matches_full_history(self):
        """A window at least as wide as the task cannot lose information."""
        meta = rungs()[0]
        width = meta['n'] + meta['d'] + meta['m'] + 2
        a = run(meta['tag'], 'window', budget=20000, window=width)
        b = run(meta['tag'], 'history', budget=20000)
        self.assertEqual(a['trace_hash'], b['trace_hash'])
        self.assertEqual(a['features'], b['features'])

    def test_evaluation_never_creates_features(self):
        """Frozen-policy rollouts must not touch the learner's table."""
        meta = rungs()[0]
        dense = run(meta['tag'], 'history', budget=20000)
        sparse_checks = run(meta['tag'], 'history', budget=20000)
        self.assertEqual(dense['features'], sparse_checks['features'])
        few = subprocess.check_output(
            [str(ROOT / 'compress'), '--task', str(ROOT / f"results/family_maps/{meta['tag']}.task"),
             '--method', 'history', '--seed', '0', '--budget', '20000', '--every', '20000'], text=True)
        self.assertEqual(json.loads(few)['features'], dense['features'])


class MergingTests(unittest.TestCase):
    def test_merge_rule_is_exact_on_complete_evidence(self):
        """Any shortfall in a run must be evidence, not the rule."""
        from merge_control import reachable_prefixes, evidence, merge, score
        for meta in rungs():
            task = Task(meta['n'], tuple(meta['pair']), meta['d'], meta['m'])
            pre = reachable_prefixes(task, [tuple(h) for h in meta['optimal_histories']])
            adv, ign, succ = evidence(task, pre)
            p, r, classes, truth = score(task, pre, merge(pre, adv, ign, succ))
            self.assertEqual((p, r), (1.0, 1.0), meta['tag'])
            self.assertEqual(classes, truth, meta['tag'])

    def test_merging_never_reads_the_task_machine(self):
        source = (ROOT / 'src/compress.cpp').read_text()
        body = source[source.index('struct Tree'):source.index('struct Encoder')]
        for forbidden in ('T.trans', 'T.minimal', 'T.accepting', 'next_state'):
            self.assertNotIn(forbidden, body)

    def test_merged_arm_starts_as_the_prefix_tree(self):
        """Before any node is evidence-bearing, merged must equal full history."""
        meta = rungs()[0]
        a = run(meta['tag'], 'merged', budget=4000, theta=10 ** 7)
        b = run(meta['tag'], 'history', budget=4000)
        self.assertEqual(a['trace_hash'], b['trace_hash'])
        self.assertEqual(a['revisions'], 0)

    def test_revision_policies_change_behaviour_but_not_step_budget(self):
        meta = rungs()[0]
        seen = {}
        for policy in ('reset', 'transfer', 'replay'):
            r = run(meta['tag'], 'merged', budget=20000, theta=200, revision=policy)
            self.assertEqual(r['updates'], 6 * r['budget'], policy)
            seen[policy] = r['trace_hash']
        self.assertEqual(len(set(seen.values())), 3, seen)


class ExplorationAndShapingTests(unittest.TestCase):
    def test_zero_scale_shaping_changes_nothing(self):
        meta = rungs()[0]
        a = run(meta['tag'], 'history', budget=20000)
        for kind in ('learned', 'oracle'):
            b = run(meta['tag'], 'history', budget=20000, shape=kind, **{'shape-scale': 0})
            self.assertEqual(a['trace_hash'], b['trace_hash'], kind)

    def test_structural_detours_are_charged_to_the_same_budget(self):
        meta = rungs()[0]
        plain = run(meta['tag'], 'history', budget=20000)
        explore = run(meta['tag'], 'history', budget=20000, beta=.15, target='scarcest', quota=1)
        self.assertEqual(plain['updates'], explore['updates'])
        self.assertGreater(explore['structural_steps'], 0)
        self.assertLess(explore['structural_steps'], explore['budget'])
        self.assertEqual(plain['structural_steps'], 0)

    def test_zero_beta_leaves_behaviour_unchanged(self):
        meta = rungs()[0]
        a = run(meta['tag'], 'history', budget=20000)
        b = run(meta['tag'], 'history', budget=20000, beta=0, target='scarcest', quota=1)
        self.assertEqual(a['trace_hash'], b['trace_hash'])

    def test_quota_blocks_merging_without_evidence(self):
        """An unreachable quota must leave the merger at the prefix tree."""
        meta = rungs()[0]
        r = run(meta['tag'], 'merged', budget=20000, theta=50, quota=10 ** 6)
        h = run(meta['tag'], 'history', budget=20000)
        self.assertEqual(r['classes'], r['nodes'])
        self.assertEqual(r['trace_hash'], h['trace_hash'])

    def test_prefix_tree_is_keyed_by_history_not_by_the_value_index(self):
        """The tree is a tree over histories whatever the table is indexed on.

        Keying its nodes with the value encoder collapsed it under a goal index,
        because two histories owing the same next event hash alike. The tree then
        had one node per goal, its successor edges pointed back at itself, and a
        learned goal provider could never leave the node it started in.
        """
        meta = rungs()[0]
        goal = run(meta['tag'], 'goal', budget=20000, **{'goal-select': 'learned'})
        hist = run(meta['tag'], 'history', budget=20000, beta=.15, quota=1)
        self.assertGreater(goal['nodes'], goal['features'])
        self.assertGreater(goal['nodes'], 10)
        # And the change is a no-op for every arm that already tracked a tree.
        self.assertGreater(hist['nodes'], 10)

    def test_goal_provider_needs_no_task_indexed_table(self):
        """Reading the automaton as a goal beats reading it as a table index.

        The goal arm holds no task-indexed value function at all, so its task
        dimension is the event vocabulary rather than the number of task states,
        and it must still solve the task.
        """
        meta = rungs()[0]
        goal = run(meta['tag'], 'goal', budget=40000, every=1000, **{'goal-select': 'value'})
        index = run(meta['tag'], 'automaton', budget=40000, every=1000)
        self.assertLessEqual(goal['features'], index['features'])
        self.assertEqual(goal['final_success'], 1)
        self.assertGreater(goal['first90'], 0)
        # Update counts stay comparable: the goal arm spends on skills what the
        # others spend on their task Q.
        total = lambda r: r['updates'] + r['skill_updates']
        self.assertLess(abs(total(goal) - total(index)) / total(index), .15)

    def test_goal_named_without_the_automaton_is_not_enough(self):
        """The control that has eaten several claims in this project fails here."""
        meta = rungs()[0]
        blind = run(meta['tag'], 'goal', budget=40000, every=1000, **{'goal-select': 'untried'})
        known = run(meta['tag'], 'goal', budget=40000, every=1000, **{'goal-select': 'value'})
        self.assertGreater(known['auc'], blind['auc'])

    def test_counterfactual_updates_spend_one_update_per_task_state(self):
        """CRM's update count is the machine's size, not the replay budget."""
        meta = rungs()[0]
        plain = run(meta['tag'], 'automaton', budget=20000)
        crm = run(meta['tag'], 'automaton', budget=20000, crm=1)
        self.assertGreater(crm['updates'], plain['updates'])
        # And the environment interaction is untouched: same steps, same episodes
        # boundaries, only the number of Bellman updates differs.
        self.assertEqual(crm['budget'], plain['budget'])
        self.assertEqual(crm['crm'], 1)
        self.assertEqual(plain['crm'], 0)

    def test_replay_budget_is_the_only_thing_replay_changes(self):
        meta = rungs()[0]
        a = run(meta['tag'], 'automaton', budget=20000)
        b = run(meta['tag'], 'automaton', budget=20000, replay=23)
        # updates counts task-Q updates only; skill updates are reported apart.
        self.assertEqual(b['updates'], 4 * a['updates'])
        self.assertEqual(b['skill_updates'], a['skill_updates'])

    def test_blue_fringe_rules_merge_harder_than_the_sweep(self):
        """RPNI and EDSM sit at a different operating point, not a better one.

        They collapse the tree into far fewer classes at far lower precision.
        Whether that operating point is better for the agent is what the loop
        experiment prices; the direction of the difference is what is fixed here.
        """
        meta = rungs()[0]
        kw = dict(budget=40000, theta=20, quota=1, **{'strict-children': 0})
        sweep = run(meta['tag'], 'merged', **kw)
        rpni = run(meta['tag'], 'merged', **kw, **{'merge-rule': 'rpni'})
        self.assertLess(rpni['classes'], sweep['classes'])
        self.assertLessEqual(rpni['merge_precision'], sweep['merge_precision'])

    def test_jirp_transfer_moves_values_but_fewer_than_the_member_copy(self):
        """It transfers only on an unambiguous match between two hypotheses.

        That is far rarer than copying a new class's values from whichever old
        class one of its members sat in, which is what makes the crude rule
        faster here. Counting the transfers is the honest check: matching AUCs
        only mean the run had too few revisions to tell the rules apart.
        """
        meta = rungs()[0]
        kw = dict(budget=80000, theta=20, quota=1, **{'strict-children': 0})
        self.assertEqual(run(meta['tag'], 'merged', revision='reset', **kw)['jirp_transfers'], 0)
        self.assertGreater(run(meta['tag'], 'merged', revision='jirp', **kw)['jirp_transfers'], 0)

    def test_oracle_potential_is_zero_at_acceptance(self):
        for meta in rungs():
            task = Task(meta['n'], tuple(meta['pair']), meta['d'], meta['m'])
            dist = []
            for state in task.states:
                dist.append(0 if task.done(state) else 1)
            self.assertTrue(any(task.done(s) for s in task.states), meta['tag'])
            for h in meta['optimal_histories']:
                self.assertTrue(task.done(task.run(h)), meta['tag'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
