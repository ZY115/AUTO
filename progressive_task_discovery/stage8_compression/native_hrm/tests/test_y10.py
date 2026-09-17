import copy
import pickle
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from native_hrm.y10 import StateScopedFormulaBank, IHSAAlgorithmHRLPerState
from reinforcement_learning.ihsa_hrl_tabular_algorithm import FormulaBankTabular
from gym_hierarchical_subgoal_automata.automata.condition import FormulaCondition
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula
import numpy as np


def formula(*literals):
    return FormulaCondition(DNFFormula([list(literals)]))


class RecordingBank(FormulaBankTabular):
    def __init__(self, *args):
        super().__init__(*args)
        self.selections = []
        self.rewards = []

    def _get_subgoals_to_update(self):
        result = super()._get_subgoals_to_update()
        self.selections.append(tuple(map(str, result)))
        return result

    def _get_subgoal_pseudoreward(self, condition, obs, terminal, goal):
        result = super()._get_subgoal_pseudoreward(condition, obs, terminal, goal)
        self.rewards.append((str(condition), result))
        return result


TASK = SimpleNamespace(observation_space=SimpleNamespace(n=6),action_space=SimpleNamespace(n=3))
GOALS = [formula(x) for x in ('a','b','c')]


def bank(selected=2):
    b = RecordingBank(dict(formula_update_sel_num=selected, learning_rate=.3,
                           pseudoreward_after_step=-.01), False)
    for g in GOALS: b.add_q_function(TASK, g)
    for obs in (set(), {'a'}, {'b'}, {'c'}): b.on_task_observation(obs, TASK)
    b.set_active_subgoals(set(GOALS))
    return b


class BankTests(unittest.TestCase):
    def equal_banks(self, a, b):
        self.assertEqual(a._q_function_step_counter,b._q_function_step_counter)
        self.assertEqual(a._q_function_update_counter,b._q_function_update_counter)
        self.assertEqual(a._num_update_calls,b._num_update_calls)
        self.assertEqual(a.selections,b.selections)
        self.assertEqual(a.rewards,b.rewards)
        for g in GOALS: np.testing.assert_array_equal(a.get_q_function(g),b.get_q_function(g))

    def test_collapsed_bank_bitwise_equals_native_with_sampled_updates(self):
        native = bank(); scoped = StateScopedFormulaBank(bank(),collapse=True)
        for i in range(180):
            scoped.bind('m'+str(i%2),'u'+str(i%7))
            obs=[set(),{'a'},{'b'},{'c'}][i%4]
            args=(TASK,i%6,i%3,(i+1)%6,i%11==0,i%22==0,obs)
            np.random.seed(1000+i);native.update_q_functions(*args)
            np.random.seed(1000+i);scoped.update_q_functions(*args)
            native.inc_num_steps(GOALS[i%3]);scoped.inc_num_steps(GOALS[i%3])
            self.equal_banks(native,scoped.current_bank)
        self.assertEqual(len(scoped.banks),1)

    def test_uncollapsed_all_goals_same_sets_and_pseudorewards_only_local_writes(self):
        native=bank(None);scoped=StateScopedFormulaBank(bank(None))
        for j in range(3):scoped.bind('m','u'+str(j))
        for i in range(120):
            scoped.bind('m','u'+str(i%3)); key=scoped.context
            before={k:copy.deepcopy(b._q_functions) for k,b in scoped.banks.items() if k!=key}
            args=(TASK,i%6,i%3,(i+1)%6,i%13==0,False,{'abc'[i%3]})
            native.update_q_functions(*args);scoped.update_q_functions(*args)
            self.assertEqual(native.selections[-1],scoped.current_bank.selections[-1])
            self.assertEqual(native.rewards[-3:],scoped.current_bank.rewards[-3:])
            for k, tables in before.items():
                for g,v in tables.items(): np.testing.assert_array_equal(v,scoped.banks[k]._q_functions[g])
        for b in scoped.banks.values():self.assertEqual(b._num_update_calls,40)
        self.assertEqual(scoped.template._num_update_calls,0)

    def test_counters_values_and_storage_are_isolated(self):
        scoped=StateScopedFormulaBank(bank())
        scoped.bind('m','u0');a=scoped.current_bank
        a.get_q_function(GOALS[0])[0,0]=7
        scoped.inc_num_steps(GOALS[0])
        scoped.bind('m','u1');b=scoped.current_bank
        self.assertFalse(np.shares_memory(a.get_q_function(GOALS[0]),b.get_q_function(GOALS[0])))
        self.assertEqual(scoped.get_num_steps(GOALS[0]),0)
        self.assertEqual(scoped.get_q_function(GOALS[0])[0,0],0)
        self.assertEqual(b._q_function_update_counter[GOALS[0]],0)

    def test_formula_root_sharing_and_split_remain_native_within_each_state(self):
        a,ab=formula('a'),formula('a','~b')
        scoped=StateScopedFormulaBank(RecordingBank({},False))
        for g in (a,ab):scoped.add_q_function(TASK,g)
        scoped.on_task_observation({'a'},TASK)
        for st in ('u0','u1'):
            scoped.bind('m',st)
            self.assertIs(scoped.get_q_function(a),scoped.get_q_function(ab))
        scoped.bind('m','u0');scoped.get_q_function(a)[0,0]=3
        scoped.on_task_observation({'a','b'},TASK)
        for st in ('u0','u1'):
            scoped.bind('m',st)
            self.assertIsNot(scoped.get_q_function(a),scoped.get_q_function(ab))
        scoped.bind('m','u1');self.assertEqual(scoped.get_q_function(a)[0,0],0)

    def test_roundtrip_keeps_all_banks_counters_and_sampling_state(self):
        s=StateScopedFormulaBank(bank())
        for i in range(7):
            s.bind('m','u'+str(i%2));s.update_q_functions(TASK,0,1,2,False,False,{'a'})
            s.inc_num_steps(GOALS[0])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.pkl';s.export_bank(p)
            restored=StateScopedFormulaBank(bank());restored.import_bank(p)
            self.assertIsNone(restored.context)
            for key in s.banks:self.equal_banks(s.banks[key],restored.banks[key])
            collapsed=StateScopedFormulaBank(bank(),True)
            with self.assertRaises(ValueError):collapsed.import_bank(p)
            with p.open('wb') as f:pickle.dump({'bank':{}},f)
            with self.assertRaises(ValueError):restored.import_bank(p)

    def test_unbound_access_fails_instead_of_silently_using_previous_state(self):
        s=StateScopedFormulaBank(bank())
        with self.assertRaises(RuntimeError):s.get_q_function(GOALS[0])

    def test_unvalidated_hierarchy_and_neural_modes_are_rejected(self):
        for config in ({'training_mode':'learn'},
                       {'training_mode':'handcrafted','state_format':'full_obs'},
                       {'training_mode':'handcrafted','state_format':'tabular','use_flat_hierarchy':False}):
            with self.assertRaises(ValueError):IHSAAlgorithmHRLPerState(config)


if __name__=='__main__':unittest.main()
