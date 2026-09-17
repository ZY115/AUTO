"""State-scoped native tabular skills, retaining the author's update algorithm.

One untrained template holds formula/observation metadata. Each (automaton,
state) owns a deep copy of the native bank, including both formula counters and
the sampler call counter. Metadata is broadcast; learned values never are.
FormulaTree root sharing is preserved within each bank.
"""
import copy
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'external/hrm-learning/src'))

from reinforcement_learning.ihsa_hrl_tabular_algorithm import IHSAAlgorithmHRLTabular


class StateScopedFormulaBank:
    FORMAT = 'native-hrm-state-scoped-v1'

    def __init__(self, template, collapse=False):
        self.template = template
        self.collapse = bool(collapse)
        self.banks = {}
        self.context = None

    def bind(self, automaton_name, automaton_state):
        self.context = ('*', '*') if self.collapse else (automaton_name, automaton_state)
        if self.context not in self.banks:
            # The template has never learned a transition or advanced a counter.
            self.banks[self.context] = copy.deepcopy(self.template)

    @property
    def current_bank(self):
        if self.context is None:
            raise RuntimeError('Bind the pre-transition task state before reading/updating a skill.')
        return self.banks[self.context]

    def add_q_function(self, task, condition):
        self.template.add_q_function(task, condition)
        for bank in self.banks.values():
            bank.add_q_function(task, condition)

    def on_task_observation(self, observation, task):
        # Sharing observable/formula semantics does not share Q values. Calling
        # the native routine also handles root splits/merges independently.
        self.template.on_task_observation(observation, task)
        for bank in self.banks.values():
            bank.on_task_observation(observation, task)

    def set_active_subgoals(self, subgoals):
        self.template.set_active_subgoals(subgoals)
        for bank in self.banks.values():
            bank.set_active_subgoals(subgoals)

    def get_observations(self):
        return self.template.get_observations()

    def get_root_sat_subgoals(self, subgoals, roots):
        return self.template.get_root_sat_subgoals(subgoals, roots)

    def get_q_function(self, condition):
        return self.current_bank.get_q_function(condition)

    def get_num_steps(self, condition):
        return self.current_bank.get_num_steps(condition)

    def inc_num_steps(self, condition):
        return self.current_bank.inc_num_steps(condition)

    def update_q_functions(self, task, state, action, next_state, is_terminal,
                           is_goal_achieved, observation):
        self.on_task_observation(observation, task)
        # Native routine sees the already-observed symbol (a no-op) then does
        # precisely its own sampling, pseudoreward, TD update and counting.
        return self.current_bank.update_q_functions(
            task, state, action, next_state, is_terminal, is_goal_achieved, observation)

    def export_bank(self, path):
        payload = dict(format=self.FORMAT, collapse=self.collapse,
                       template=self.template, banks=self.banks)
        with open(path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def import_bank(self, path):
        # As with native model files, only load trusted, locally produced pickle.
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or payload.get('format') != self.FORMAT:
            raise ValueError('Not a state-scoped bank; native Y11 models are incompatible.')
        if payload['collapse'] != self.collapse:
            raise ValueError('Saved state-collapse mode differs from this configuration.')
        expected = set(self.template.formula_nodes)
        if set(payload['template'].formula_nodes) != expected:
            raise ValueError('Saved formula vocabulary differs from the current task.')
        self.template, self.banks = payload['template'], payload['banks']
        self.context = None


class IHSAAlgorithmHRLPerState(IHSAAlgorithmHRLTabular):
    """Native flat, handcrafted HRL with only state-local skill banks.

The native metacontrollers, option logic and training loop are inherited.
Binding happens before action selection, so terminal/progress transitions and
the subsequent step counters belong to the state that generated the action.
"""

    def __init__(self, params):
        if params.get('training_mode') != 'handcrafted':
            raise ValueError('Y10 v1 requires training_mode=handcrafted.')
        if params.get('state_format') != 'tabular':
            raise ValueError('Y10 v1 requires state_format=tabular.')
        if not params.get('use_flat_hierarchy', False):
            raise ValueError('Y10 v1 requires use_flat_hierarchy=true; nested call contexts are not validated.')
        self._skill_state_collapse = bool(params.get('skill_state_collapse', False))
        super().__init__(params)

    def _init_formula_q_functions(self):
        # Parent constructor has made empty native banks, but has not yet added
        # formulas, observations or loaded models. Wrap at that precise point.
        self._formula_banks = [StateScopedFormulaBank(b, self._skill_state_collapse)
                               for b in self._formula_banks]
        super()._init_formula_q_functions()

    def _choose_action(self, domain_id, task_id, state, hierarchy, hierarchy_state):
        self._get_policy_bank(task_id).bind(hierarchy_state.automaton_name,
                                             hierarchy_state.state_name)
        return super()._choose_action(domain_id, task_id, state, hierarchy, hierarchy_state)
