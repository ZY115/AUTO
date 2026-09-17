"""Y10: the goal reading with skill reuse switched off.

The authors' HRL keeps one Q-function per formula condition, shared by every
automaton state that asks for that subgoal (and, through the formula tree's
subsumption roots, shared more widely still). That is the arm this project calls
Y11. Y10 is the same reading of the same hierarchy with that one sharing removed:
a separate execution policy per (automaton, automaton state, formula condition).

Rather than re-key the bank's dictionary — which would also have to re-key the
subsumption roots, the two counters, and the export format, and would risk
changing more than one thing — this keeps the bank class exactly as it is and
holds **one whole bank per automaton state**. Consequences, each of which is the
intended one and none of which is extra:

  * the subsumption sharing inside a state is preserved, so the only difference
    from Y11 is sharing *across* states;
  * intra-option updates reach only the current state's bank, because
    `update_q_functions` is called on whichever bank `_get_policy_bank` returns;
  * the per-step update budget is unchanged: one bank is updated per step, and it
    samples the same `formula_update_sel_num` subgoals it always did;
  * the step and update counters separate automatically, so the sampling
    probabilities in one state are not skewed by another state's updates.

The banks are copied lazily from the fully initialised shared bank the parent
builds, so every state starts from an identical structure with zeroed values.

The key comes from `_choose_action`, which is the only place the hierarchy state
is in scope. It is called immediately before `_update_q_functions` in the same
loop iteration, so the stored key is the state the action was actually taken in.
"""
import copy
from reinforcement_learning.ihsa_hrl_tabular_algorithm import IHSAAlgorithmHRLTabular
from utils.container_utils import get_param


class IHSAAlgorithmHRLTabularPerState(IHSAAlgorithmHRLTabular):
    # When set, the key never changes, so every automaton state shares one bank.
    # That configuration must reproduce the parent arm exactly, and is the check
    # that this class removes sharing and nothing else.
    COLLAPSE_KEY = "perstate_collapse"

    def __init__(self, params):
        self._perstate_collapse = get_param(params, IHSAAlgorithmHRLTabularPerState.COLLAPSE_KEY, False)
        # Both must exist before the parent constructor runs: it calls
        # _init_formula_q_functions, which goes through _get_policy_bank.
        self._perstate_key = None
        self._perstate_banks = None
        super().__init__(params)
        self._perstate_templates = self._formula_banks
        self._perstate_banks = {task_id: {} for task_id in range(self.num_tasks)}

    def set_perstate_key(self, key):
        """Exposed so the equivalence check can pin the key to a constant, which
        must reproduce Y11 bit for bit."""
        self._perstate_key = key

    def _get_policy_bank(self, task_id):
        if not self._perstate_banks or self._perstate_key is None:
            return self._formula_banks[task_id]
        banks = self._perstate_banks[task_id]
        if self._perstate_key not in banks:
            banks[self._perstate_key] = copy.deepcopy(self._perstate_templates[task_id])
        return banks[self._perstate_key]

    def _get_policy_banks(self):
        out = list(self._formula_banks)
        for task_banks in (self._perstate_banks or {}).values():
            out.extend(task_banks.values())
        return out

    def _choose_action(self, domain_id, task_id, state, hierarchy, hierarchy_state):
        if self._perstate_collapse:
            self._perstate_key = ("collapsed", "collapsed")
        else:
            self._perstate_key = (hierarchy_state.automaton_name, hierarchy_state.state_name)
        return super()._choose_action(domain_id, task_id, state, hierarchy, hierarchy_state)

    def _on_initial_observation(self, observation):
        super()._on_initial_observation(observation)
        for task_id, task_banks in (self._perstate_banks or {}).items():
            for bank in task_banks.values():
                bank.on_task_observation(observation, self._get_task(0, task_id))

    def _export_policy_banks(self):
        """One file per (task, automaton state). The parent's format holds a
        single bank per task, so the key is folded into the filename rather than
        changing the format, which keeps the parent's loader usable per state."""
        for task_id, task_banks in (self._perstate_banks or {}).items():
            for key, bank in task_banks.items():
                bank.export_bank(self._perstate_model_path(task_id, key))

    def _import_policy_banks(self):
        raise NotImplementedError(
            "import is not needed for the exploitation runs this arm is used in")

    def _perstate_model_path(self, task_id, key):
        base = self._get_formula_bank_model_path(
            IHSAAlgorithmHRLTabular.EXPORT_MODEL_EXTENSION, task_id)
        stem, ext = base.rsplit(".", 1)
        safe = "-".join(str(k) for k in key).replace("/", "_")
        return f"{stem}-{safe}.{ext}"

    def num_perstate_banks(self):
        return sum(len(b) for b in (self._perstate_banks or {}).values())
