"""Y10 for the neural arm: the goal reading with skill reuse switched off.

The tabular counterpart lives in `ihsa_hrl_tabular_perstate_algorithm.py` and the
idea is the same — hold one whole formula bank per automaton state instead of
re-keying the bank's dictionaries, so the subsumption roots, the two counters and
the export format all separate on their own and the only difference from the
parent arm is sharing *across* automaton states.

**One choice here has no tabular counterpart and has to be declared.** The DQN
bank owns its replay buffer, so splitting banks would split the buffer too. With
`er_start_size = 100000` — the authors' own setting — a state visited a fifth of
the time would not begin learning until half a million steps, and a rarely
visited one might never begin at all. Y10 would then lose for a bookkeeping
reason rather than because it cannot share. So by default **the buffer is shared
across the per-state banks**: every bank sees the same transitions and the same
start threshold, and only the networks, target networks and optimizers are
separate. That isolates parameter sharing, which is the factor under test, and it
is the strongest fair form of the arm.

`perstate_split_buffer` restores the other reading, where each state also keeps
its own buffer. It is the more literal "nothing is shared", and it is available
so the two can be reported side by side rather than argued about.

A per-state bank still only *updates* while its state is active, in both
readings. That is not a confound to remove: it is what not sharing means.
"""
import copy
from reinforcement_learning.ihsa_hrl_dqn_algorithm import IHSAAlgorithmHRLDQN
from utils.container_utils import get_param


class IHSAAlgorithmHRLDQNPerState(IHSAAlgorithmHRLDQN):
    COLLAPSE_KEY = "perstate_collapse"
    SPLIT_BUFFER = "perstate_split_buffer"

    def __init__(self, params):
        self._perstate_collapse = get_param(params, IHSAAlgorithmHRLDQNPerState.COLLAPSE_KEY, False)
        self._perstate_split_buffer = get_param(params, IHSAAlgorithmHRLDQNPerState.SPLIT_BUFFER, False)
        self._perstate_key = None
        self._perstate_banks = None
        super().__init__(params)
        self._perstate_template = self._formula_bank
        self._perstate_banks = {}

    def _get_policy_bank(self, task_id):
        if self._perstate_banks is None or self._perstate_key is None:
            return self._formula_bank
        if self._perstate_key not in self._perstate_banks:
            bank = copy.deepcopy(self._perstate_template)
            if not self._perstate_split_buffer:
                # Same object, so every bank appends to and samples from one
                # stream of experience. Only the parameters are separate.
                bank._er_buffer = self._perstate_template._er_buffer
            self._perstate_banks[self._perstate_key] = bank
        return self._perstate_banks[self._perstate_key]

    def _get_policy_banks(self):
        return [self._formula_bank] + list((self._perstate_banks or {}).values())

    def _choose_action(self, domain_id, task_id, state, hierarchy, hierarchy_state):
        if self._perstate_collapse:
            self._perstate_key = ("collapsed", "collapsed")
        else:
            self._perstate_key = (hierarchy_state.automaton_name, hierarchy_state.state_name)
        return super()._choose_action(domain_id, task_id, state, hierarchy, hierarchy_state)

    def _on_initial_observation(self, observation):
        super()._on_initial_observation(observation)
        for bank in (self._perstate_banks or {}).values():
            bank.on_task_observation(observation, self._get_task(0, 0))

    def _export_policy_banks(self):
        """One file per automaton state; the parent's format holds a single bank,
        so the key goes into the filename rather than into the format."""
        for key, bank in (self._perstate_banks or {}).items():
            base = self._get_formula_bank_model_path(IHSAAlgorithmHRLDQN.EXPORT_MODEL_EXTENSION)
            stem, ext = base.rsplit(".", 1)
            safe = "-".join(str(k) for k in key).replace("/", "_")
            bank.export_bank(f"{stem}-{safe}.{ext}")

    def _import_policy_banks(self):
        raise NotImplementedError(
            "import is not needed for the exploitation runs this arm is used in")

    def num_perstate_banks(self):
        return len(self._perstate_banks or {})

    def shares_buffer(self):
        return not self._perstate_split_buffer
