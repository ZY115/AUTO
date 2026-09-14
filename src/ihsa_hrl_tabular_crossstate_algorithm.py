"""C-full: independent parameters per automaton state, plus cross-state experience broadcast.

The third arm. Y11 shares one skill bank across every automaton state; Y10 gives
each state its own bank and updates only the current one. Those two differ in
**two** things at once — whether parameters are shared, and whether one piece of
experience can reach the policies used in other task states — so `Y11 - Y10`
cannot be read as the value of parameter sharing alone.

C-full separates them: parameters stay independent exactly as in Y10, but every
environment experience is applied to **every legal state's bank**. What it asks
is whether extra computation can substitute for sharing.

    arm      parameters              each experience updates      per-step budget
    ------   ---------------------   --------------------------   ---------------
    Y10      one bank per state      the current state's bank     bank's own rule
    Y11      one shared bank         the shared bank              bank's own rule
    C-full   one bank per state      *every* legal state's bank   each bank's own rule

This is a **cross-state experience-propagation control**, not a reproduction of
Icarte et al.'s HRL: the option definition, the pseudorewards and the termination
semantics all stay as the HRM implementation has them. Only the propagation scope
changes.

Three things the implementation has to get right, each of which would otherwise
silently measure something else:

**Unvisited states must receive updates too.** Banks are pre-created for every
legal (automaton, state) pair the first time the hierarchy is in scope, not
lazily on first visit. Broadcasting only to banks that already exist would
measure "sharing after the state has been reached", which is a different and
weaker mechanism.

**The pseudoreward must be correct for the receiving state.** It is — and this is
checkable rather than assumed. `FormulaBank._get_subgoal_pseudoreward` is a
function of `(formula, observation, is_terminal, is_goal_achieved)` only; none of
those depends on the automaton state. `is_terminal` and `is_goal_achieved` come
from the environment transition and are identical for every bank. So the same
experience carries the same pseudoreward into every bank, and no per-state
recomputation is needed. `check_pseudoreward_is_state_independent()` asserts this
against the live class rather than trusting the reading.

**An update is not an execution.** `_inc_formula_q_function_step_count` still goes
through `_get_policy_bank`, i.e. only the current state's bank, so broadcasting
does not inflate skill-execution counts or accelerate epsilon decay. Only
`_q_function_update_counter` moves, which is what should move. The two counts are
reported separately.

**Equivalence holds for the Q tables, not for the run.** With every bank updating
the same subgoal set from the same experience, N identical tables stay identical
to one shared table; that is provable and the check confirms it. It does **not**
follow that C-full and Y11 produce the same trajectory, because at least three
things still differ:

  1. *Independent sampling.* With `formula_update_sel_num` set, each bank draws
     its own subset. `SHARED_SAMPLING_KEY` turns this off for the controlled check.
  2. *The global RNG stream.* Those draws come from NumPy's global stream, so
     extra sampling shifts every later exploration draw too.
  3. *The exploration schedule.* `_get_formula_exploration_rate` anneals on
     `_get_formula_q_function_step_count`, which goes through `_get_policy_bank`
     — the **current state's** bank. Y11 accumulates one execution count per
     subgoal across all states; C-full counts per state. Measured on a 60-episode
     run: the two arms' counts already differ on 46% of exploration queries, first
     diverging at query 36 (Y11 36, C-full 0).

Keeping (3) is deliberate — inflating execution counts by broadcasting would be
wrong — but it means a whole-run equality can only be demanded after the
exploration schedule and both RNG streams are aligned as well. A short run whose
episode summaries match is weaker evidence than it looks: at 150 episodes the
exploration rate is still ~0.9997 and the per-arm difference sits in the fifth
decimal, so it rarely changes a drawn action. That is an artifact of the run
length, not equivalence.
"""
from reinforcement_learning.ihsa_hrl_tabular_perstate_algorithm import (
    IHSAAlgorithmHRLTabularPerState,
)
import numpy as np

from utils.container_utils import get_param


class IHSAAlgorithmHRLTabularCrossState(IHSAAlgorithmHRLTabularPerState):
    # Broadcast to every legal state's bank (C-full). Turning this off leaves the
    # class behaving exactly as Y10, which is how the arms are kept comparable.
    BROADCAST_KEY = "crossstate_broadcast"

    # Controlled-acceptance only: force every bank to update the *same* sampled
    # subgoal set, chosen once per step from the acting state's bank. This removes
    # difference (1) above and the extra RNG draws of (2). It is not an experiment
    # setting — the real C-full keeps each bank's own sampling, as agreed.
    SHARED_SAMPLING_KEY = "crossstate_shared_sampling"

    def __init__(self, params):
        self._broadcast = get_param(
            params, IHSAAlgorithmHRLTabularCrossState.BROADCAST_KEY, True)
        self._shared_sampling = get_param(
            params, IHSAAlgorithmHRLTabularCrossState.SHARED_SAMPLING_KEY, False)
        # Set before the parent constructor: it reaches _get_policy_bank.
        self._legal_keys = None
        self._update_calls = 0          # bank-level update calls, summed over banks
        self._scalar_updates = 0        # individual (subgoal, s, a) Q-value writes
        self._broadcast_steps = 0       # environment steps that triggered a broadcast
        self._last_broadcast_banks = [] # which bank keys the last step reached
        self._last_broadcast_goals = [] # which subgoals each of them updated
        self._shared_rngs = {}          # per task, seeded exactly as Y11's bank
        super().__init__(params)

    def _legal_state_keys(self, domain_id):
        """Every (automaton, state) pair an agent can act in, terminal states
        excluded. Computed once; the hierarchy does not change during a run."""
        if self._legal_keys is not None:
            return self._legal_keys
        hierarchy = self._get_hierarchy(domain_id)
        keys = []
        for name in hierarchy.get_automata_names():
            automaton = hierarchy.get_automaton(name)
            for state in automaton.get_states():
                if automaton.is_terminal_state(state):
                    continue
                keys.append((name, state))
        self._legal_keys = sorted(keys)
        return self._legal_keys

    def _ensure_banks(self, task_id, domain_id):
        """Pre-create every legal state's bank. Going through _get_policy_bank
        reuses the parent's deepcopy-from-template path, so a bank created here is
        identical to one the parent would have created on first visit."""
        saved = self._perstate_key
        try:
            for key in self._legal_state_keys(domain_id):
                self._perstate_key = key
                self._get_policy_bank(task_id)
        finally:
            self._perstate_key = saved

    def _update_q_functions(self, domain_id, task_id, state, action, next_state,
                            is_terminal, observation):
        if not self._broadcast or not self._perstate_banks:
            return super()._update_q_functions(
                domain_id, task_id, state, action, next_state, is_terminal, observation)

        self._ensure_banks(task_id, domain_id)
        task = self._get_task(domain_id, task_id)
        is_goal_achieved = task.is_goal_achieved()
        self._broadcast_steps += 1
        banks = self._perstate_banks[task_id]

        fixed = None
        if self._shared_sampling:
            # Draw once and make every bank use that same set.
            #
            # The draw must come from a stream seeded **exactly as Y11's single bank
            # is**, not from whichever bank happens to be acting. Each per-state bank
            # carries its own seed (it has to, or independent sampling would not be
            # independent), so drawing from the acting bank would make C-full's
            # subset sequence differ from Y11's from the first step — and then the
            # two arms are not comparable even though every C-full bank agrees with
            # every other. That mistake cost one failed equivalence check.
            #
            # The draw still needs a bank to read _active_root_sat_subgoals and the
            # update counters from; all banks agree on those under shared sampling,
            # so the acting one is as good as any. Only the *stream* is replaced.
            acting = banks.get(self._perstate_key) or next(iter(banks.values()))
            if task_id not in self._shared_rngs:
                self._shared_rngs[task_id] = np.random.default_rng(
                    self.seed_value + 7919 * task_id)
            saved = acting._update_rng
            acting._update_rng = self._shared_rngs[task_id]
            try:
                fixed = list(acting._get_subgoals_to_update())
            finally:
                acting._update_rng = saved
            for b in banks.values():
                missing = [g for g in fixed if not b._has_q_function(b.get_root(g).get_formula_condition())]
                if missing:
                    raise AssertionError(
                        f"shared sampling picked {len(missing)} subgoal(s) a bank "
                        "does not hold; the banks' active sets have diverged")

        self._last_broadcast_banks, self._last_broadcast_goals = [], []
        # Iterate the banks directly rather than moving _perstate_key, so the key
        # the behaviour policy set in _choose_action is never disturbed.
        for key, bank in banks.items():
            if fixed is not None:
                bank._get_subgoals_to_update = lambda _f=fixed: _f
            try:
                before = sum(bank._q_function_update_counter.values())
                bank.update_q_functions(task, state, action, next_state, is_terminal,
                                        is_goal_achieved, observation)
                written = sum(bank._q_function_update_counter.values()) - before
            finally:
                if fixed is not None:
                    del bank._get_subgoals_to_update
            self._update_calls += 1
            self._scalar_updates += written
            self._last_broadcast_banks.append(key)
            self._last_broadcast_goals.append(written)

    def _on_initial_observation(self, observation):
        # The parent already forwards to every bank that exists; with
        # pre-creation that is every legal state, which is the point.
        super()._on_initial_observation(observation)

    def update_counts(self):
        """For the cost accounting: TD-update calls and the environment steps that
        produced them. Skill-execution counts live in the banks and are untouched
        by broadcasting."""
        return {
            "bank_update_calls": self._update_calls,
            "scalar_td_updates": self._scalar_updates,
            "broadcast_steps": self._broadcast_steps,
            "banks": self.num_perstate_banks(),
            "legal_state_keys": len(self._legal_keys or []),
        }

    def execution_counts_by_bank(self, task_id=0):
        """Per-bank skill-execution counts. These are what the exploration rate
        anneals on, and they are *supposed* to differ from Y11's single shared
        count — that difference is a live channel, not a bug, and it has to be
        reported rather than left implicit."""
        return {key: dict(bank._q_function_step_counter)
                for key, bank in (self._perstate_banks or {}).get(task_id, {}).items()}


def check_pseudoreward_is_state_independent():
    """Assert the claim the broadcast rests on: the subgoal pseudoreward does not
    read the automaton state. Checked against the live signature, so it fails
    loudly if upstream ever threads the task state through."""
    import inspect
    from reinforcement_learning.ihsa_hrl_algorithm import FormulaBank

    sig = inspect.signature(FormulaBank._get_subgoal_pseudoreward)
    params = [p for p in sig.parameters if p != "self"]
    expected = ["formula", "observation", "is_terminal", "is_goal_achieved"]
    if params != expected:
        raise AssertionError(
            f"pseudoreward signature changed: {params} != {expected}. "
            "The broadcast assumes it cannot depend on the automaton state; "
            "re-derive it per receiving state before trusting C-full.")
    src = inspect.getsource(FormulaBank._get_subgoal_pseudoreward)
    for bad in ("automaton_state", "hierarchy_state", "state_name"):
        if bad in src:
            raise AssertionError(
                f"pseudoreward body now mentions {bad!r}; the broadcast is no "
                "longer sound as written.")
    return True
