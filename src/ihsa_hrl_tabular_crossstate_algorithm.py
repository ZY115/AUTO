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

On the equivalence check: with `formula_update_sel_num: null` every bank updates
the same full set of subgoals with no sampling, so N identical tables receiving
identical updates stay identical to one shared table. C-full should then equal
Y11 bit for bit on a fixed experience stream. With sampling on, each bank draws
its own subset and consumes its own RNG, so the streams diverge by design — the
equivalence check only holds in the all-goal setting, and that is its whole point.
"""
from reinforcement_learning.ihsa_hrl_tabular_perstate_algorithm import (
    IHSAAlgorithmHRLTabularPerState,
)
from utils.container_utils import get_param


class IHSAAlgorithmHRLTabularCrossState(IHSAAlgorithmHRLTabularPerState):
    # Broadcast to every legal state's bank (C-full). Turning this off leaves the
    # class behaving exactly as Y10, which is how the arms are kept comparable.
    BROADCAST_KEY = "crossstate_broadcast"

    def __init__(self, params):
        self._broadcast = get_param(
            params, IHSAAlgorithmHRLTabularCrossState.BROADCAST_KEY, True)
        # Set before the parent constructor: it reaches _get_policy_bank.
        self._legal_keys = None
        self._update_calls = 0          # TD-update calls issued, summed over banks
        self._broadcast_steps = 0       # environment steps that triggered a broadcast
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

        # Iterate the banks directly rather than moving _perstate_key, so the key
        # the behaviour policy set in _choose_action is never disturbed.
        for bank in self._perstate_banks[task_id].values():
            bank.update_q_functions(task, state, action, next_state, is_terminal,
                                    is_goal_achieved, observation)
            self._update_calls += 1

    def _on_initial_observation(self, observation):
        # The parent already forwards to every bank that exists; with
        # pre-creation that is every legal state, which is the point.
        super()._on_initial_observation(observation)

    def update_counts(self):
        """For the cost accounting: TD-update calls and the environment steps that
        produced them. Skill-execution counts live in the banks and are untouched
        by broadcasting."""
        return {
            "td_update_calls": self._update_calls,
            "broadcast_steps": self._broadcast_steps,
            "banks": self.num_perstate_banks(),
            "legal_state_keys": len(self._legal_keys or []),
        }


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
