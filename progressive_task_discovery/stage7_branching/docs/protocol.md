# Stage 7: branching with exact feedback — protocol before results

Archive note: this is the initial, pre-development protocol. The final main experiment inserts one shared event before the different next goals, and the stress test inserts three. The last-event baseline, matched-epsilon intervention and same-map control were added as recorded in `decision_log.md`. The proposed freely chosen branch control became a best-single-branch evaluation diagnostic, not a separately trained policy. Use `REPORT.md` and the saved batch specifications for the final executed design.

Scope: run Stage 7 only. No feedback noise, stochastic motion, context-dependent dynamics, PPO or robot. Preserve all Stage 1–6 code and outputs.

## A branch that cannot be avoided

Merely adding a freely chosen branch does not force history-dependent control: an agent can repeatedly choose one branch. We use two balanced initial locations, alternating in training and equally weighted in evaluation. From either initial location the only exit crosses its cue A or B. Both routes physically join at the same junction. Movement is deterministic; variation in initial context is not transition noise.

The map is a 13×13 grid of walls and 28 walkable cells: horizontal corridor y=1, x=4..8; vertical corridor x=6, y=1..12; horizontal corridor y=6, x=0..12. Start locations (4,1),(8,1); cue cells A=(5,1), B=(7,1); goal cells C=(0,6), D=(12,6), E=(6,12). Every goal is reachable only through central junction J=(6,6). No teleports or intermediate resets.

The hidden task accepts either cue initially. It then requires a different first goal for each cue (two distinct labels from C,D,E). Following that goal, both branches merge to the same remaining suffix. Each route has k progress events including its cue; common suffix labels are sampled from C,D,E with no adjacent repeats, including the first-goal boundary. Agent is not given either target order. The branch is established by the first progress event, not by a privileged hidden cue observation. At J, both histories have the identical position and progress count=1 but different next goals. Each reset starts a genuine new episode at one of the two initial locations.

Wrong events are ignored, progress feedback and labels are exact, reward=progress-.01, horizon=20k initially. All skills and task Q values start from zero. A deterministic stationary policy conditioned only on (position,count) must choose the same excursion from J for both branches; on the branch where that goal is wrong it returns to J at count=1 and repeats. Its greedy-evaluation success is therefore at most 50%. This ceiling is specific to this bottleneck geometry and policy class, **not** a bound on every randomized or memoryful controller. An independent witness/cycle check will verify it. A freely chosen initial branch control will also be evaluated to show how an unbalanced design can hide this problem.

## Representations and methods

* `count_replay`: Q(position, progress count, action), plus five ordinary replay updates per real step.
* `history_replay`: complete successful-event prefix as the task feature, plus five replay updates. Discarding non-progress events is valid under this declared ignore-wrong-events family. This is an unbounded-within-episode sufficient history, not a short last-event window.
* `hand_replay`: an explicitly engineered monitor using observed count and first cue; it distinguishes A/B at count=1 and merges them after the first branch-specific goal. This monitor has privileged knowledge of the branch-merge pattern, but no future target labels and no access to the environment's internal state. It is a strong comparator, not an unknown-structure peer.
* `learned_replay`: incrementally constructed prefix-tree task states and observed positive transitions, no skill use; same task-Q/replay rules as history_replay. These equivalent interfaces must yield identical behavior. In this exact-feedback task, a prefix tree is identifiable directly from successful event history. It is NOT general automaton induction or automatic minimization.
* `delayed_skill`: learned prefix tree and goal-conditioned skill bank; use skills only after successful full episodes have been observed under BOTH source cues.
* `immediate_skill`: same skill bank and updates; use each observed outgoing task edge immediately. If a state has multiple known valid outgoing labels, choose the lowest label deterministically. Unknown edges use task Q exploration.
* `oracle_skill`: true task transition labels and the hand monitor available from the outset; skills still start from zero and are trained from physical interaction. It is a privileged reference, not a mathematical performance bound.
* `history_skill`: a direct history-to-next-goal dictionary with the same behavior as immediate_skill, an exact handwritten-representation control.

All methods use alpha=.3, gamma=.98, zero initialization, common epsilon candidates, four-action epsilon-greedy control. Every physical step causes exactly six Bellman updates: task Q + five goal-conditioned skill updates for skill methods, or task Q + five replay updates for replay methods. Physical trajectories generally differ because policies differ. We do **not** claim different online policies receive identical data; a fixed-data or frozen-bank intervention belongs to Stage 8. The skill code, relabeling semantics and update counts are already shared to remove avoidable implementation differences.

## Validation and metrics

Exact deterministic rollout from both starts at every checkpoint; no evaluation learning. Report balanced success, success per branch, success AUC, first/stable90 with right censoring, training prefix/frontier steps, first success per branch, task-state aliasing on a balanced set of true prefixes (plus the count=1 junction subset), next-goal coverage/precision, wrong bindings, skill-attempt outcomes, transition-discovery times and structure revisions (zero under the conservative exact-feedback prefix learner). State IDs are aligned by optimal many-to-one purity mapping; never compare arbitrary integer IDs directly. Accuracy is evaluation-only and never fed back to learners. Ground-truth state purity is an information diagnostic, not a proof of minimality or algorithmic inference quality.

Before interpreting results: independently verify map transitions and ambiguity witness, reachable optimal routes, 50% greedy count ceiling, learned/history equivalence, update and step budgets, and independently replay saved final policies. H1 is interpreted as the value of distinguishing task histories over count. H2 needs improvement over delayed use AND competent replay/history controls. Neither H1 nor H2 alone establishes automaton syntax or inference novelty.

Development: seeds0..11, k4,8,12, epsilon .005,.02,.05,.2, budget80k, checkpoint1k. Choose one epsilon per method by pooled development AUC, pair equivalent controls to the same choice. Freeze, then evaluate 32 new seeds400..431 over k4,8,12 and H factors16,24. Results determine whether to recommend Stage 8/9; do not automatically begin them.
