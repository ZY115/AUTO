# Frozen initial protocol — 2026-09-09

## Question and assumptions

Does immediate use of partial task structure improve sample efficiency beyond a competent agent with the same events, reward and history memory? Decompose (a) task-state memory, (b) exploration scheduling, and (c) reuse of observed transitions across known task states.

The physical environment is a 9 x 9 open GridWorld, start at its center, four cardinal actions, eight labeled event cells on its boundary. With probability `slip`, the selected action is replaced by a uniformly sampled cardinal action. An event occurs only on entry to an event cell; staying against a wall does not emit another event. Wrong/out-of-order events do nothing. The hidden task is a random length-k ordered sequence of labels; adjacent equal labels are excluded. Successful progress pays +1; each physical step costs .01, including the success step. There is no extra terminal bonus. Episodes end at task completion or H steps. H is initially 24k. No resets to intermediate states, demonstrations, privileged paths, or free prefix execution.

Each unknown-task agent gets only position, observed event, scalar progress bit, and done. It knows the linear, deterministic, ignore-wrong-events task family and event vocabulary, but does not receive the target order. Its task-state counter is updated from its own feedback. A discovered edge records the event, destination stage and whether completion occurred. This is deliberately **prefix identification**, not general automaton inference. In this task family a progress counter is already a sufficient task-state statistic.

## Initial methods

* `flat`: Q(position, action), epsilon .2. Weak memoryless diagnostic, not main evidence.
* `count`: Q(position, observed progress count, action), epsilon .2. Main memory-aware baseline.
* `discovered`: same learner as count; records a partial chain without changing behavior. Must be identical to count under matched randomness.
* `oracle_state`: true task state provided, otherwise count. Must also be identical to count here. This tests whether an oracle-state advantage even exists under informative feedback.
* `progressive`: count learner; epsilon .5 at unmastered stages and .02 at mastered stages. A stage is mastered after at least 5 recent successes, with success fraction at least .8 over up to 10 most recent attempts. An attempt ends on progress or episode end. Previously mastered stages can unlock after failures. Q values continue updating: reuse does not mean irrevocable freezing.
* `count_frontier`: identical scheduling based on count and observed outcomes, with no use of learned event transitions. Must match progressive exactly. Tests whether explicit discovered labels are needed for this mechanism.
* `learned_qrm`: uniform epsilon .2; additionally reuses each observed physical transition at other already discovered task stages, using their identified next-event labels to compute counterfactual progress. This is a strong incremental-structure baseline, not a reproduction of JIRP.
* `progressive_qrm`: learned_qrm with progressive exploration schedule.
* `oracle_qrm`: full target order available for counterfactual updates from the beginning. A privileged comparison, not a mathematical upper bound.

All task-Q learners use alpha .3, gamma .98, zero initialization, identical reward and horizon. Counterfactual updates are online, one per other known stage, in ascending order. They do not synthesize physical transitions or unobserved rewards for unknown edges. Computational update counts are reported separately from environment steps. At time limits Q-learning bootstraps (time-limit truncation); true task completion has zero bootstrap. The finite evaluation horizon remains part of the performance metric. Time is not a policy input in this initial stationary-policy pilot.

## Evaluation and accounting

Initial development seeds 0..7, k=2,4,6,8, slip=0, training budget 200k steps; checkpoint every 5k steps. Seeds couple task order and training random draws across methods, without promising identical trajectories once policies differ. Training continues to the full fixed budget to avoid selection bias. Episodes can be truncated at the last budget step for accounting.

At checkpoints, freeze the greedy policy (lowest-action tie break) and compute its complete-task success probability by exact finite-horizon dynamic programming using the evaluation-only ground-truth task. Evaluation is read-only and cannot feed confidence, discovery or policy updates. Report evaluation-equivalent expected steps separately, not as training interactions. Primary outcome: first checkpoint with success >=.9; also require two successive >=.9 checkpoints as a stability diagnostic. Runs not reaching .9 are right-censored at budget: capped threshold cost is a budgeted score, not an uncensored mean time-to-solution. Report solved fraction, success AUC and final success. Analysis will use paired seed bootstrap intervals.

## Adaptive follow-up policy

After initial data, write the interpretation and next-stage design before running it. Candidate axes: global exploration tuning, frontier/mastery parameters, horizon slack, actuation noise, and shared goal-conditioned skills. Choose follow-ups to challenge or explain observations, including null/adverse results. Select configurations on development seeds, then freeze and test on new seeds. Do not claim exponential-to-linear scaling from four finite task lengths. Do not escalate to PPO/robot unless the mechanism survives strong controls.

## Required validation

Independent event-trace oracle versus monitor; reward/event semantics; exact evaluator versus independent deterministic rollout and Monte Carlo under noise; no unknown-edge counterfactual updates; identical action/Q fingerprints for equivalent controls; reproducible rerun; complete budget accounting. Tests are required here because experimental conclusions depend on simulator and metric correctness.
