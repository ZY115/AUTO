# Stage 8 execution decisions

## Before Gate
Preserve PLAN.md verbatim and all Stage 1–7 code/data. Add hand_skill only in a copied Stage7 simulator. Its Q index and next-goal discovery index both use the hand monitor; targets are learned only from observed progress, never from oracle sequences. Same skill code, six updates per step, epsilon .02 matched to immediate_skill, exact original held-out and same-map seeds/configurations. 384 new runs. Existing Stage7 immediate_skill is the paired control. No merging implementation until Gate result.

## Gate result and sparse calibration (before runs)
Gate main cell hand_skill mean first90=4406.25 versus immediate_skill=7375 (32 paired seeds). Follow the full-plan branch. This gate changes both value and binding indexes, so it is a combined benefit, not a pure Q-compression intervention.

Before changing task family, calibrate terminal-only task reward on the unchanged branching environment. Scan gamma=.98,.99,.995 and terminal reward/(cost*shortest length)=.5,2,8; matched epsilon=.02, alpha=.3, six updates, seeds0..11, k4/12,H24k. Run 200k trajectories and assess the SAME runs at 50k/100k/200k rather than spend three times the data. Methods: full history replay, hand replay, immediate skills. Goal-conditioned skill pseudo-rewards remain their original event relabeling, not extra environment progress rewards. Choose gamma/reward by pooled AUC at100k, tie-break toward smaller reward then gamma. Budget is smallest tested budget with at least90% of runs solved in EVERY method/length cell for the selected setting; otherwise stop. No confirmation seeds used for selection.

Check the requested permutation-start coverage before writing merging code. In a static grid with one fixed cell per event, a first-event cell determines the optimal remaining event order independently of starting position. Thus at most n distinct uniquely optimal orders can be forced by starts, below n! for n>=3. This is being checked explicitly, rather than silently weakening uniqueness or encoding the requested order in dynamics.

## Frozen sparse calibration → independent confirmation
Selected gamma=.99, reward/cost ratio8, budget200k via the predeclared criterion. Freeze before seeds600..631. Confirm k4/12,H24k with full-history/prefix replay, hand replay, immediate, hand skill, delayed skill; all epsilon=.02 and six updates. This is a branching-task sparse-reward confirmation, not yet the unordered-task H4–H6 result.

## Feasible n=2 pilot (before runs)
Proceed with n=2 while keeping the incompatible n=3/4 start-coverage question open. First visit A and B in either order, then a fixed random C/D/E suffix of length m=0,4,8,16. Original static map has both uniquely optimal orders, different starts, overlapping post-set physical routes, no teleport. m+2 events; optimal unordered route length=3+12m. Fixed-order same-map control requires A then B from either start. Its right start costs two additional steps; terminal reward remains identical for paired semantics and is normalized to the unordered shortest length. The sparse parameters are frozen from the branching calibration, not retuned on confirmation data.

Pilot only F0(position),F1(count),F2(last successful event),H(history),P(prefix),O(true set/suffix state),P+S and hand_skill; no learned merging yet. For n=2, last event plus count is sufficient and should match O behavior although Q-table layout differs. Raw replay stores physical transition/event/progress/base reward/episode/time; labels are reconstructed on demand. This pilot cannot test the requested n>=3 factorial scaling.

## n=2 pilot outcome → bounded confirmation
H/P remain exact. At m4/8, oracle-state replay and full-history replay separate, with all12 seeds solved; m16 exceeds the sparse budget (no H or O greedy policy solves), so it is retained as a censored exploratory condition, not assigned artificial successful learning times. H m16 has no complete training successes; only1/12 O runs saw both starts complete. Do not treat that absence of terminal data as evidence against compression.

Freeze all parameters and confirm the measurable m0/4/8 cells on seeds700..731, both unordered and fixed-order semantics, six methods (count,last-event,H,O,immediate,hand-skill). Add visit counts and effective state counts defined as inverse Simpson concentration, with entropy perplexity as sensitivity measure. Last-event is sufficient in n=2 but retains an unnecessary distinction immediately after the unordered set, so its table is not exactly O. No inference that this solves the n3 coverage gate.

The full n3/n4 compression scaling is blocked under single-cell events. Before implementing M and H6, the remaining design must also distinguish ordinary raw-buffer replay (already reconstructs current labels) from an extra version-triggered replay intervention; those cannot honestly be called no-reparse versus reparse if both perform it. No learned-merging or H5/H6 gain is claimed by these prerequisites.

## Layout search outcome → denominator correction and variant B (before further runs)

Five layouts found, optimal-trajectory compression 1.29 to 5.67, funnel verified, and short-window witnesses present at every width below the minimal sufficient window, all at a shared junction cell with differing completed sets. Two problems make those targets the wrong axis.

Scope. Sufficiency was verified over reachable histories, while the layout `compression` field counts prefixes of optimal trajectories only. A denominator must be counted over the same set it is sufficient over. Recomputed over reachable histories, the pure-unordered ladder against the smallest sufficient generic representation is 2.11, 4.53, 4.15, 4.15, 3.62: not monotone, and layouts 3 and 4 both have n=4, m=4, so they are one point rather than two.

Bag sufficiency. In the pure unordered family the bag of completed events equals the minimal automaton exactly; bag/|Q| is 1.00 in all five layouts. A generic bag feature therefore attains the minimal representation with no structure learning at all, so H4 as posed cannot separate learned merging from a feature choice. This is the Stage 7 hand-monitor problem with the privilege removed, which makes it worse rather than better.

Variant B. The required suffix branch depends on which of two designated unordered labels was completed first. Bag is then never sufficient, and for m>=2 no bounded window is sufficient either, because the deciding evidence precedes the suffix by more than any window width. Compression over full history survives: 2.00 to 7.14 over the usable range.

Over reachable histories the honest ratio depends only on n and m, not on the map. Targets are therefore chosen analytically and map search is reduced to forcing, witness and funnel properties. The n=2 pilot already showed m=16 exceeds the frozen sparse budget, so m<=8. Proposed ladder: n in {3,4}, m in {0,2,4,8}, giving 1.60, 2.00, 2.22, 2.46, 3.25, 4.71, 5.75, 7.14.

Every layout reports three scopes: reachable, optimal-trajectory, and after runs the visitation-weighted effective count already planned as inverse Simpson. Headline axis is the reachable-scope honest ratio; the visitation-weighted count is the empirical check on the cost law, not a substitute for it.

Variant A is retained as the upper-bound case in which the correct abstraction is a generic feature. The bag arm must be run there and labelled as a task-family prior, not as a peer of the learned merger. Neither variant is claimed to test H5 or H6; no learned merging exists yet.

Computation in `src/baseline_denominators.py`, output in `results/baseline_denominators.json`.

## Family redesign → analytic precondition check (no training)

The pure unordered family cannot test H4 at any scale. With the denominator
taken as the family-sufficient generic feature having the fewest states, the
honest compression ratio is exactly 1.00 for n=3,4,5: the bag of completed
events is family-sufficient and coincides with the minimal automaton. Fixing a
single order-sensitive pair for the whole family only relocates the target,
since 'bag plus which of those two came first' is writable in advance.

Two changes, both to task generation, neither to the learner:

Randomise the order-sensitive pair per instance. Any two permutations of the
same set differ on some pair, and some instance is sensitive to that pair, so no
fixed feature coarser than the required window is sufficient across the family
while each instance keeps a small automaton. Honest ratio becomes 1.69, 3.70,
9.72 for n=3,4,5 at d=0, m=2.

Add a shared bridge of length d before the branch diverges. Without it the
minimum sufficient window is n-1, because at the decision point the single
missing unordered label is inferable from the count, and any branch-specific
label observed afterwards gives the branch away at window width 1. With the
bridge the deciding evidence sits at an unpredictable position in the unordered
phase and at least d steps behind the decision. Measured rule over every cell
tried: minimum sufficient window width = n + d - 1. At n=4, d=6 the denominator
is 229 states against 257 for full history, so bounded windows retain almost no
advantage.

Both tails must differ at their first label. An earlier constructor applied the
no-adjacent-repeat fix in a way that made them identical, silently collapsing
those cells back to pure unordered and reporting a false ratio of 1.00. Fixed.

Proposed ladder, all with task length n+d+m <= 12 so the frozen sparse budget
from the branching calibration remains plausible:

  n=3 d=0 1.69 | n=3 d=2 2.00 | n=3 d=4 2.19 | n=4 d=0 3.70 | n=4 d=2 4.93
  n=4 d=4 5.84 | n=4 d=6 6.54 | n=5 d=0 9.72 | n=5 d=2 14.00 | n=5 d=4 17.61

Measurable range is bounded by what the denominator arm can actually learn: at
the top of the ladder it holds 898 task-feature states and will likely be
right-censored under the frozen budget. Those cells still yield the weaker
statement that the generic baseline fails where the automaton succeeds, which
is how Stage 7 already reported the count arm; they do not yield a ratio.

Computation in `src/family_design.py`, output in `results/family_design.json`.
No learned merging exists yet and no H4/H5/H6 result is claimed.

## Step 3 executed: does compression convert into learning gain

Yes, partly. Pooled over 26 solving arm/rung cells, steps to 90% = 4417 + 399 x
visited task-feature states, R2 = 0.92; the proportional model without an
intercept fits worse at 0.87. The intercept is 16% to 55% of the minimal
automaton arm's total cost. Under Stage 7's dense per-step reward the same fit
had an intercept of about 3.5%, so the representation-independent term is the
cost of finding the first success, which compression does not touch.

1152 runs: 6 rungs x 6 representations x 32 seeds, all tabular, no learned
merging, no shaping, no skills, no noise. Progress count never solved any rung;
the true minimal automaton solved 32/32 everywhere. History over automaton cost
ratios run 1.62 to 2.95 with paired bootstrap intervals excluding 1.

Three results that constrain how this may be reported.

Design-time compression ratio is a poor predictor of the cost ratio, r = 0.21.
The ratio computed from states actually visited in training predicts it well,
r = 0.81, because the design-time count uses only optimal-trajectory prefixes
while an exploring learner fills many more. All later axes use visited counts.

An insufficient representation is not necessarily a failing one. Wrong events
are ignored, so an aliased agent can sweep candidate labels and still finish at
a detour cost. The bag arm solved four rungs and stalled near 0.42 on two;
windows narrower than the aliasing-free width solved some rungs. The aliasing
check is a necessary condition for trouble, never a sufficient one, and which
under-sized representation gets away with it is an empirical question.

A more compact but insufficient representation can beat a correct one. Bag beat
the minimal automaton on four rungs. This follows directly from the affine law:
the detour costs less than tens of extra states.

Two prerequisites were forced by failures rather than chosen. Open rooms grow
as the square of their side; at side 13, 187 walkable cells, every arm
including the oracle failed to complete the task even once in 200k steps under
terminal-only reward. Corridor layouts give 3 to 8 distinct forced orderings in
25 to 31 cells at 25 to 27 step optima. And the branch consequence must sit
behind a shared bridge, or a window of width n-1 recovers the bit at the
decision point from the count, and any branch-specific label gives it away at
width 1.

Realised ratios reach only 2.6 against family-level potential of 5.8 at n=4
d=4. Geometric forcing through start positions cannot realise the combinatorial
compression; the gap is 2x to 6x and is a property of the forcing device.

Next: the learned merging arm, which is the first thing in this stage that is
an algorithm rather than a reference. Nothing here supports H5 or H6.

## Step 4 executed: learned state merging, negative

The merge rule is sound and the evidence is the binding constraint.

Rule: prefix tree plus evidence-driven merging that reads nothing from the task
machine (asserted by a source scan). Two nodes conflict when the same event has
been seen advancing the task at one and being ignored at the other. Comparing
observed advancing sets for equality, the first version, is far too weak: early
nodes have seen one or two advancing events and all look alike, precision 0.27.
Merging is greedy with successor propagation, recomputed whenever evidence
changes, so a thin merge is undone when a counterexample arrives. A revision
counts only when two nodes that shared a class stop sharing it; counting new
nodes as revisions was a bookkeeping bug, now fixed and tested.

Control on complete evidence: precision and recall 1.00 and classes equal to
true classes on all six rungs. At 90% of (state, event) pairs tried, precision
is already 0.13 to 0.69; at 80%, 0.09 to 0.33.

A trained agent reaches 0.66 to 0.75 structural coverage, and raising epsilon
from .02 to .6 leaves coverage flat while the policy degrades. The evidence
needed is negative evidence at states the agent has no task reason to revisit,
and random actions in a corridor map do not navigate to a distant label. This
is the coverage tail from the liveness pilot appearing a second time in this
project, in a different mechanism.

Cross-rung means over 32 held-out seeds: oracle automaton 13,365 steps with AUC
.947; full history 31,276 with .892; learned merging 32,042 with .373 and final
success .28; count censored. Learned merging recovers none of the step 3
compression benefit and is slightly worse than not merging.

Metric trap worth carrying: on four of six rungs the merged arm's first90 is
identical to full history, because before any node is evidence-bearing it is
the prefix tree. First90 alone reports a tie; AUC and final success show that a
later revision destroys the policy. Whenever structure can be revised, first90
must be read with AUC and final success.

Development selection over theta in {100,200,400} and revision in
{reset,transfer,replay}, 648 runs, picked theta 400 with transfer, which has
merge precision .14 and 53 classes against 23 true. The fastest configuration
is the one that merges least; at achievable coverage merging is a net loss.

Consequence for the plan: the deferred structural discovery bonus is no longer
an optional axis, it is the identified bottleneck. What it must buy is not more
exploration but directed trials of specific events for the purpose of telling
states apart, which is a different objective from exploring for reward. Nothing
here tests noisy feedback, confidence thresholds, shaping or skill reuse, and
the three revision policies were compared on development seeds only, so this is
not an H6 result.

## Step 5a and 5b executed: H4 closed, two larger mechanisms found

H4 via learned merging is No-Go against its preregistered criterion. The quota
rule fixes what step 4 diagnosed: requiring every event to have been tried at
both nodes turns "nothing contradicted this" into "nothing contradicted this
although we looked", and merge precision goes from .14 to 1.00. But recall
falls to .23-.39, so safe merging is nearly no merging. Against full history
with the identical directed exploration the paired cost ratio is 1.00 with
interval [.99, 1.00], tied on every rung, while AUC is .871 against .961 and
final success .89 against 1.00. Learned merging contributes exactly nothing and
costs stability.

The detours bought to feed the merger turned out to be a mechanism in their own
right, and a bigger one. Over six rungs and 32 held-out seeds, mean steps to
90%: full history 32,188; undirected detours 24,281; oracle compression 13,354;
directed structural trials 11,693; automaton shaping 10,677; shaping and trials
5,432; all three 3,766. Paired ratios against full history: trials 2.75
[2.62,2.90], shaping 3.01 [2.87,3.16], compression 2.41 [2.33,2.50]; and
compression still adds 1.44 [1.38,1.51] on top of the other two.

Targeting is the active ingredient, not detouring: random targets buy 1.33x,
least-tried targets 2.75x. What pays is knowing which events the current task
state has not yet tried, and that comes from recording history, not from
merging or from an automaton.

Shaping fails loudly when mis-scaled. At scale .4 both the learned and the
privileged potential go to 0/72 solved and zero final success: stopping at the
end of the known structure becomes a stable policy once the shaping term
outweighs the task reward. The frontier-stopping mode written into the plan is
real and the scale must be calibrated. The learned potential costs 10,677
against the privileged 5,589, and that factor of two is precisely the price of
not knowing how long the task is before solving it once.

A prediction from the previous round is refuted by these data. The intercept is
not the discovery cost that shaping removes. Fitting cost against visited
states: no aid 3,202 + 413 per state, directed trials 3,128 + 131, shaping and
trials 2,834 + 41. The aids flatten the slope; the floor barely moves. R2 falls
from .96 to .55 as aids are added, so cost-proportional-to-states is a property
of the unaided learner, not a general law.

Standing asymmetry this stage established: discovered edges are reliable
because they need only positive evidence, which doing the task supplies; state
equivalences are not, because they need negative evidence, which doing the task
never supplies. Two of the three mechanisms sit on the reliable side. The best
fully learnable combination, full history plus shaping plus directed trials at
5,432, already beats the oracle minimal automaton used alone at 13,354.

Not tested: feedback noise, confidence thresholds, skill reuse evaluated
separately, retroactive relabelling as a controlled arm. Shaping and
exploration hyperparameters were selected on development seeds and only the
selected values were confirmed.

## Query-selection follow-up: the step 5a attribution was wrong

Implementing an L*-style query rule, which picks the experiment that would
settle a doubtful merge in the current optimistic hypothesis rather than the
least-tried event, produced a paired ratio of 1.00 with interval [1.00, 1.00]
against the least-tried rule, at every exploration budget from .01 to .15. In
this task family the two rules pick the same event almost always.

The control it forced is the useful part. Decomposing the 2.7x attributed to
"directed structural trials" at beta .15 over 32 paired seeds: detouring at all
1.28 [1.23,1.34]; switching to labels not yet collected this episode, which
needs no structure whatever, a further 1.63 [1.55,1.71]; switching to the
least-tried event per task state, which needs the history-indexed tally, a
further 1.28 [1.22,1.33]; the L* rule 1.00. Absolute: no detours 31,250; random
targets 24,411; uncollected labels 14,979; least-tried 11,745.

So the structure-dependent part is 1.28x, not 2.75x, and the mechanism is
misnamed. It is not buying structural evidence: coverage sits at about .80
regardless of rule or budget, .81 at beta .01 after 7k detour steps and .78 at
beta .15 after 75k, and the rule that reaches the highest coverage, random
targets, is the worst learner. What the detours buy is goal-directed
exploration under a terminal-only reward.

The step 5 headline stands, three mechanisms composing 32,188 to 3,766, but the
middle mechanism must be described as goal-directed exploration of which only a
small part consults the task state. REPORT_STEP5.md section 5b records the
correction.

## Stage 8R: audit found three defects; two Step 5 conclusions reverse

Review alleged three implementation defects. All three were verified against the
source and the data before any repair.

Defect one, the merge conflict test read only the union-find representative's
evidence. Three-node counterexample: A has no evidence on event 0, B has seen it
advance, C has seen it ignored; merging A with B makes A the representative, and
the later test of A against C misses B's evidence and merges a directly
contradicting pair. Defect two, the quota was checked only on the candidate
pair while propagated successor pairs were united unconditionally. Defect three,
`deepest` was never assigned anywhere in the source, so the potential was zero
before the first success and the "learned" potential was a post-success
progress-count potential. A fourth item is mine: the reported merge precision of
1.00 came from a single-seed single-rung probe; the confirmation average is .699
with per-rung values .555, 1.000, 1.000, .310, .606, .725.

Repairs: class-level evidence aggregation, tentative merge with an undo log and
whole-closure rollback, quota on propagated children behind a flag, `deepest`
fixed, shaping split into named variants none/count/frontier/oracle at a common
scale chosen on development.

Gate 1, full reachable task language with complete evidence: every rule
including the old one recovers the exact minimal partition on all six rungs. The
inference machinery is not the problem; the defects do not bite when evidence is
complete, which is why the earlier control missed them. That earlier control was
also weaker than claimed, covering only prefixes of the selected optimal
trajectories.

Gate 2, evidence dumped from real training, policy removed, all rules on the
same data at coverage .54: old rule precision .24 recall .65; new without quota
.51/.57; new with quota 1 .88/.39; new with quota 1 and strict children
1.00/0.00. Strict children has recall exactly zero and was wrongly made the
default in the rewrite; corrected to loose propagation.

H4 retested, development selection theta 20 quota 1 loose, confirmation on seeds
800-831: full history 31,599; learned merging 17,932 with precision .93, recall
.44, 1.54x the true class count; oracle 13,318. Paired, learned merging beats
full history 1.76 [1.69,1.83] and sits 1.35 [1.30,1.39] from the oracle, closing
75% of the gap. **The Step 5 No-Go is retracted.** With shaping added the
merger's recall falls to .08 and its edge over history narrows to 1.12, because
learning finishes before enough evidence accumulates.

H5 retested at a common scale on seeds 700-731: none 32,073; count 11,641;
frontier 5,526; oracle 5,698. Restricted to steps before the first success,
count is 1.00 [1.00,1.00], exactly no effect, with the same first-success step
as no shaping; frontier 4.71 [4.39,5.05]; oracle 4.99. Frontier against oracle
is .97 [.94,1.00], so a potential computed only from discovered structure
matches the privileged one. The earlier claim that the learned potential costs
twice the oracle was a scale artifact, .1 against .05.

The L* query rule survives its null and is now marginally negative, .968
[.943,.993] against least-tried, re-run with the corrected hypothesis.
Unaffected by all of this: the value of compression itself, the sparse-reward
redesign, the affine cost law, coverage saturation under any epsilon, and the
1.63x layer for aiming at labels not yet collected.

## Step 6 executed on the recheck's recommendation: depth versus task graph

Built twelve tasks where progress depth and task-graph distance disagree, by
giving the two branch tails different lengths. At the end of the shared bridge
both branches stand on the same cell with the same progress count but owe a
different number of task transitions; the disagreement is 2 to 4 transitions,
verified exactly on the product, alongside the existing check that position plus
count cannot substitute for history. Physical route lengths 11 to 34 and task
lengths 6 to 13 are reported separately, which turned out to matter.

A scale confound was found and controlled first. Under discounting, PBRS is not
invariant to the potential's magnitude: a step that changes nothing carries
(gamma-1)*Phi, a per-step subsidy proportional to distance from acceptance, so
the potential with the larger range delivers a stronger signal at the same
nominal scale. At a common nominal scale the depth potential wins; normalised to
a common range the ranking flips. Settled by giving each potential its own
development-selected scale, and on twelve tasks all three select .07.

Result, twelve tasks by 32 held-out seeds, 384 runs per arm, all solved: none
28,990; depth 4,336; learned task graph 6,250; oracle task graph 5,904; product
distance 1,128. Block bootstrap over the twelve tasks: depth over none 6.69
[4.15,10.42]; task graph over depth .73 [.48,1.16], interval includes 1 so
unresolved and negative in direction; learned task graph is indistinguishable
from the oracle one, so this is not an inference failure. Per the recheck's
predeclared reading order that means the task shows no extra structural value
and the inference is not to blame.

The unplanned result is the third arm. A potential over the product of task
machine and map, that is environment steps still owed rather than task
transitions still owed, beats the depth potential 3.85 [2.79,5.21] and the task
graph 5.24 [3.47,7.46], on twelve of twelve tasks individually, cutting first
success from 5,209 steps to 269. This is the project's older thesis appearing as
a shaping number: the task graph alone is worth .73 against a naive progress
heuristic, the same graph multiplied with the map is worth 3.85.

One explanation ruled out: converged policies are all within 1.03 to 1.06 of the
shortest route, and the graph arms are marginally closer, so nothing is dragged
off course; the deficit is entirely post-first-success. One association kept:
the deficit grows with the depth/distance disagreement, 1.19, 1.42, 1.52 at gaps
2, 3, 4, slope .150, r .64 over twelve points, reported as association not law.

The product potential is privileged and marks headroom, not a method. The
learnable version counts backwards from successful episodes to estimate steps to
success per (task state, cell); that is the next arm. The task-graph comparison
is unresolved rather than refuted and would need more frozen tasks.

Recording discipline from the recheck adopted: every confirmation batch stores
source and binary SHA256, per-task file SHA256, all switches and scales, seed
range and budget, with strict_children written explicitly rather than defaulted.
Regression checked after appending the product table to the task files: existing
arms reproduce their trace hashes exactly.

## Step 7: reviewer pass on Step 6, then the learnable versions

The Step 6 headline needed two things removed before it could stand. The
product distance is monotonically equivalent to the optimal value function under
this step cost and discount, so shaping with it is close to handing over V*; and
the task-graph arm counted transitions while the product counted steps, a unit
confound.

Decomposed over twelve tasks and 32 held-out seeds, each potential at its own
development scale. Position alone .98 [.93,1.06], nothing. The same numbers
attached to shuffled task states .66 [.60,.72], actively harmful, so this is not
"any structured potential helps". Progress depth 6.76 [5.28,8.35]. Task state
converted to step units .74 [.67,.81], so the unit confound is not the
explanation and a task-only potential remains worse than the naive depth
heuristic in either unit. Adding position while judging the branch wrongly 2.03
[1.69,2.50]. Judging the branch correctly a further **2.55 [1.95,3.15]**.

That last layer is the defensible claim: the part of the structure that only a
task memory can represent, which branch you are on as decided by an event many
steps back, is worth 2.55x once you already have the map half. It also answers
the V* objection, since the wrong-task product table is equally a V* for some
task and reaches only 2,862 against 1,122.

Two learnable versions of the product potential were then built and both are
worse than the naive depth potential. Backward counting from successful episodes
reaches 11,062 with first success 1,099; computing backward reachability over a
learned map graph crossed with the discovered prefix tree reaches 10,438 with
first success 2,192; depth is 4,281 and 1,144, the oracle product 1,125 and 276.
The reasons are complementary and both diagnosable. Backward counting has no
data before the first success, so in exactly the phase where the product
potential earns its advantage it simply is the depth potential. Frontier-targeted
reachability sets the potential to zero at the deepest node discovered, making
that an attractor, and first success degrades. The frontier-stopping failure
written into the original plan appears here in a second form.

An implementation error found and fixed on the way: the learned table is in
environment steps while its fallback counted events, so the potential jumped at
the edge of what had been learned. Converted using the agent's own measured
steps per progress event.

Conclusion for the shaping line: the product potential's advantage lies almost
entirely in knowing the distance to a goal never yet reached, and that quantity
cannot be constructed from discovered structure. This bounds structure-derived
shaping independently of inference quality. Two designs were tried, so this is
"these two cannot", not "nothing can".

## Step 8: does the learned automaton survive a change of map

Back to the original vision's unspoken promise, that the automaton once drawn is
worth keeping. Every experiment until now learned one automaton for one map and
discarded it, which forces the structure to repay its own discovery cost inside a
single run.

What travels is the tree: which event advanced the task after which history,
which histories ended it, and how often each event was tried there. No cell
appears in that statement. What does not travel is every cell-indexed table, the
learned map graph and the product distances, all of which restart at zero.

Six map pairs, each pair carrying an identical task machine on two different
corridor layouts, each layout passing the existing checks independently. 768 runs
over 32 seeds. On the new map: full history 27,688; learn the structure there
from scratch 18,245; carry the structure over 16,417; true automaton 12,823.
Block bootstrap over the six pairs: learning structure locally beats full history
1.52 [1.28,1.75]; carrying it over beats learning it locally 1.11 [1.05,1.17];
the carried structure sits 1.28 [1.12,1.55] from the true one.

The carried structure is a head start, not a shortcut. Share of starts solved at
10k steps: 0.293 learning locally against 0.440 carried, the largest gap; by 50k
the two are level. It is worth only 1.11x overall because in this setting the
structure is cheap to relearn, which is the same exact-feedback effect that has
shown up in every stage.

Amortisation, sweeping the source budget: the structure saturates between 10k and
20k source steps at 57 nodes and about 1,600 to 1,800 target steps saved, and
training the source longer buys nothing more. A structure from only 2,000 source
steps is worse than none, costing the target 453 extra steps. Cheapest purchase
is 10k source steps, paying back over about 6 new maps against an agent that
relearns structure on each map, or on the first new map against an agent that
uses no structure at all.

Two implementation errors on the way, both of a kind this project keeps hitting.
A hand-rolled JSON parser anchored on the wrong field and read past the end,
then failed on bracket characters; replaced with a plain table written by Python.
And seeded nodes were given true class zero, so the transfer arm's merge
precision and recall were meaningless until the true class was recomputed by
replaying each route through the task machine.

Limits: six pairs is few blocks; both layouts come from the same corridor
generator; the task machine and event vocabulary are identical across the pair,
so this measures a change of map, not a change of task. Next along this line
would be irreversibility, which the project's own dose-response says is where
the product pays, and which this whole line has never had.

## Step 9: hardening the transfer experiment, and half of Step 8 comes back

Acted on three requirements from the transfer literature note: make "the target
map must exercise orderings the source map rarely produced" a controlled
variable, add the arm that carries the same source evidence without merging, and
check whether the target map refutes merges the source licensed.

Ordering overlap is the share of the target's forced orderings that the source
also produced. Enumerating every ordered pair over two tasks and binning:
high band 10 pairs at overlap 1.00, mid 10 at .42, low 5 at .00, no map reused
within a band. Steps to 90% on the target, cold-merged against transfer-merged:
18,375/18,088, 19,588/16,300, 29,862/28,875. Block bootstrap by pair gives 1.02
[.82,1.24], 1.20 [.99,1.47], 1.03 [.94,1.18]; **every interval contains 1**.
Pooled over all 25 pairs, 1.08 [.95,1.24].

Step 8 reported 1.11 [1.05,1.17] from 6 pairs and 32 seeds. The point estimate
barely moved; the interval did. Blocks are map pairs, not seeds, and between-map
variance dwarfs between-seed variance, which six blocks cannot measure. Per pair,
19 of 25 favour transfer, median 1.09, range .64 to 1.87; sign test one-sided
p=.0073, but 14 of 28 maps appear in more than one pair, up to five times, so the
pairs are not independent and that p is optimistic. Honest statement: direction
consistent, magnitude small, not resolved against between-map variance.

The no-merge transfer arm turned out to be a bitwise identity with the plain
full-history arm, trace hashes equal. With merging off the feature is the
individual history and the Q table restarts at zero, so having seen a node
before changes no number. That is the separation the note asked for and the
answer is clean: what transfers is the equivalences, not the events.

Source bias is real and measurable. Revisions rise from .35 and .58 at high and
mid overlap to 1.60 at zero overlap, four to five times, so novel orderings do
refute source-licensed merges. Final precision is level with the cold arm at .89
against .90, so revision absorbs it.

Unaffected: learning structure locally beats full history 1.22 [1.05,1.39]; the
true automaton beats it 1.80 [1.64,1.95]; the carried structure sits 1.36
[1.17,1.62] from the true one. Structure is worth having; carrying it across
maps is what is not established.

Consequence: Step 8's amortisation conclusion is withdrawn, since a payback
period needs a resolved saving. The acquisition figure stands, because it
measures when the structure saturates, not what transfer is worth.

## Step 10: irreversibility splits correct structure from learned structure

Executed the literature note's second stage. A commit action is added at the one
place where the two histories share a cell and a progress count; a wrong commit
costs nothing, ends the episode, or wastes the rest of it, three doses.

Two design defects, both hit before being fixed. First, with the suffix labels
strung along one corridor, reaching one branch's first label means walking over
the other's, so under fatal rules a whole branch became impossible and the
evaluator silently dropped 10 of 25 map pairs. Fixed with a forked exit, one
commit label a single step off the fork in each direction; both versions share
the same forked map and the equality of their optimal routes, starts and lengths
is asserted rather than assumed. Second, entering the failure sink is a change of
task state, so the progress test reported it as progress and the learner recorded
the fatal label as advancing. That made the two branch states look alike, licensed
the wrong merge and left nothing that could refute it. With the bug the headline
read "learned structure becomes a 2.85x liability under irreversibility", which
was entirely artefact; corrected it is 1.12 [1.05,1.17].

15 maps by 16 seeds by three doses by three arms, 2,160 runs. Steps to 90% for
history, learned merging, true automaton: ignored 19,506 / 16,013 / 10,788;
terminating 20,863 / 18,214 / 12,358; wasting 25,271 / 24,786 / 12,644, with
24,910 doomed steps, 12.5% of budget, and 2,227 failures.

Paired bootstrap over the 15 maps. Learned structure against full history falls
1.20 [1.05,1.36], 1.15 [1.02,1.32], 1.01 [0.90,1.12] as the penalty rises; the
true automaton rises 1.71 [1.51,1.99], 1.63 [1.47,1.81], 1.84 [1.60,2.18]. Cost
of the heaviest dose per arm: history 1.30 [1.22,1.36], oracle 1.21 [1.12,1.27],
learned merging 1.53 [1.27,1.80], hurt the most.

This is the project's oldest dose-response applied to the learning side and it
forks: irreversibility raises the value of correct structure and removes the
value of imperfect learned structure, at once. The learned arm used the
hyperparameters frozen in Step 8 with no retuning for the irreversible
condition, and a more conservative merge threshold is the obvious next thing to
try, which is exactly the confidence-aware commitment question the first round
of this project posed and never got to test.

A third analysis bug worth recording: an ad-hoc bootstrap resampled blocks
independently in numerator and denominator, breaking the pairing and inflating
every interval. Corrected by drawing one block sample per replicate.

## Step 11: confidence-aware commitment, the first round's H3, answered

H3 asked when a discovered equivalence may be trusted enough to act on. A
reversible environment could never answer it, because trusting a wrong one only
costs a detour. Step 10's irreversible version makes the question askable.

The ceiling first. Splitting the unweighted runs by final merge precision, a run
with a perfect structure beats one without by 1.09, 1.13 and 1.16 across the
three doses. **That is the ceiling for any confidence mechanism**, and it is the
same order as what merging is worth at all, 1.20.

Two mechanisms tried. A global evidence threshold from 1 to 16 buys precision
with recall everywhere: at 16 the precision is ~1.00 and the recall ~0.00, which
is the full-history arm again. The best threshold is 1 or 2 in all three doses,
so the optimum does not move with the cost of being wrong.

Targeting failures does nothing in its obvious form, and the diagnosis is the
useful part: **the node that pays is not the node that erred.** In every run with
wrong merges, every wrongly merged node had zero failures of its own; failures
land on deep commit nodes, one with 1,302 of them, while the harmful merges are
shallow, in the unordered phase, on nodes that never failed. Propagating blame
back along the failing episode's own path fixes precision, from .50/.00/.00 to
1.00 on the runs selected for having wrong merges.

It still does not pay. 2,880 runs, 15 maps by 16 seeds by three doses by four
risk settings. Precision rises .78 to .93 under the heaviest dose but recall
falls .04 to .01, and against risk zero the paired ratios are 1.00 exactly under
"ignored", a bitwise identity and a built-in null control; 0.84 [0.73,0.96] at
risk 4 under "terminates", significantly worse; and 1.07 [0.92,1.27] under
"wastes", unresolved.

So in this setting always-commit beats confidence-aware commitment, and the
reason is arithmetic rather than mechanism: the loss from wrong merges is capped
at 1.16, recall is already only .04 so there is no margin to spend, and a merge
foregone costs more than a wrong merge prevented. That also explains why the
first round found no evidence for confidence-aware reuse: the effect does not
exist in a reversible environment and is too small in an irreversible one.

Blame was distributed flatly over every node of the failing episode, the crudest
possible assignment; discounting by recency or by when the merge was made might
do better, and the merge hyperparameters were never retuned for irreversibility.

## Step 12: the floor that compression could not touch does not exist

Step 3 fitted steps-to-90% against state count with a straight line and read the
intercept, 4,417, as a representation-independent floor; Step 7 built on that
with "the aids flatten the slope and the floor barely moves". Both are withdrawn.

Ruled out first: the intercept does not move with the discount factor, 6,166 /
5,990 / 6,035 / 5,437 over gamma .95 to .995, so it is not a value-propagation
rate.

A fitting error also explains the erratic per-task R2 seen along the way. The bag
and narrow-window arms are *insufficient* on these tasks, so they solve a harder
problem and do not lie on the same curve; restricting to sufficient
representations makes the fits clean.

Model comparison on sufficient representations only. Step 3's own ladder, 18
points: affine 3,128 + 413 x states R2 .934; a power law with **no intercept**
679 x states^0.91 R2 .936, equally good; with route length 3.29 x length^1.86 x
states^0.79 R2 .976. Depth-versus-graph tasks, 24 points: .807 / .784 / .953.
Pooled 36 points spanning route lengths 13 to 29: **25.7 x length^1.20 x
states^0.79, R2 .962**.

So the intercept was a straight line fitted to a concave curve. The real
regularity is a state exponent of about 0.8: halving the state count buys 0.58,
not 0.50. Compression carries diminishing returns of its own, with no floor
needed to explain them.

Step 7 restated correctly: the aids lower the exponent rather than leaving a
floor. No aid 736 x states^0.89; directed trials 898 x states^0.61; shaping and
trials 1,071 x states^0.39. At 0.39 halving the states is worth 0.76, which is
why compression retained only 1.44x on top of the other two mechanisms.

Consequence for the compression claim: "cost is proportional to state count"
weakens to "cost grows as the 0.8 power of state count", and the gains reported
in Step 3 were slightly optimistic.

Limits: the exponent is fitted over two task families, 36 points, route lengths
13 to 29, with no derivation and no validation outside that range. The route
length exponent is 1.13 in one family and 1.86 in the other, so that term is not
well identified and may be entangled with the horizon setting. Only
steps-to-threshold was modelled; AUC was not.

## Step 13: third attempt at a learnable product potential reaches parity, no more

The first two failed for complementary reasons: backward counting has no data
before the first success and so *is* the depth potential there; frontier-targeted
reachability makes the frontier an attractor. The depth potential is zero at the
frontier too and is fine, the difference being that it has no spatial component
and so creates no attractor in space.

Third design splits the potential. The spatial half points at the nearest cell
bearing an event this task state has not tried, over the learned map graph, so
its target moves as evidence arrives and cannot become an attractor. The depth
half is events still owed, with an optimistic deepest-plus-one horizon before any
success. Both halves use only what the agent observes.

Two implementation errors, either of which reverses the conclusion. Converting
the depth half into steps needed an average steps-per-event, which is wildly
unstable early: a few hundred steps before the first event makes it enormous and
the potential explodes, and every configuration collapsed. Removed by leaving the
depth half in events and the spatial half in steps. And sharing one scale between
halves of different range held the depth half at a fifth of its own optimum.
Before the fixes the best guided result was 9,094; after, 4,273. All of that
difference is implementation.

Confirmation, 12 tasks by 32 held-out seeds, 384 runs per arm: none 28,872;
depth 4,331; guided 4,273; backward counting 10,911; privileged product 1,143.
Paired by task: guided against depth is 1.01 [0.99,1.04] overall but **1.19
[1.15,1.23] before the first success**; backward counting 0.40; the privileged
product 3.79 [3.15,4.37] over depth and 3.74 [3.10,4.31] over guided.

So the spatial half works and its effect is resolved, and it is given back during
consolidation, 3,218 to 3,340, even though it is switched off after the first
success. The likely mechanism is that arriving sooner means arriving with a
thinner value table; that is an explanation, not a measurement.

Three attempts now stop at parity and none closes the 3.74x gap. The diagnosis
from Step 7 has survived all three: the product potential's advantage lies almost
entirely in knowing the distance to a goal never yet reached, and that cannot be
built from discovered structure. Recommend closing this line; there is no new
mechanism hypothesis left in it.

## Step 14: making full history infeasible, and a terminology correction

The third item asked whether the claim could be pushed from "structure is
cheaper" to "structure is necessary". Checking the baseline first turned out to
matter more than the plan.

What every report from Stage 7 onward called "full history" keeps only the events
that produced progress. Stage 7 declared this and justified it as sufficient
under the ignore-wrong-events semantics, which is true, but it embeds a
task-family prior: non-progress events may be discarded. An agent that knows
nothing does not know which events matter. The baseline with no prior records
every observed event.

12 tasks by 16 seeds, 192 runs per arm, feature cap 4,096. Raw history: 14,545
distinct states, 10,449 overflows, **0/192 solved**, AUC .039. Progress-event
history: 75 states, 28,714 steps, 192/192. True minimal automaton: 24 states,
12,901 steps. Raw over progress is 195x in state count; progress over automaton
3.1x.

So remembering everything really is infeasible rather than merely expensive. But
the step that makes it feasible is not the automaton: it is a one-line generic
rule, discard events that did not advance the task, which needs only the progress
feedback every arm already has. The automaton is a further 3.1x compression worth
2.2x in speed on top of that.

Unplanned observation: a good potential partly rescues a broken representation.
Raw history with depth shaping solves 124/192 at 85,411 steps, against 0/192
without. Consistent with shaping being the largest mechanism in this project.

Terminology correction for the whole project: "full history", "remember
everything" should read "remember everything that mattered". No reported number
changes, since every comparison used the same baseline, but their meaning does:
the automaton's 1.7 to 2.2x is measured on top of a baseline that has already
performed one abstraction, not on top of no knowledge at all.

Answer to the third item: **abstraction is necessary, the automaton is not.**
Deflationary, and in the same direction as every other step: each claim that
structure helps gets partly eaten by a cheaper substitute. This time what got
eaten was necessity; what remains is economy.

## Step 15 — the automaton as a goal provider, not as a table index

Prompted by a question that exposed a gap running through all of Stage 8: every
step from 7 to 14 measured one use of the automaton, as a dimension of the value
table, Q(cell, task state, action), and compared arms on how small that dimension
was. The other use was never measured here: the automaton names the next event
and the table becomes Q(cell, goal, action), whose task dimension is the event
vocabulary rather than the number of task states.

Half the framing in the question is right and half is not. The value table really
does not need the task state: the goal arm holds no task-indexed Q at all, task
dimension 6.4 against the minimal automaton's 24.2. But naming the goal still
requires tracking the task state; that tracking is a constant-memory fold over the
event stream, not a dimension of the table. History is data, not state.

Headline, 12 tasks by 16 seeds, 192 runs per arm, Bellman updates matched to
within 4%, evaluated every 100 steps. Steps to 90%: progress-event history
27,253; true automaton as index 12,604; automaton as index plus depth shaping
1,880; history plus product shaping 701 (the strongest arm of the previous
fourteen steps, privileged with a V*-equivalent distance table); learned
automaton as goal 838; true automaton as goal 739. Paired block bootstrap over
the twelve tasks: goal over index **17.05 [13.64, 20.71]**, 12 of 12 tasks,
range 9.1x to 28.3x. Learned goal over true goal 0.88 [0.83, 0.93]. Goal over
history-plus-product 0.95 [0.83, 1.07], parity with the privileged arm.

Three things worth separating out. **First, for the first time in this project a
cheap substitute did not eat the claim.** The control that names a goal without
consulting the automaton, pick any event this episode has not collected, solves
**0/192**. Ordering knowledge has no substitute here. **Second, compression buys
nothing in this reading**: the learned tree has 52.8 nodes against the minimal
automaton's 24 and performs the same, because the table dimension is the event
vocabulary either way. **Third, the mechanism is data sharing, not table size.**
Under the cost law of Step 11, 24 states to 6 is worth about 3x; the measurement
is 17x. The goal reading factors one task into |events| goal-reaching problems
and every physical step is evidence for all of them at once, which the index
reading cannot do because a transition belongs to one task state.

The price is myopia, and it is real: converged route length 20.5 for the index
arm, 21.2 for the true goal arm, 22.8 for the learned one. Three to eleven per
cent worse final policy for 17x the learning speed.

Under irreversibility, 15 maps by 12 seeds at the heaviest dose, the project's
oldest fork reappears one level up. True automaton as goal 944 steps against
11,428 as index, **12.10 [11.02, 13.07]**. Learned automaton as goal 9,858, which
is **0.10 [0.08, 0.12]** of the true one, with fatal commits up from 1,904 to
4,700 and wasted steps from 22,800 to 53,707. A goal is a commitment; a
commitment made from a half-built automaton is a bet, and the goal arm holds no
task-indexed value function that could learn to hedge. Both goal arms still solve
180/180 against 63 to 80 for the index arms: what collapses is speed, not
solvability.

Two implementation errors. Mine first: the sub-goal reward carried the task's
terminal bonus, so one goal row learned which instance of an event wins and the
policy pinned itself against a wall, 0/192. Removing the terminal term and
driving the arm from the goal-conditioned skills the code already trains fixed
it. The second is a real bug with wider reach: **the prefix tree keyed its nodes
with the value encoder.** For the history and merged arms those two keys are the
same expression, so fourteen steps never exposed it. Under a goal index the
encoder hashes the required next event, so two different histories owing the same
next event became one node, the tree collapsed to one node per goal, successor
edges pointed at themselves, and the learned goal provider could never leave its
starting node. Fixed by giving the tree its own history key; the merged arm's
`trace_hash` is bitwise identical after the change, so it is an identity
transformation on every previously reported number. Two regression tests added,
28 total, all passing.

What this does to the stage. No reported number is withdrawn. What changes is the
qualifier on all of them: the cost law, the 1.7 to 2.2x for the minimal
automaton, the 1.22 to 1.76x for learned merging, the 1.08 transfer result and
the 1.16 ceiling on confidence are **results about the automaton used as a
state**. The same automaton read as a goal provider is worth 17x on the same
tasks. And this is the first positive foothold for the project's original vision:
building the automaton online and using it to say what to do next already reaches
88% of the true automaton in the reversible setting. What it lacks is not
inference accuracy but doubt about its own unbuilt part under irreversibility,
which is the mechanism Step 11 tested and rejected, now asked again in a reading
where it finally has room to matter.

## Step 16 — the prior-art comparison the plan made mandatory

`docs/PLAN.md:326` states that before writing, the contribution must be compared
face to face with JIRP's equivalent-state transfer and with reward machines'
counterfactual experience, and that novelty may not be claimed until then.
`REPORT_STEP9.md:73` records that neither was run. Four comparisons, all now run.

**Counterfactual experience (CRM).** Stage 7 to 15's index arms used one online
update plus five plain replays; the prior art has had counterfactual relabelling
across task states since 2018, and round one of this project even implemented the
kernel. 12 tasks by 16 seeds. Steps to 90%: index plain 12,604; index with the
replay budget raised to CRM's update count 6,333; index with CRM 5,012. So CRM
beats the standing baseline **2.51 [2.09, 2.91]**, but against matched updates it
is only **1.26 [1.02, 1.52]**, 9 of 12 tasks: most of the gain was compute, not
counterfactuals. **Every Stage 8 comparison therefore used a denominator weaker
than the prior art.** And on a machine being inferred online, CRM is *negative*:
0.83 [0.76, 0.95], 160/192 solved against 182/192, with merge precision
identical (0.962 both ways). The degradation is not worse inference; it is the
same imperfect structure driven into the value function at 6.6 times the weight.
That bounds JIRP-plus-QRM style methods while the structure is still moving.

**Step 15's mechanism claim is refuted by its own control.** Step 15 attributed
its 17x to data sharing. But the goal arm with relabelling switched *off*, two
updates a step, still beats CRM at thirty-one updates a step by **3.03 [2.83,
3.30]**. The correct decomposition: the reading itself is worth 3.03, relabelling
a further **2.24 [1.96, 2.52]**, together **6.78 [6.19, 7.42]** over the prior
art's best arm. The 17.05 figure is not withdrawn but its denominator is a
baseline without counterfactual updates; the number to quote outside is 6.78.

**Passive inference: the merge rule is not a contribution.** AALpy is not
installed here, so RPNI and blue-fringe EDSM were written on top of this
project's own `Classes`, sharing the consistency test, the merge closure and the
rollback, so only the search differs. On complete evidence all four rules return
**exactly the minimal partition** on all six rungs. On real dumped training
evidence they sit at different operating points, not better ones: project loose
precision .882 recall .392, RPNI .473/.830, EDSM .423/.821, RPNI and EDSM
producing *fewer* classes than the truth. Priced in the loop: reversible family
RPNI over sweep 1.14 [0.68, 1.81], unresolved; irreversible family **0.70 [0.56,
0.87]**, significantly worse, and worse than not merging at all. Consistent with
this project's oldest finding, since aggressive merging buys recall with exactly
the errors irreversibility charges for. EDSM's scoring is indistinguishable from
RPNI everywhere. **Note also that steps-to-90% misleads for these arms**: RPNI
reaches 90% earlier than sweep and then falls back, AUC .591 against .844, 76/144
solved against 136/144.

**A qualifier this batch forces on an older number.** Learned merging over
progress-event history is 1.07 [0.91, 1.27] here, interval containing 1. The
recorded 1.22 to 1.76x was measured on the `family_maps` rungs at 200,000 steps;
on the gap family at 100,000 it does not resolve. Not a contradiction, but that
number is **family-dependent**.

**JIRP's equivalent-state transfer.** The reviewer's point that `--revision
transfer` is not JIRP's rule is correct, and `--revision jirp` now implements
future-equivalence transfer as three rounds of signature refinement over a
class's observable behaviour and its successors, transferring only on an
unambiguous match. 12 tasks by 12 seeds, steps to 90%: reset 74,939; raw-buffer
replay 72,955; JIRP 43,601; the project's member copy 25,906. JIRP over reset
**1.72 [1.43, 2.23]**, so the principle works; the project's cruder rule over
JIRP **1.68 [1.42, 2.02]**, because unambiguous future-equivalence is rare in a
prefix-tree-shaped hypothesis, so JIRP transfers far less.

**Net.** Three claims lose ground: the merge rule is not a contribution, Stage
8's baseline was weak, and Step 15's mechanism was misattributed. One survives a
head-to-head for the first time in this project: the goal reading is still
**6.78x** faster than the strongest prior-art arm, with a table four times
smaller and 2.4 times fewer updates.

## Step 17 — fixing the evidence boundary, and the irreversibility fork was an artefact

First step of the agreed route: repair the experimental foundation and write the
information boundary down. An external review named five implementation defects,
three with independent witnesses. All were reproduced here against the unmodified
program before anything was changed; the headline witness matched to the digit:
65 real entries into the failure sink, 1,797 steps taken after it, `failures`
reporting 1,862 = 65 + 1,797, `doomed_steps` 26,525, and **1,766 events written
into the healthy history that preceded the failure, 738 of them directly
contradicting a progress record already stored at that same node.**

The cause is that the failure sink is absorbing, so every remaining step of the
episode satisfied the death test: it counted another failure, charged the whole
remaining horizon again, and recorded the current event as "ignored at the
history the episode had reached before it died".

**What was fixed.** (A) A transition taken after the task has already failed
writes no task-side evidence at all and updates a dedicated failure-state row, so
per-step update counts stay equal across arms; failures counts entries and
doomed_steps counts post-failure steps once each. (B) **Fatal is a third verdict,
not a kind of ignored** — the tree and the partition carry it, and advanced /
ignored / fatal are pairwise exclusive in the conflict test. Folding them let a
node that had seen an event kill the task merge with one that had seen the same
event do nothing, with no evidence able to refute it afterwards. (C) `relabel=0`
now replays the goal and node recorded at collection time instead of recomputing
them from current skill values with a null node. (D) Counterfactual updates on a
learned machine read **class-level** evidence rather than one representative
node, and the transition the agent actually took always uses the real outcome, so
enabling the rule can only add updates. (E) JIRP transfer compares against a
**snapshotted** old hypothesis and refines the signature to a fixed point rather
than three rounds; it reports `jirp_transfers`. (F) One bootstrap deduplicated
resampled blocks; the other five were already correct.

One defect was introduced by the fix and caught by the re-run: the learned goal
router's optimistic fallback treated "not seen ignored" as safe, which after
separating fatal began proposing lethal goals. A third verdict must be consumed
everywhere it is produced.

**The reversible family is bitwise unchanged** for history, automaton, merged and
goal, and the mechanism decomposition holds: reading 3.09 [2.89, 3.31],
relabelling 2.19 [1.98, 2.40], together 6.78 over CRM.

**The irreversibility fork is retracted.** Re-running the three doses, 15 maps by
16 seeds, learned merging over progress-event history is 1.21 / 1.15 / **1.19
[1.08, 1.33]**, against the recorded 1.20 / 1.15 / **1.01 [0.90, 1.12]**. The
first two doses reproduce almost exactly and only the heaviest changes — which is
the only dose that produces post-failure steps at all. Learned structure pays
about the same under every dose. "As the penalty rises the learned automaton's
value falls to nothing" was contamination.

**Step 15's irreversibility section is retracted too.** Learned goal router over
true goal router is **0.75 [0.67, 0.83]**, not 0.10; over merged indexing
**12.25 [9.51, 15.76]**, not 2.01; and the learned router now wastes *fewer*
steps than any index arm (1,774 against 2,085-2,282) with comparable fatal
entries. A narrower claim survives: across the three doses the learned router
sits at 0.91 / **0.54 [0.34, 0.79]** / 0.75 of the true one, so its deficit is
largest where a wrong commit **ends** the episode, not where it wastes it. The
three doses are three rules, not a monotone scale.

**Two Step 16 results are retracted.** Counterfactual updates on a learned
machine are **1.30 [0.99, 1.58]**, not 0.83 — the degradation was starved
evidence and a dropped real update, so "the same wrong structure hurts more
because it is used more" is withdrawn. And blue-fringe learners under
irreversibility are 0.86 [0.67, 1.13], not 0.70 [0.56, 0.87]; with clean
semantics every arm solves 120/120 where before only 39-49 did, and learned
merging beats no merging **1.15 [1.05, 1.28]** there. JIRP's numbers all move:
over reset 1.08 [1.04, 1.14] rather than 1.72, and the member copy over JIRP 2.68
[2.12, 3.51] rather than 1.68. Direction unchanged, magnitude very different, and
the implementation now deserves the name much better.

**The information boundary, written down as the step's completion condition.**
The learner observes progress, ignored, **fatal** and success. History is made of
progress events only; the failure state is a state of its own and post-failure
steps go to its own row. A replayed row is reinterpreted under the current
encoder but its goal and node are recorded, not recomputed. Navigation skills
keep training after a failure because navigation is task-independent, which is
also what keeps per-step update counts equal across arms. When comparing with the
literature this feedback must be stated: it is not "structure discovered from
sparse reward alone".

**Naming, half of defect E.** `--goal-select learned` is a history-conditioned
**prefix router**: it reads which events have advanced at the current history
node and does not consult the merged global automaton. It will be called that
from here on. Its fair control, a same-evidence history dictionary, belongs to
step three of the route and is not built yet; `untried` is not that control.

**One older result now in doubt and not yet re-measured.** The 1.16 ceiling on
confidence-aware commitment was computed from `quota_sweep.jsonl` and
`blame_sweep.jsonl`, both produced by the contaminated program under the heaviest
dose. Its diagnosis (blame lands deep, harmful merges are shallow) is probably
robust, but the ceiling needs recomputing before it is quoted again.

## Step 18 — does the structure generalise off its own map? Yes, narrowly; coverage binds

Second step of the route: freeze a batch of source-map data and check offline,
with no new algorithm and no policy involved, whether structure learned there
predicts correctly on histories the *target* map can produce and the source batch
never saw. 25 map pairs graded by how much their forced orderings overlap, three
source budgets, two collection policies, 1.63M questions. From one batch, two
artefacts: the history dictionary (prefix tree keyed by the exact history) and
the candidate machine (same tree quotiented by the merge rule), plus the same
tree quotiented by the **true** task state as a ceiling on what that batch can
support. The true machine only marks answers. Three outcomes kept apart: unknown,
correct, wrong — saying "unknown" is honest and can be fallen back on, being
confidently wrong is what costs a decision.

**The dictionary answers 0.000 of the novel questions**, at every budget and both
policies. True by construction, but it confirms the evaluation set is genuinely
novel and that carrying evidence alone says nothing on a new map.

**The machine does generalise and is nearly always right when it speaks.** At the
ample budget it answers **0.096 [0.070, 0.125]** of novel questions with a wrong
rate of 0.0027 [0.0013, 0.0048], so **precision among answers is 0.972 [0.947,
0.988]**. That meets this step's continue condition.

**But the ceiling is 0.192 [0.147, 0.246] and the rule reaches half of it.**
Eighty per cent of novel questions are unanswerable from that batch however it is
merged. The limit is the data, not the compression. The same gap shows on
histories the source *did* visit: 41% of (history, event) questions there were
never tried at all; merging closes 41.4% to 36.5%, a perfect merge to 31.8%.

**Directed collection raises the ceiling and the rule converts less of it.**
Spending 15% of the same budget walking to untried events multiplies the ceiling
by **4.08 [3.57, 4.62]** at 2k, 2.11 [1.80, 2.46] at 10k, 1.23 [1.07, 1.43] at
40k, and lifts dictionary coverage from .586 to .681. Yet the machine's answer
rate at 40k *falls*, .096 to .067, because conversion drops from **0.50 [0.36,
0.65] to 0.28 [0.17, 0.43]**: richer collection creates more nodes with partial
evidence and more genuine contradictions, so the rule correctly becomes more
conservative. This reproduces "structural coverage blocks merging" offline, with
no policy confound. **The bottleneck moves from data to inference**: under plain
collection the data is worth improving, under directed collection the rule is.

**Environment bias is measurable.** At 40k, plain collection, by ordering
overlap: ceiling .249 at overlap 1.0, .194 at .33, .121 at 0.0 (the .5 band has
too few questions to read). The less the source map's forced orderings resemble
the target's, the less the batch can say. That is TALearner's concern, with a
magnitude attached, under this project's explicit progress feedback.

**Three consequences for step three.** The selective-use rule it proposes is
well-founded, since the structure is 97% precise where it speaks. But it can only
act on ~10% of novel decisions, so the proposed 2 tasks by 3 target maps by 5
seeds is very likely underpowered for that effect and should be power-checked
first. And the larger lever is not the use policy but conversion: taking the rule
from half the ceiling to all of it doubles the answer rate, which is worth more
than choosing well among 9.6%.

## Step 19 — conversion: the lever is what the rule will say about an event it never saw

Step 18 left the merge rule converting 0.28 to 0.50 of what its own batch could
support, and found that collecting better made conversion worse. Conversion was
therefore attacked directly, and entirely offline: collect one batch per
condition with merging blocked during collection, then apply every inference rule
to that identical batch. Nothing about the data depends on which rule is scored.

Two axes, because they are different kinds of claim. The **partition** says which
histories are the same state. The **verdict policy** says whether the rule will
speak about an event it never saw in that class: open-world answers only from
observed evidence, closed-world completes the hypothesis and asserts that an
event never seen to advance anywhere in the class does not advance. That is what
a task automaton claims, and it is strictly stronger than the evidence, so it is
scored apart. "Unknown" is split into its two causes, which call for different
fixes: the history cannot be folded into any class, or it can but the class is
silent about that event.

**The lever is the verdict policy and it is nearly free.** Same data, same
partition: closed-world takes correct answers on novel histories from 0.073 to
0.192, **2.62 [2.47, 2.77]**, and precision *rises*, 0.963 to 0.979, because
almost everything it adds is "does not advance" and almost every unseen event
genuinely does not. The rule's caution about unseen events was costing more than
half its answers and buying nothing.

**Two candidate levers died cheaply.** The visit threshold does nothing at all:
theta 1 over theta 20 is **1.01 [1.00, 1.03]**. Dropping the evidence quota buys
answers at a real price: 0.073 to 0.102 correct but 0.003 to 0.031 wrong,
precision 0.963 to 0.768. Blue-fringe merging is a genuine lever, 2.74 [2.33,
3.18] more correct answers, at precision 0.963 to 0.868; and here EDSM and RPNI
finally differ, 0.903 against 0.868.

**A correction to Step 18.** The true partition is not a ceiling on how much can
be answered, only on answering without error: blue-fringe answers 0.201 in
open-world against the true partition's 0.146, by over-merging and being wrong
3.1% of the time. Call it the error-free ceiling.

**And the mechanism behind Step 18's anomaly.** Directed collection raised the
ceiling but lowered the answer rate because it pushes "cannot fold into any
class" from 0.672 to 0.779: more nodes with partial evidence and more genuine
contradictions make the partition finer, so target histories fold in less often.
The rule did not get more cautious; the partition got finer.

**The safety price, which only the irreversible family can measure** — the
overlap family contains no fatal questions at all. Sixty source/target pairs over
three tasks. Closed-world is **not** the danger: it moves fatal-misjudgement from
0.0022 [0.0014, 0.0032] to 0.0023 [0.0016, 0.0034] while taking correct answers
from 0.042 to 0.115. The danger is wrong merges, and their characteristic error
is **saying a fatal event advances the task**, not saying it is harmless: 7.9% of
truly fatal questions for the current rule, 12.3% for blue-fringe, 0.0% for the
true partition. That is exactly the error that bites when structure is used to
recommend a goal.

**Inputs to step three.** Adopt closed-world completion as the default: it is the
only free change found, and it widens the surface a selective-use policy can act
on from 7.3% to 19.2% of novel decisions. Do not switch to blue-fringe where a
wrong commit is expensive. And trust now has measured tiers on the same batch and
partition: observed evidence (precision .963, fatal error .0022), closed-world
completion (.979, +.0001), extra answers bought by over-merging (.868, .0033).

**What is left is not an inference problem.** Even the true partition with
closed-world completion leaves 0.672 of novel questions unfoldable. Moving that
needs better collection, and directed collection makes it worse. That is the open
question for the next round.

## Step 20 — the minimal cross-map comparison: structure adds nothing here, and the ceiling was 1.05 before it started

Third step of the route. Task semantics and event labels fixed across the pair;
navigation skills, map model and every value table reset on the target. Only task
knowledge crosses, and the arms differ solely in what that knowledge is and how
far it is trusted: nothing, the frozen tree read by exact history, the same tree
folded through the conservative partition, the same tree folded through the
over-merging partition, the tiered policy, and the true machine. Groups two to
five share one frozen batch. Verification spends the target budget like any other
step. 25 map pairs by 4 source seeds by 4 target seeds.

**Both of the criteria the route set are answered, and both negatively for
structure.** Selective use beats using the structure directly, **1.161 [1.125,
1.196]**, so the tiering works. It does not beat the dictionary, **0.838 [0.769,
0.915]**, so the answer to "can group four beat both" is no. And the stated
failure criterion fires: the **true** task machine against the same-evidence
dictionary is **1.000 [0.946, 1.059]**. Stated plainly: in this condition a
complete task structure has no value over a same-evidence history dictionary.

**The finding that matters more is that the experiment could not have shown much
either way.** Learning from scratch on the target reaches 90% in 922 steps; the
true machine in 878. The whole available prize was **1.051 [0.989, 1.120]**, an
interval containing 1. This project's own rule is to compute the ceiling before
building the mechanism, and the ceiling here is one free division,
oracle-over-scratch. I did not compute it before running and should have.

**Why the ceiling is so low, and it is a real result.** The goal reading learns
this task on a new map in about 900 steps. The property that made Step 15's
result large — task knowledge being cheap to acquire once the task is factored
into goals — is exactly the property that makes knowing the task in advance worth
almost nothing. The ceiling does widen as the maps diverge, 1.02 at full ordering
overlap to 1.12 at zero, but never becomes interesting.

**Charging the source sampling reverses the sign.** Total cost, target steps plus
the 40,000 source steps: dictionary over scratch is **0.023 [0.020, 0.025]**.
Forty thousand source steps bought forty-four target steps. Whether transfer pays
cannot be reported without the acquisition cost; without it even the direction is
wrong.

**One finding independent of the ceiling: merging dilutes first-hand evidence.**
Using the structure is 0.722 [0.659, 0.799] against the dictionary and the
over-merging partition is 0.041, on identical evidence. At a history the agent
has a record of, the exact per-history evidence is the best available; folding it
into a class mixes in whatever the other members saw, and a wrong merge mixes in
something false. Inference is for histories with no record, not for histories
with one. That is why the tiered policy's first tier is first-hand evidence, and
why it recovers 1.161 over using the structure directly.

**Two implementation traps, both caught by the pilot.** The arms initially
differed in their tie-break as well as in their evidence source — `learned` took
the most-frequently-advancing event while the class readers took all advancing
events — so the comparison would have measured the tie-break. Fixed by giving the
partition class-level advance counts. And two `replace` calls silently failed to
match, so the build succeeded and nothing changed; this project has lost a result
to that before, so every edit is now grepped afterwards.

**Where this leaves the line.** Not "transfer does not work" but "to study
transfer you need headroom, and the goal reading removed it in this family".
Three options: a family where even the goal reading is slow; the irreversible
family, where the learned router sits at 0.75 of the true one and there is at
least some room; or abandoning this line and returning to coverage, where Step 19
left 0.67 of novel histories unfoldable even under a perfect partition with
closed-world completion. The cheap first move is the offline half of the first
option: compute ceilings only, train nothing.

## Step 21 — the ceiling survey: knowing the task is worth about a tenth, everywhere, in the goal reading

Step 20 found its own ceiling after the fact, and the ceiling is one free
division: true structure over learning from scratch. This step performs that
division across everything the family generator can reach, under **both**
readings. No new algorithm; two arms per family. 24 families: n in 3,4,5 by
bridge 2,4 by tails (1,4),(3,6) by corridor (2,7),(3,11), 8 seeds, budget
300,000, nothing censored.

| | median | max | min |
|---|---|---|---|
| goal reading ceiling | **1.11** | 1.57 | 0.79 |
| index reading ceiling | **1.81** | 5.09 | 1.26 |

**No family parameter opens the goal reading's ceiling**, and longer corridors
close it further: 1.19 at corridor 7 against 1.03 at corridor 11. Meanwhile the
index reading's ceiling climbs steeply with difficulty, to 4.4-5.1 at n=5.

**The mechanism, on the hardest family.** Index reading: 44,969 knowing the task
against 216,719 not knowing it. Goal reading: 1,562 against 1,375. The index
arm's ignorant version needs **158 times** more steps than the goal arm's. A hard
task is hard for the index reading, so foreknowledge is precious there; the goal
reading does not find it hard, so foreknowledge is nearly worthless. This closes
the causal loop from Step 15: the goal reading is fast because progress feedback
**directly names the next event**, so task structure in that reading is not
something to be inferred, it is something handed over. Inferring, compressing and
transferring it can only work inside the tenth that is left.

**A caveat on the measurement.** The oracle arm is the true machine plus
skill-value goal selection, not the optimal use of perfect structure; six of the
24 families have the learned router *beating* it (min 0.79), because the router's
"most-frequently-advancing event" is a stronger prior than ranking every allowed
event by a noisy early skill value. So these are approximate ceilings. The
direction is unambiguous: nothing exceeds 1.57, and the denominator is already
only a few hundred steps.

**What this closes.** The transfer line, across the whole family space rather
than one map pair. And the coverage gap from Step 19, since filling all of it is
still bounded by the same 1.11. The project's central question now has a clean
split: **changing the reading is worth 6.78 [6.19, 7.42] over the strongest
prior-art arm; everything done to the structure afterwards is capped around
1.11.** The large half is the reading, not the structure — the same pattern as
every other line here, except that this time even the economy is thin.

**What stays open, all outside the current design space.** Irreversibility, where
the ceiling is 1.33 rather than 1.11, the same order. **Weaker feedback** — the
progress signal here directly reports that the task state changed; with terminal
reward only, or with noisy labels, the goal reading could no longer take task
structure for free, and this is the only direction that could move the ceiling by
an order of magnitude. And function approximation with stochastic dynamics, still
deferred. Recommendation: under the present feedback, further work on structure
inference has no room; either weaken the feedback so structure must be inferred
again, or stop and write.

## Step 22 — separating task rules from the source map's taste

A second external review found two defects, asked for two claims to be narrowed,
and proposed one new control. All four done.

**Defect one, reachability enumeration.** `taskfile.py` used a stack and closed
each (cell, history) at whatever depth it first reached; under a finite horizon a
long route arriving first blocked a shorter route from being expanded. Replaced
with equal-cost BFS, and the review's independent figures reproduce exactly: 176
task files, 13,914 to **13,970** histories, 14 files affected, `taskQ_high0B`
103 to 109. Steps 18 and 19's offline audits were recomputed; the ample-budget
novel-history numbers move from 0.096 to 0.084 for the learned machine and from
0.192 to 0.217 for the error-free ceiling, with the dictionary still at 0.000.
Conclusions unchanged. Two regression tests added.

**Defect two, closed-world completion is not a pruning licence.** Pairing
open-world against closed-world rows: the reversible family gains 2,402 answers
of which 2,377 are genuinely "ignore", 25 are advancing events called ignore, and
none are fatal; the irreversible family gains 2,436 of which 24 are advancing and
**5 are fatal events called harmless**. The overall precision gain comes from the
mass of correct ignores. These rows are query instances from repeated runs, not
independent events, so no interval and no accident rate is claimed. Step 19's
conclusion is narrowed: closed-world completion improves prediction accuracy **on
this distribution** and may be used as a revocable ranking preference, never to
delete an action because it was never seen.

**Narrowing one, the ceiling had a mismatched oracle.** `value` took the true
allowed set and picked by skill value; `learned` first narrowed by advance counts.
Two differences, not one, which is why the oracle sometimes lost. Added
`--goal-select value_ranked`: true allowed set through the identical selection
rule. Re-running the 24 families, the median ceiling moves 1.11 to **1.08**, and
the families where the oracle loses drop from 6 to **2**. Individual families move
a lot (0.79 to 1.11 in one case); the aggregate does not. Two wordings corrected:
the ratio is scratch over oracle so the denominator is the oracle, not scratch;
and "the line is closed" becomes **"under this parameter grid, map sample,
feedback condition and selector, no large prize for knowing the task in advance
was observed."**

**The new control, and it is the step's result: source advance counts are a
behavioural preference, not task semantics.** Same frozen batch, same partition,
same target-side selection rule; the only switch is whether source counts may
narrow the candidate set. The transferred evidence is untouched — only the
ranking weight restarts at zero and re-accumulates on the target. Pilot on 5
high- and 5 zero-overlap development pairs; confirmation on **25 freshly
generated pairs sharing no map seed with the development batch**, because only
five zero-overlap pairs exist there and all were used in the pilot. 35 blocks
pooled.

| comparison | all 35 | overlap 1.0 (15) | overlap 0.0 (10) |
|---|---|---|---|
| no-frequency ÷ with-frequency, dictionary | 1.037 [1.000, 1.077] | **0.968 [0.910, 1.021]** | **1.059 [1.006, 1.118]** |
| dictionary with frequency ÷ scratch | 1.023 [0.971, 1.080] | **1.166 [1.086, 1.253]** | **0.893 [0.827, 0.953]** |
| dictionary without frequency ÷ scratch | 1.061 [1.012, 1.110] | 1.129 [1.063, 1.202] | 0.946 [0.869, 1.008] |
| matched oracle ÷ scratch | 1.108 [1.066, 1.156] | 1.125 [1.053, 1.200] | 1.041 [0.983, 1.107] |

**The sign flips with ordering overlap.** Where the two maps force the same
orderings the counts are useful knowledge, the strongest positive transfer in the
table at 1.166. Where they share none the counts are **resolved negative
transfer** at 0.893, and removing them takes it to 0.946 with an interval
containing 1. **Carrying evidence without frequency is the only variant that
never harms at any overlap.** The boundary between transferable semantics and
non-transferable preference has a concrete member: advance counts are outside it.

**It does not explain the structure arms.** Structure sits at 0.661-0.699 of the
dictionary, while no-frequency over with-frequency on the structure arms is only
1.027-1.041 with intervals containing 1. So Step 20's proposed mechanism, that
merging dilutes exact per-history evidence, remains a candidate and is still not
isolated. A related diagnostic: the ranking weights intervene 120k-130k times per
run and change the chosen goal 33k-54k times, yet outcomes move only a few per
cent, because target-side skill values usually correct the choice. That is why
these effects are directionally clear and small.

**Next, per the review's order:** progress verification on a budget — event
labels stay visible and accurate, but confirming whether an event advanced the
task costs budget, at full, limited and terminal-only levels. Missing feedback
must encode as *unknown*, never as ignore, and history filtering may no longer
use hidden true state changes or turning feedback off still leaks the structure.

## Step 23 — pricing the progress question: structure costs ~176 queries, and the ceiling was conditional on that question being free

Every experiment before this one told the agent, free and on every step, that the
task state had changed. This step meters exactly one thing: event labels stay
visible and accurate, the terminal signal stays free, and asking **"did that
firing advance the task?"** costs one verification.

**Why it does not leak.** A `(history, event)` pair never paid for stays
**unknown** — nothing recorded — and the agent falls back to acting as though it
did not advance. That fallback is the agent's own guess, so the history is built
only from labels it owns. The frozen-policy rollout uses the same cached labels
and never touches the true machine. Re-observing a pair already paid for is free:
a verification buys the *label*, repetition only sharpens the counts. Control:
with the budget set to 10^9 the run is **bitwise identical** to the free-feedback
path, so the metering itself changes nothing. Four tests hold this, including
"terminal-only must leave the tree at one node", which is the direct check that
turning the feedback off really removes the structure.

**The curve, 24 families by 6 seeds, environment budget 200,000.** Steps to 90%
for the learned-rules arm: 0 verifications never (0/144, tree of 1 node); 40
never (1/144); 80 → 115,757 (61/144); 160 → 20,710 (127/144); 320 → 1,264
(144/144); unlimited → 1,264. Given the rules outright: 1,142. Given over learned:
**≥175 [153, 204]** at terminal-only (censored, so a floor), **101 [69, 134]** at
80, **18.1 [5.5, 33.3]** at 160, **1.11 [1.06, 1.17]** at 320 and above.

**The price of the structure is ~176 verifications**, 63 to 328 across families,
and the threshold roughly doubles with task size: median 80 at n=3, 160 at n=4,
320 at n=5. That is a more meaningful complexity measure for this project than
state count: it is how many questions the agent must ask.

**What it does to the ceiling.** Step 21's 1.08-1.11 carried an unstated
condition. Charging the queries at c environment steps each, given-over-learned
becomes 1.11 at c=0, **1.26 at c=1, 2.65 at c=10, 16.5 at c=100** (learned
1,264 + 176c against given 1,142). So the honest form of "no large prize" is
**"no large prize when confirming a step of progress is no dearer than taking a
step"** — which a human check on a robot plainly is not.

**And it reframes Step 15.** The goal reading is fast partly because progress
feedback names the next event; that feedback now has a price, and the goal
reading is the reading most dependent on it — 0/144 with terminal-only signals.
The index reading's history arm leans on the same progress filter (Step 14), so
this is not peculiar to the goal reading, but the whole line has been standing on
a strong feedback assumption that was never stated.

**Next, and now with coordinates.** The review's third step becomes runnable:
turn limited feedback into priced queries and compare, under one query budget,
random querying against querying on disagreement against querying only where a
disagreement could change the current decision. The informative band is **80 to
320 queries**, which is the whole transition from total failure to saturation;
below it there is no signal and above it no difference. Report environment
interactions, query count, success, failures and negative transfer together, with
the cost weight public and a sensitivity table in the form given above.

## Step 24 — where to spend the queries, and the first thing the automaton actually saves

Step 23 priced the progress question and located the informative band at 40 to
320 queries. This step compares **query policies** inside that band, and
separates them from a second choice that had always been bundled with them:
**what to believe about a firing no query was spent on.** Four policies (ask
about every new pair; coin-flip; ask only where the agent does not yet know what
to do; ask only where the structure has no opinion) crossed with two fallbacks
(assume it did not advance; adopt the class's verdict). Every arm that consults a
partition consults the same blue-fringe one, because the conservative partition
barely merges and would collapse two of the four policies into the others.

**The ranking of policies reverses with the budget.** At 80 queries, asking only
where the agent is lost beats asking about everything new by **3.05 [1.96,
4.89]**, solving 116/144 against 59/144 on 28 queries against 79. At 320 it loses
by **0.03 [0.02, 0.05]**, because it still spends only 28: once a history has one
known advancing event it never asks there again, so it is cheap and it can never
repair a node whose single answer was wrong. It plateaus at 116/144 forever.

**The crossover has a price.** Total cost as steps plus c per query: the two
policies are equal at **c = 258 steps per query [162, 385]**. Below that, ask
about everything new; above it, ask only where you do not know what to do. A
human confirmation is plainly above the crossover; an automatic check usually
below.

**Random skipping is not merely worse, it is degenerate**: 0.59 at 80 queries,
0.09 at 160, 0.01 at 320, and it never spends its budget (85 of 320). The
mechanism is worth stating because it is a property of the interface: skipping a
query on an advancing firing makes the agent believe it did not advance, which
**desynchronises its history from the task**; it then sits at the wrong node,
stops meeting new pairs, and so never gets another chance to spend. In this
interface **a skipped query is not a deferred query, it is a corrupted state
estimate** — and that is the real cost of the rule that unknown must be acted on
as "did not advance".

**The first thing the automaton actually saves.** Asking only where the structure
has no opinion, at budget 320 and with an identical outcome: **172 queries down
to 142**, a saving of **30.0 [25.1, 35.0]**, ratio **1.211 [1.191, 1.233]**, with
steps to 90% identical at 1,233 (ratio 1.000) and 144/144 both. **Twenty-one per
cent fewer questions for the same result.** What it saves is exactly the
closed-world effect of Step 19: almost every unseen event genuinely does not
advance, so the places where the class says "does not advance" need not be paid
for.

**But adopting the structure's answers buys nothing.** The class fallback is 0.99
to 1.00 against the own fallback at every budget, and `novel` with the class
fallback spends *more* queries (193) than with the own fallback (142). **The
structure's value is in deciding what not to ask, not in answering.** Those two
had always been bundled; separated, only the first half survives.

**Position in the line.** Step 23 showed every earlier ceiling rested on free
progress confirmation. This is the first positive result after pricing it, and
its shape is specific: structure does not make the agent learn faster (1,233
against 1,233), it makes the agent **ask 21% less**. What that is worth depends
on the price of a confirmation; the 21% saving is net positive at any price,
since it costs nothing in outcome.

## Step 25 — the 21% does not go higher without paying for it

Three attempts to enlarge Step 24's query saving, all under one rule: a saving
counts only if the outcome does not degrade, so steps to 90% and the solved count
are reported beside every query count. Four skip rules crossed with three
consulted partitions (conservative, over-merging, and the true one as a
privileged ceiling) and two support thresholds. 24 families by 4 seeds, budgets
160 and 320.

**The frontier at budget 320.** arrival 173.1 queries, 1,255 steps, 96/96.
**novel 144.2, 1,255, 96/96 — 1.20 [1.17, 1.23] fewer queries at an identical
outcome.** aligned 156.9, 1.10 [1.09, 1.12]. closed 85.2 queries, **2.03 [1.86,
2.22]** fewer, but 30,242 steps and **81/96**. closed on the true partition 52.5
queries, 3.30 [2.98, 3.60] fewer, 47,445 steps and **74/96**.

**The principled fix is worse than the crude rule.** `aligned` skips only where
the class says the event does *not* advance, so that skipping and the fallback
agree, and pays wherever the class says it advances. The argument is sound and
the measurement is 1.10 against the crude rule's 1.20. The pairs where the class
says "advances" mostly sit at nodes where the agent already knows another
advancing event, so paying there changes nothing. **The places that most deserve
a question turn out not to affect the decision.**

**Closed-world completion is a real saving but not a free one.** A single-family
probe showed 192 queries down to 50 *and* a faster run; across 24 families it is
2.03x fewer queries and fifteen families that never reach 90%. Its error mode was
already priced in Step 19 — calling an advancing event "ignore" — and its
consequence here is the desynchronisation of Step 24. **A single-family probe is
not a grid, and this project nearly repeated that mistake.**

**A better partition does not make a better skip rule.** The true partition with
`novel` solves 32/96 and with `aligned` 57/96, against 96/96 on the conservative
one. It merges histories the agent's node-level machinery keeps apart, so its
verdicts get applied confidently at nodes whose belief state differs, and the
agent skips questions it needed. Same pattern as everywhere else in this project:
making the structure more correct is not the same as using it better.

**If one is willing to pay.** Total cost as steps plus c per query: closed on the
conservative partition crosses arrival at **330 steps per query [159, 518]**, and
on the true partition at 383 [216, 590]. Both are quoted as pricing only, not as
recommendations, because **the total-cost view hides the solved count**: a rule
that never reaches 90% on fifteen families is not better for being cheaper.

**Conclusion.** 21% is the non-degrading frontier here. That tightens rather than
weakens Step 24: structure buys **fewer questions**, by 1.20 [1.17, 1.23], and
that is close to the ceiling of this mechanism. Going further needs neither a
better partition nor a stronger completion but an interface in which the agent
can **notice that it missed an advance** — it currently has no way to detect the
desynchronisation, which is the single gap both Step 24 and Step 25 point at.

## Step 26 (P0) — the desync, cache and evidence-attribution audit

The question: how much of the negative results comes from wrong caching rather
than from the query policy. Labels are filed under the agent's history node; once
it skips a query on a firing that did advance, its history stops matching the
task and every later label is filed under, and read from, a node describing a
different task state. Three cache settings against the same policies and budgets:
node with reuse (what every experiment did), node without reuse (isolates the
policy from the cache entirely), and **true task state with reuse**, which is
privileged and exists only as the ceiling on perfect evidence attribution.
Counters are measurement only. 24 families by 4 seeds by two budgets. The default
path is bitwise unchanged.

**The best two policies do not desync at all, so Step 24's 21% is not
contaminated.** In 200,000 steps `arrival` desyncs once and uses a stale label
zero times; `novel` desyncs 77 times, uses 19 stale labels and misses zero
advances. That is P0's most important output: the standing positive result does
not need redoing.

**Random skipping's collapse is substantially, but not wholly, attribution.**
Same budget and policy, labels filed by true task state: solved goes **3/96 to
41/96**, 1.53 [1.31, 1.82] faster. It is the only arm missing four figures of
advances (4,585). But 55/96 still fail, so it is both an attribution failure and
an information failure.

**The `decision` plateau is not attribution, and perfect attribution makes it
worse** — 77/96 down to **20/96**, 0.25 [0.17, 0.35]. It asks only where it does
not yet know what to do; sharing labels across all histories of one task state
makes nodes stop being "lost" sooner, so it asks even less (27 to 24) and learns
even less. **Perfect attribution makes a lazy policy lazier.**

**The cache is necessary, not a liability.** Disabling reuse ruins every policy
(0 to 61/96); `arrival` spends its whole 320 and still solves only 61/96. The
cache's errors are far smaller than the cost of not having one.

**A number this project did not have: the attribution ceiling is 2x and the
structure currently captures a quarter of it.** All four arms below give an
identical outcome, 96/96 at 1,271 steps; only the query count differs. Baseline
173.4; the structure rule 138.8, **1.25 [1.21, 1.29]**; perfect attribution 84.9,
**2.04 [1.92, 2.15]**; both together 77.4, 2.24 [2.10, 2.36]. They are not
additive because **merging is approximate attribution** — perfect attribution
alone gets 2.04 and the structure adds only ten per cent on top. Tightening the
budget to 160 magnifies it: steps 19,857 to 1,271, a factor of **15.6**, and
solved 85/96 to 96/96. The tighter the queries, the more attribution is worth.

**What this does to the hypothesis that fixing desync is the way to grow the
saving.** It stays a hypothesis, now partly supported and partly contradicted.
Supported: the attribution ceiling, 2.04x and up to 15.6x at tight budgets, is
far above the structure's 1.25x. Contradicted: the two best policies never
desync, so what they gain from perfect attribution is **not desync repair but
sharing evidence across equivalent histories**, which is what a correct automaton
does and is a different problem from detecting desync. And `decision` gets worse
under perfect attribution, so better attribution is not an improvement for every
policy. The accurate statement is: **in this interface the headroom is in evidence
attribution rather than in query policy, and desync repair only matters for
policies that desync.** The true-state cache is a privileged reference, not a
method.

## Step 27 (P2 + P3) — information is not the bottleneck, and only one conclusion travels to a public domain

**P2: neither too little information nor our own discarding of it.** Which label
fired is always visible; only whether it advanced costs a query. So baselines that
keep the whole visible event stream spend **zero** queries. 24 families by 6
seeds: counting events 0/144, the last two events 0/144, the last four 0/144, and
**every observed event 0/144 with 18,035 distinct states and 13,939 table
overflows**. The metered progress-history arm also fails, 0/144 at both 80 and
320 queries. The goal reading solves 144/144 on 175 queries. So the failure is
that **the representation cannot use raw information**, not that information is
missing and not that we threw it away. The sharpest internal contrast: progress
history with 320 queries solves 0/144 while the goal reading with the same labels
and the same query budget solves 144/144.

**P3: OfficeWorld, exported exactly.** `get_observations` depends only on the
agent's cell and movement is a deterministic function of cell and walls, so
(cell, action) to (cell, label) is an exact table; 15 tasks across five task
families and three map seeds, 108 cells, 4-6 task states, zero multi-label cells.
Arms are grouped by what they are given, because a method handed the rules may
not be tabled against one that must learn them.

**One conclusion travels, and it shrinks.** The goal reading beats the index
reading given the same true rules, **2.10 [1.29, 3.56]**, and beats index+CRM
**2.69 [1.92, 4.22]** — against **6.78 [6.19, 7.42]** on our own family. The
number to quote outside is 2.69, marked as a public-domain figure.

**Two do not travel, and one reverses.** OfficeWorld's learned prefix tree has
**3 to 4 nodes against our family's 73**, so the whole task needs only **14 to 17
verifications**: the informative 80-to-320 band does not exist there, and budgets
of 160, 320 and 640 give bitwise identical results. With 17 queries to spend
there is nothing to save, so Step 24's 21% becomes **0.98 [0.92, 1.00]** — the
one positive mechanism of the last several steps **does not appear on the public
domain**. And CRM, worth 2.51x in Step 16, is *worse* than plain indexing here,
0/90 against 20/90: the same mechanism has opposite signs in the two domains.

**The sentence that matters most for positioning: our synthetic family's
structures are about twenty times larger than the public benchmark's** (73 prefix
nodes against 3-4; 24 minimal classes against 4-6). The entire paid-verification
line lives in a regime the public benchmarks do not occupy. Two readings, and the
second is closer to the evidence: either nobody has measured how query cost scales
with structure size, which is a gap; or our positive mechanism appears only on
self-built tasks far more complex than anything currently benchmarked, and **there
is at present no public evidence that it helps on problems other people care
about**. Making the first reading stick needs *public tasks with larger automata*,
not more sweeps on our own family.

Absolute performance on OfficeWorld is poor for every arm (best 68/90 at 137,678
steps); ISA uses QRM with subgoal options and far more episodes there, so these
numbers support arm-to-arm comparison only and must not be set beside the curves
in the ISA papers.

## Step 28 — the review's four points, and two retractions

**P2 had two defects, both real.** The history a representation reads was tied to
the method name, so `count` and `window` read the verification-filtered progress
history rather than the raw event stream; at zero queries all three saw an empty
history and produced **identical trace hashes**, which the review's independent
check reported and which reproduced here to the digit. Source and encoding are
now separate (`--history raw|progress`), and `--window-pure` stops a window from
also hashing the total prefix length. Corrected, the three arms genuinely differ
(42 / 30 / 336 / 18,122 states, 144 distinct traces each) and the conclusion is
unchanged but now earned: at zero queries, 0/144, 0/144, 1/144, 0/144.

**The second defect mattered more: I ran a different experiment from the one
proposed.** The proposal was to keep the goal controller and file the query
evidence under the complete visible prefix; I swapped the controller for
raw-history-indexed Q-learning, changing two things at once. The comparison as
asked, one controller and three filing keys, solved counts out of 144:

| filing key | 20 | 40 | 80 | 160 | 320 |
|---|---|---|---|---|---|
| progress-history node | 0 | 3 | 60 | 121 | 144 |
| complete visible prefix | 0 | 0 | 4 | 57 | 113 |
| true task state (privileged) | 0 | 12 | 105 | 144 | 144 |

**Filing by the complete visible prefix is worse at every budget**, because it is
a *finer* key, not a coarser one: every distinct raw prefix must be paid for
separately and evidence never pools. Attribution wants coarseness aligned with
the task, not completeness. That is the answer P2 was supposed to produce, and it
runs opposite to "keeping more information must help".

**Two retractions on P3.** The 2.69x was a ratio of censored means (CRM censored
90/90, the goal arm 24/90) and is withdrawn as a speed figure. And "CRM fails on
OfficeWorld" is withdrawn: it was **my derived protocol**. The native environment
gives 1 on success and 0 otherwise from a single start with a 250-step episode,
and ISA's configuration uses epsilon .1 and rate .1; I had used a -0.01 step cost,
a re-chosen start set and a horizon of twice the shortest path. The review's
hypothesis that a negative step cost plus fast failure termination biases early
exploration is **confirmed**. Re-run under the published protocol, 15 tasks by 8
seeds, AUC as the headline rather than any censored metric: goal reading 0.977,
index+CRM 0.877, index 0.479. Goal over CRM is **+0.100 AUC [+0.056, +0.146]**;
CRM over plain indexing **+0.398 [+0.223, +0.579]**, so Step 16's CRM result
stands. Learning the rules versus being given them: **-0.000 [-0.001, -0.000]**.

**One observation to flag rather than assert:** the index, merged-index and
progress-history arms return byte-identical aggregates, and their trace hashes
are identical with `features` = 2. The agent never gets deep enough on this task
for the three representations to differ, so they are one row, not three.

**The scarce-query region now has data.** 160/320/640 were all saturated; sweeping
0/4/8/16/32/64 puts the transition between 8 and 32 (1/90, 45/90, 65/90). Inside
it, sharing evidence still changes nothing: `novel` 43/90 against arrival's 45/90,
the privileged true-state cache 46/90. So the negative result now covers the
scarce region rather than resting on an over-generous budget — while still only
saying that a domain needing about seventeen queries has nothing to save.

**Still undone:** the review's priority 2, running the authors' own QRM against
our index and CRM on the same task and protocol. Our side is now aligned to the
published protocol; the cross-implementation check is not done, and only it can
separate implementation from protocol from mechanism. Also recorded: 73 against
3-4 is *learned prefix-tree* size and 24 against 4-6 is *minimal class* count;
they must not be merged into one "twenty times larger automaton" claim.

## Step 29 — our CRM and the authors' QRM agree

The review's priority 2: use the authors' implementation to separate our
implementation from the protocol from the mechanism. Step 28 aligned our side to
the published protocol; this puts ISA's own QRM into the same one.

**Two configuration keys, and the first one alone is a trap.** Running ISA's QRM
on the true automaton needs `interleaved_automaton_learning: false` *and*
`initial_automaton: "target"` — the field is not `initial_automaton_mode`. With
only the first, QRM receives a trivial automaton and its success rate is 0.000
for all 4,000 episodes. Another instance of installation not being a baseline.

**Everything else published: OfficeWorld DeliverCoffee, environment seeds 0/1/2,
reward 1 on success and 0 otherwise, single start, 250-step episodes, epsilon .1,
rate .1, discount .99.** Our side runs the exported `ow_native_coffee_*` with the
same constants, three training seeds per map.

**An episode budget is not a step budget**: ISA's 4,000 episodes come to 210,727
steps on seed 0 but 58,909 and 62,643 on seeds 1 and 2, because successful
episodes end early. Any comparison against ISA curves must convert first.

**At the two step counts where all three ISA runs have data, our CRM matches the
authors' QRM**: 0.33 against 0.33 at 25k, 0.33 against 0.34 at 50k. Beyond 75k
only seed 0 still has data, so those columns cannot be compared. **The goal
reading leads at every matched point**: 0.56 and 0.78 against 0.33 and 0.34, and
1.00 by 100k while both index-style methods are between 0.01 and 0.44.

**So the difference is not our implementation.** Step 28's "+0.100 AUC for the
goal reading over CRM" is a mechanism difference, and the retraction of "CRM
fails on OfficeWorld" now has cross-implementation support: CRM works under the
published protocol in both codebases.

**Limits to state whenever this is cited:** one ISA run per map with no seed
repetition against three of ours; **different metrics** (ISA reports training
reward over a 200-episode window, we report frozen-policy success at
checkpoints), so only trend and magnitude are comparable; and one task family on
three maps, a spot check rather than a grid. Raw logs and the ISA config are
archived in `results/qrm_crosscheck/`.

## Step 30 (P1 + P2) — the attribution ladder, and the gap decomposed

Debugging phase closed. One controller, one query rule, one environment, one set
of training parameters; the only thing that varies is **who a paid answer is
shared with**. The query rule is the most mechanical available: ask the first time
an (attribution state, event) pair appears, reuse afterwards. No novel/aligned/
decision heuristics, because mixing attribution with query selection is what made
the earlier results unreadable.

**The ladder, 24 families by 6 seeds, unbounded query budget:**

| key | queries | necessary | redundant | over-split share | stale | missed advance |
|---|---|---|---|---|---|---|
| complete visible prefix | 1,298.7 | 85.3 | 1,213.4 | 93.4% | 0.0 | 0.0 |
| progress-history node | **176.9** | 85.6 | 91.3 | 51.6% | 0.0 | 0.0 |
| learned merged class | 214.7 | 81.7 | 132.9 | 61.9% | **304.2** | **3.8** |
| true task state | **85.6** | 85.6 | **0.0** | 0% | 0.0 | 0.0 |

**"Necessary" is about 85 for every key** — the number of distinct (true task
state, event) pairs the agent meets. It is invariant to attribution, which
validates the decomposition itself.

**The gap is entirely over-splitting.** Of the node baseline's 176.9 queries,
**91.3 are redundant**: that pair was already bought under a different attribution
state. At unbounded budget, complete-prefix, node and true-state all show **zero**
stale uses, so over-merging costs nothing here — not because it is safe but
because none of these rules over-merges. The second metric is therefore empty
until a learned abstraction exists to fill it.

**Worse than expected: the existing learned merge is negative as an attribution
key**, 0.82 [0.80, 0.85] against the node baseline. It spends more (214.7 against
176.9), introduces 304 stale uses and the first missed advances (3.8). The extra
spend is traceable: merging changes beliefs, so it changes the trajectory — nodes
rise from 67 to 98 and more (node, event) pairs exist to buy.

**A configuration fact found on the way:** under `--merge-rule sweep --theta 20
--quota 1` the partition does not merge at all (73 classes from 73 nodes), which
is why an earlier reading showed it bitwise identical to the baseline. With EDSM
it does merge (27/84), taking 192 queries to 179 with 4 stale uses and no missed
advances. So **the safe setting of the current rule buys 192 to 179; opening the
thresholds reaches 165 at the cost of 1,724 missed advances.**

**The research space is now an interval with numbers on both ends:** complete
prefix 1,299 > learned merge 215 > progress node 177 > ? > true state 86. The
target is **177 to 86, a factor of 2.07 [1.94, 2.17]**, and current structure
learning contributes *negatively* to it. Both metrics a candidate must satisfy
are in place: redundant queries (over-splitting) and missed advances (dangerous
over-merging). Node is at (91.3, 0.0), the oracle at (0.0, 0.0), the learned merge
at (132.9, 3.8). A usable learned abstraction has to land down-left of those.

## Step 31 (P3) — the simplest learned abstraction fails its own acceptance test

The task was to move the attribution key from 177 queries toward the oracle's 86
and land down-left of the existing points on (redundant queries, missed
advances). Built the simplest version asked for: **partial feedback signature, a
support gate, and successor closure** — no neural network, no probabilistic model.

**The rule.** Two histories may share when no event has both of them labelled
differently, and they agree on at least `support` events where both are labelled.
The support gate is necessary because a brand-new history's signature is all
unknown and therefore compatible with everything.

**The one non-trivial design point, and a mistake worth recording.** The first
implementation grouped by signature and then refined on successors, and it barely
merged at all (73 classes from 73 nodes). The reason is specific to a prefix tree:
two merged nodes have distinct successors, refinement puts those in different
classes, and that splits the parents straight back, cascading to singletons. **A
merge must carry its successors with it as it is made, rolling back whole on any
contradiction** — the classic state-merging shape, with the visit quota replaced
by signature support. Classes then fell from 73 to 20-34.

**Result, 24 families by 6 seeds, unbounded query budget.** Queries / redundant /
missed advances / steps / solved: node **176.9 / 91.3 / 0.0 / 1,281 / 144**;
signature support 4 **162.2 / 77.9 / 70.4 / 5,427 / 141**; support 3 **157.7 /
75.9 / 153.0 / 15,115 / 133**; support 2 **159.5 / 79.7 / 423.7 / 40,000 / 115**;
true state **85.6 / 0.0 / 0.0 / 1,281 / 144**. Query saving against node is 1.12
[1.09, 1.16] at best, closing 21% of the gap.

**It fails the acceptance test.** Every variant moves left on redundancy and *up*
on missed advances; apart from the oracle, node remains the only point at
(·, 0). And the cost is not confined to that counter: steps to 90% rise from
1,281 to between 5,427 and 40,000 and the solved count falls from 144 to 141, 133,
115. **Trading 1.12x on queries for 4x to 31x on learning is not a trade worth
making.**

**A single-task probe had shown support 3 saving 32 queries with zero missed
advances and unchanged steps. It did not survive the grid.** Third time in this
project a single-family probe has pointed the wrong way.

**Diagnosis, and it is precise: agreeing on the events both histories have
observed constrains nothing about the events neither has.** Nodes carry evidence
on 2.6 of 7 events on average and overlap on 1 to 2, so "agrees on three events"
leaves four unconstrained, and the task-state distinction often lives exactly
there. Raising the gate only trades one failure for the other: support 4 cuts
missed advances from 153 to 70 and the saving from 21% to 16%, and never reaches
zero. **There is no point on this curve with both zero missed advances and a query
saving.**

**So a usable rule needs something other than a higher threshold** — either
evidence on the *same set* of events rather than agreement where they happen to
overlap, or a bound on the risk carried by the unobserved entries. That is the
next concrete question.

**P4 stays closed.** Its condition was a clear move from the baseline toward the
oracle; the best safe point is still the baseline's 177, and the learned
abstraction only reaches 158 by paying in missed advances.

## Step 32 (P3.5) — the identifiability audit, and the answer is zero

P3 used the wrong quantifier. Sharing a paid answer is safe only when **every**
task machine still consistent with the data gives the same answer, not when some
machine merely permits the two histories to coincide. This measures the gap
between the two, splitting the baseline's 91.3 redundant queries into identifiable
and fundamentally ambiguous.

**Two prior facts: the states are easy to tell apart.** Across the 24 families,
the shortest separating suffix between distinct task states is **one event for
96.2%** of pairs, one or two for 97.6%, average 1.09, maximum 8. So P3's failure
is not that the states are subtle.

**The reference arm.** No true automaton; only the data the ordinary learner had.
An ASP encoding of every machine consistent with it: each observed history
assigned to one of k latent states, transition and output functions both
functional, every paid observation respected, **progress must change the state**,
**ignore must self-loop**. For the pending (history, event), ask whether a
consistent machine exists for each verdict. One satisfiable verdict means the
query was inferable for free; two or more means genuine ambiguity. Replayed
against a real query log, so at query i the evidence is queries 0..i-1.

**Result: zero, at every state bound and every stage of the log.** Two tasks, k at
half, exactly, and above the true class count: **0 determined, 40 ambiguous**
every time. Splitting the log into earliest, middle and last twenty: 0 determined
in all six cells. Accumulating evidence does not create determinacy. (An earlier
version reported a few determinations at k = 3, 4, 6, some of which **contradicted
the truth** — that is the signature of a bound too small to contain the true
machine. With the progress-changes-state constraint and an adequate k, both
contradictions and determinations go to zero.)

**So the decomposition is filled in:** 91.3 = **≈0 missed inference** + **≈91.3
missing distinguishing evidence**. Combined with the separator statistics: the
states separate on a single event, but at the moment of the query the agent does
not hold that event's outcome on both histories. The ambiguity is an evidence
property, not an inference weakness.

**This formalises why P3 had to move up-left.** Every query it saved was a bet
between two admissible models, so its missed advances were not an implementation
defect but the price of the route. As a proposition: if at least one of two
histories has not been observed on event e, and the data is consistent both with
their agreeing and with their differing on e, then no algorithm that shares that
feedback without querying e can guarantee zero error. The audit says **almost
every redundant query falls under that condition.**

**Against the pre-set decision rule — "if the answer is only 15, the oracle's 86
is not a learnable target" — the measured answer is 0.** 86 is unreachable on the
present information. **177 states precisely this and nothing more: under the
current hypothesis class, demanding zero prediction error, and acquiring no
further evidence, this is the observed safe bound.** It is not a general claim
that data cannot do better — change the hypothesis class, accept non-zero risk,
or go and acquire evidence, and the bound moves. The three
remaining routes are not better inference: acquire more observation (and the
separators are short, 96% of length one, so this is the one door still open);
add structural assumptions about the machine; or accept non-zero risk, which is
what P3 did and whose price is already measured. **P4 stays closed, for a changed
reason: not "no good algorithm was found" but "no good algorithm exists on this
information".**

**Limits:** two tasks, one seed each, 60 queries audited per task rather than all;
hypothesis class is "deterministic machines with at most k states, progress
changes state, ignore self-loops", and a stronger prior would shrink it — that is
precisely route two, so this is an answer within that class and not a universal
lower bound; and "all consistent models" is only as strong as the encoded
constraints.

## 第三十三步：主动取证的离线价值审计

**背景**：第三十二步证明冗余查询不是推理不足而是证据不足。唯一出路是主动去取证。
在实现 active distinguishing policy 之前，先离线审计这条路值不值得走。

**计价单位改成「合并」而非「查询」**。同一真状态下的冗余来自抽象把历史劈开；
消除它就是把节点并起来，一次合并一次取证。盈亏线 `S > C_nav/λ + 1`，
`+1` 是到了以后读裁决的那一次查询。2190 次冗余 → 345 个歧义簇 → 874 次待办合并。

**第一层（oracle 物理下界）：导航不是障碍。** 874 次合并的 `C_nav` 全部为 0；
即使禁用脚下正在触发的事件，中位 1 步、平均 0.71 步。原因是地图性质（28–40 格、
6–8 个事件、每事件在 2.5–4.1 格上触发）。73.3% 的合并越过盈亏线，净省 1316 次
查询当量。**停止条件未被触发。** 但这是本批地图的性质，稀疏大图上可能反转。

**第三层（事后摊销，不许额外导航）：兑换率太差。** 「共同证据全一致且至少 m 条
就合并」这条规则，最好的工作点 m=4 只覆盖 11.0% 的冗余，且每犯一次错误复用只换
8.3 次省下的查询；五个种子重跑是 9.8% ± 1.0% 与 6.3 ± 1.5。

**一个必须记下的 bug**：`TaskFile.verdict` 返回字符串 `'advance'/'ignore'/'fatal'`，
而 ASP 编码用 0/1/2。`active_value.py` 第一版把字符串直接写进 `obs/3`，于是
**base 程序无解**——而这个模块的每个问题都是以「不可满足」的形式提问的，所以无解
看起来恰好像「所有一致机器都同意」。第一版报出 192/192 = 100% 闭合，**全部作废**。
修法有两条，缺一不可：(1) `VCODE` 显式映射；(2) 新增 `version_space.consistent()`
守卫，任何「不可满足即结论」的提问之前先确认证据本身可满足。
排查过全部 `.verdict(` 调用点，字符串/整数混用只存在于这一个新文件，早先结果未受污染。

**第二层（版本空间，不许偷看真状态）：智能体没法知道自己取证成功了。**
抽样 144 对「本该合并但版本空间不允许」的历史，三种取证预算全部 **0/144 闭合**：
一次访问（物理可行）、「在 h2 上想看什么看什么」、以及「两条历史上的每个事件都看遍」
（物理上做不到，只作上界）。放弃前花掉的取证次数中位 6、最大 8，就是整个事件表。
0 次因不可满足而作废。阳性对照（同一条历史被判为必然合并）、守卫对照（k=1/2/3 正确
作废）、阴性对照（真状态不同仍可分离）都过。含义：单步证据不足以证明等价，剩下的
歧义在后继上，要拆开需要**带重访的长区分序列**——而在不可逆任务上恰恰做不到。

**结论：P3.6（单事件 active distinguishing policy）关闭，理由是机制不成立而非成本过高。**
用户设的停止条件是「oracle separator 下成本 ≥ 节省价值就关闭」，实测成本 ≈ 0，
该条件**未触发**。审计说明第一层量错了瓶颈：主动取证有动作、有代价，但**没有验收条件**。
剩下的两个口子都比 P3.6 重得多：带重访的区分序列（不可逆任务上多半做不到），
或收紧假设类（第三层已量过放弃零错误要求的兑换率是 6.3 ± 1.5，太差）。

同时按用户的要求收紧了 177 的措辞：它是「在当前 hypothesis class 下、要求零预测错误、
且不主动获得额外证据时，观察到的安全界限」，不是「数据能做到的最好」。

## 第三十四步：形式化过程中推翻了第三十三步的负面结论

**收回：第三十三步第二层的「版本空间从不强制合并，0/144」。** 那个数字与数据无关，
是编码决定的：审计用 `k = 真状态数`（19–57），而编码进 ASP 的节点数只有 5–40。
**k ≥ 节点数时，「每个节点自己一个状态」本身就是一致模型**，于是没有任何证据能强制合并。
扫 k 可见窗口只有一格宽（某对历史在 k=6 必然合并，k=7 起就不再）。

**同一缺陷波及第三十二步**：那次用 k=18 / k=28，也远在窗口外，
所以「0 次可判定」测的是上界不是数据。

**正确的上界是 Occam 上界（与证据一致的最小状态数）**，也就是 RPNI/EDSM 一直用的先验。
n3 的 8 个族、全量日志、随机抽 126 对历史：k_min 下判定为同态 42 次、正确 41 次
（**精确率 97.6%，覆盖 66.1% 的同态对**），判定为异态 39 次、精确率 100%，
未定 21、求解超时 24。**先验强度就是全部杠杆**：k_min+1 覆盖率掉到 6.5%，k_min+2 归零。
两项代价：精确率不是 100%（k_min=8–12 小于真状态数 19–27，最小机器过度合并），
且在紧上界上证否即最小一致 DFA 问题，NP-hard，一族 16 对要 400–1000 秒。

**同一先验下「裁决」比「合并」难证得多**：第三十二步的题目在 k_min 上重做，
在线前缀 36.2% 已确定但 **41% 与真值矛盾**；留一法 13.3% 已确定、**0 矛盾**（20 次超时）。
归属需要的是合并而非裁决，所以这条对 177 → 86 是正面的。

**最小反例（已用求解器核对，`src/indistinguishability.py`）**：三个活状态、字母表 {a,b}，
状态 1 和 2 单步输出行相同而后缀 `ba` 区分之；穷举 ≤2 状态、字母表 ≤3 的全部 87 台机器
无反例，故三状态最小。命题收紧为：「单步输出行相同 ⟹ 等价」这条规则不可靠；
**推论：可判定性取决于 k，k ≥ 节点数时加证据也无用，只有压到最小可满足 k 才可能强制合并。**

**对 P3.6 的影响**：第三十三步给的关闭理由「没有验收条件」不再成立——验收条件存在，
只是不在当时那个上界上。是否关闭需要重新给依据。未回答的是：66.1% 能否换成端到端
查询节省；在线 k_min 过小导致的 41% 错误判定如何处理；NP-hard 求解能否降到可在线用。

## 第三十五步：在线 Occam 合并审计 —— 关闭 P3 / P3.6

第三十四步的 66.1% / 97.6% 是**事后现象**。轨迹固定、只用当时证据 \(D_t\) 重放
n3 的 8 个族（792 次查询，冗余 313 次）：

| prior | 复用 | 裁决正确率 | 归属正确率 | 省下占总查询 |
|---|---|---|---|---|
| k_min | 236 | 80.1% | **40.7%** | **16.9%** |
| k_min+1 | 22 | 77.3% | 63.6% | 1.8% |
| k_min+2 | 0 | — | — | 0.0% |

在线 k_min 是 8–12 而真状态数 19–27，证据少时最小机器严重过度合并：236 次强制合并里
140 次把真状态不同的历史证成同态。裁决正确率 80.1% 高于归属正确率，正是因为大量错误归属
碰巧给出相同裁决。236 次复用里还有 102 次落在**必要**查询上，那不是节省而是替换。

**对照预注册判据（节省 ≥20% 且错误共享 ≈0）：三项全不达标。**
敏感性检验：把 53 次求解超时全部判给 Occam，节省上界 23.6% 刚过线，
但错误裁决 47 次、错误归属 140 次一次不少，第二条判据仍不达标。超时不是结论来源。
学习曲线未跑，因为前两条已不达标，且第三十步已警告过查询 177→158 时学习可慢 4–31 倍。

**结论：P3 / P3.6 关闭。** 第三十四步收回的「没有验收条件」有了正确替代：
**验收条件存在，但只在事后成立；在线拿不到。** 先验强度是唯一旋钮且无可用工作点。
NP-hard 的求解代价按既定方针不优化——科学问题已答「否」。

下一步进入 goal-readout 的 factorial ablation（任务分解 × 技能复用）。

## 第三十六步：2×2 机制消融 —— 收益来自路由，不是来自技能复用

两个正交开关（`--route automaton|meta`、`--skill-key shared|perstate`），24 族 × 30 种子 ×
5 臂，预算 200,000，每 250 步一个检查点。结构臂都拿真实任务机；非结构消融刻意不获得
由任务机导出的目标信息，那正是被消融的信道。

| 臂 | AUC | 终局 | first90 中位 |
|---|---|---|---|
| Y00 平坦历史 | 0.821 | 0.992 | 38,500 |
| Y01 学习型元控制器（有复用无路由） | 0.883 | 0.942 | 4,000 |
| Y10 每状态独立技能（有路由无复用） | 0.988 | 0.999 | 3,750 |
| Y11 目标读法（都有） | 0.997 | 1.000 | 1,000 |
| QRM | 0.972 | 1.000 | 7,000 |

配对自助（按族）：只加复用 +0.062 [0.008,0.122]；只加路由 **+0.167 [0.117,0.223]**；
在路由之上再加复用 **仅 +0.008 [0.008,0.009]**；目标读法 vs QRM +0.025 [0.021,0.028]。
**主效应：路由 +0.140，复用 +0.035，交互 −0.054（替代而非互补）。**

按规模：路由主效应 n3 +0.081 → n4 +0.133 → n5 **+0.207**；复用主效应 n3 **−0.006** → n5 +0.090。
有路由的三臂随规模几乎完全平坦，没路由的两臂下滑。24/24 族目标读法优于平坦。

稳健性：`option-cap` 8→256 扫描下 Y01 的 AUC 只在 0.873–0.887 之间（取最好值结论不变）；
平坦臂在最大族用 274 个特征、上限 4096、溢出 0，退化不是表满造成的；
新增两个开关后所有既有臂的 `trace_hash` 与改动前二进制逐位一致。
保留不抹平的一项：Y01 终局成功率 0.942 低于其余臂，学得快但稳得差。

局限：自有任务族、真实任务机、地图小（28–40 格）因而系统性低估复用的价值。

## 第三十七步：HRM CraftWorld 结构审计（训练前）

克隆 `ertsiger/hrm-formalism-envs` 与 `ertsiger/hrm-learning` 到 `external/`。
环境要 2019 版 `gym-minigrid` 分支，装不上；但层级不需要环境，
所以给 `gym_minigrid` 打桩、在只带两个开关的壳上调用层级构造函数
（`external/hrm_craftworld_audit.py`），不 step、不产生观测。

**阶梯成立**：展平后 `book`(8 状态/深度 5) → `book-and-quill`(21/8) → `cake`(15/9)。
非展平层级分别是 3/5/6 个子自动机。

**与预期相反的一条：重复度不比我们自建族高。** 同一指标（一个子目标被多少个活状态要求推进）：
我们 n3/n4/n5 = 4.3 / 7.0 / **13.1**；HRM 最重复的 book-and-quill 只到 6，cake 到 5。
所以第三十六步 Y11 − Y10 = +0.008 **不是因为缺少可复用的重复**。

**我们真正缺的是导航成本。** 我们的地图 28–40 格、四向移动、每类 1 个物体；
HRM exploit 配置是 four_rooms **13×13** 带岩浆、**3 个动作（左转/右转/前进）带朝向**、每类 2 个物体。
学一个导航技能贵得多，所以「学 13 遍 vs 学 1 遍」才会开始有差别。
**这个 benchmark 对 reuse 的价值在于每次重复更贵，不在于重复更多。**

**修正后的预注册预测**：(1) Δ_routing 在我们族里是随状态数增长的，而 HRM 展平状态只有 8–21，
比我们 n4/n5 还小——这里的复杂度轴是**深度**不是状态数，**不是同一个 x 轴**，不能直接合画；
(2) 预测 Δ_reuse 在 HRM 上变大且由导航成本驱动，若 open_plan 7×7 上仍近 0 而 four_rooms 13×13 上转正，
即坐实「reuse 的价值 = 技能的学习成本」；(3) 若 13×13 四房间上 Δ_reuse 仍近 0，那是更强的结论。

**未做：跑起来。** 需要第四个环境（gym 0.15.3 / numpy 1.21 / py37-Linux conda / hrm-minigrid 分支），
本机最低 Python 3.10.21。替代路径：只导出展平自动机到我们的 `.task` 格式、把地图换成 13×13 四房间，
在自有模拟器上跑五臂——语义不动、不碰旧依赖，代价是变成「在作者的任务结构上做机制实验」而非「在作者环境里复现」。

## 第三十八步：HRM 公开任务结构 × 受控导航成本

`book / book-and-quill / cake` 展平后原样导入 `.task`（`src/export_craftworld.py`，
否定字面量恒真因为一格最多一个物体，代码里是断言）。只换地图：cheap = 开放 7×7（25 格），
expensive = 四房间 13×13（104 格）。按预定不加岩浆、朝向、多物体。
3 任务 × 2 条件 × 6 地图 × 20 种子 × 5 臂，预算 400,000。

**先修一个测量缺陷：AUC 差值在这里不是效应量。** Y11 几乎总在 0.994–1.000，
所以 `Y11 − Y10` 只是在量 Y10 掉多少：60 个单元上 corr(Y10 的 AUC, Y11−Y10) = **−0.998**。
改用达到 90% 所需步数（无上界，本批删失 0%）。

**检验 A 通过**：Y11−Y01 = +0.090 [0.075, 0.106]；Y10−Y00 = +0.256 [0.160, 0.362]；
Y11−QRM = +0.060 [0.040, 0.083]。routing 跨到公开任务结构成立。

**检验 B 通过**：差的差 AUC +0.043 [0.032, 0.055]；样本复杂度 **3.79× [3.10, 4.59]**。
复用倍数 cheap 1.17–2.83×，expensive **4.13–13.05×**。

**机制：是 C_g，不是 N_g，也不是乘积。** 用户建议的 C_reuse = Σ(N_g−1)C_g 在 CraftWorld
内部 r = +0.781，但合并自建族后塌到 +0.252——自建族 C_reuse 214.6 比 expensive 的 80.5 还高，
收益却更低。拆开后 60 个单元：**C_g r = +0.756**，N_g r = +0.078。
三组均值只在 C_g 上单调（2.1→5.6→7.4 对 2.00×→3.20×→5.81×），
按 N_g（2.8→8.2→2.8）和 C_reuse（22→215→81）都不单调。

**因此修改第三十六步。** 同一批数据改用样本复杂度：主效应 **路由 6.39×、复用 5.34×、交互 0.30×**；
「路由之上再加复用只值 +0.008」作废，实为 **3.20× [2.94, 3.50]**。路由仍略强，但不是 4:1。
已在 `REPORT_STEP36.md` 顶部加注。

**局限**：组内相关与组间方向不一致（C_g 在自建族内部是 −0.569），单调性只建立在三个组均值上；
未检验不可逆性与复用的交互；仍是自有模拟器与四向动作，作者原生环境留作第三十九步的外部效度检验。

## 第三十八步半：不可逆性 × 技能复用的成对 2×2

同一张网格导出两次：`safe` 危险格标签无害自环，`lava` 踩上去整幕失败。
字母表、物体位置、危险格位置、墙、最短路全同；**危险格始终可走进去，不是墙**，几何未变。
起点与时域只用 lava 变体算一次，两格共用。密度 3%（便宜 1 个、昂贵 3 个）——
**这个剂量是调出来的**：8% 时所有臂都删失，两个删失臂之间的比值不是数字。
3 任务 × 2 导航 × 2 风险 × 6 地图 × 15 种子 × 5 臂，预算 1,000,000。
口径按预定：达标率 + 条件中位 T90 + RMST 三样并排，不把预算填进去凑比值。

**问题一：岩浆极大地放大复用。** 仅路由 → 目标读法的 RMST 差：
cheap/safe +0.000，cheap/lava +0.023，expensive/safe +0.018，**expensive/lava +0.827**
（达标率 11% → 95%）。**差的差 = +0.416 [+0.270, +0.566]**，本项目至今最大的效应。
机制诊断吻合：达标前累计不可逆失败，便宜地图目标读法 4 次 vs 仅路由 24 次，
昂贵地图 269 vs 785——**安全经验可以跨任务状态复用**。

**问题二：岩浆也放大路由，但先修了一个公平性问题。** 学习型元控制器会把危险格本身选作目标，
而自动机路由臂永远不会提出它。新增 `--meta-safe`（oracle 帮助，只作对照）：
Y01 的 RMST 从 0.213→0.396（cheap）、0.044→0.161（expensive），灾难从 ~25 万降到 ~300。
用打补丁后的 Y01：Y01→Y11 的 safe→lava 差是 cheap +0.502、expensive +0.688；
Y00→Y10 是 cheap +0.853、expensive **−0.081**（两臂都贴地板 0.000/0.106，地板效应）。
四个对照三个明显为正。**路由减少选错目标，复用减少执行同一目标时重复踩坑。**

**额外发现：QRM 在不可逆条件下塌得最惨。** 昂贵+岩浆时达标率 1%，
达标前平均 116,619 次不可逆失败，比目标读法的 269 次多两个数量级。
索引读法没有任何机制把「怎样安全地做 X」从一个任务状态搬到另一个。
这是目标读法与 QRM 迄今最大的差距，且只在不可逆条件下出现。

**局限**：密度是调出来的，效应的存在不依赖它（2% 时仅路由臂同样完全删失）但大小依赖；
只改代价不改几何，未测几何型不可逆；仍是自有模拟器与真实任务机。

## 第三十八步半 修正：问题二不成立，并补剂量稳健性

**收回「岩浆也放大路由」。** 原因是两个缺陷叠加。
其一是公平性：学习型元控制器会把危险格本身选作目标，自动机路由臂永远不会，
所以必须有 `--meta-safe` 对照（oracle 帮助，只作对照）。
其二是我写的 bug：`--meta-safe` 第一版只在训练时排除致命目标，漏了两处——
冻结评估策略仍在全字母表取 argmax；更要命的是 SMDP 的 bootstrap `meta_best` 也在全字母表取 max，
而被排除目标的值永远停在 0、真实选项都带负步费，**于是 bootstrap 恒为 0，价值函数学不到任何东西**，
八个种子全是 0.00。修法是训练、评估、bootstrap 三处共用同一允许集合。
不带 `--meta-safe` 的路径逐位不变。

修好后 Y01(meta-safe) 的 RMST 从 0.213→**0.875**（cheap）、0.044→**0.882**（expensive），
灾难从约 25 万降到 428 / 1002。于是路由效应变成：
Y01→Y11 的 safe→lava 差是 cheap **+0.023**、expensive **−0.033**——**几乎没有放大**。
原报的 +0.502 / +0.688 作废。`Y00→Y10` 那一行仍大，但 Y00 没有技能库，混了太多东西。
**问题一（+0.416）不受影响**，Y10 与 Y11 都不是 meta 臂。

**顺带的强化证据**：昂贵地图加岩浆时，「有技能无路由」0.882 远胜「有路由无技能」0.106。
**不可逆性放大的是复用，不是路由。**

**剂量稳健性（昂贵导航，104 自由格，只跑三臂加一个对照）**：
危险格 0→4 时 Y11 的 RMST/达标率是 0.998/100%、0.996/100%、0.980/99%、0.933/95%、0.883/91%，
Y10 是 0.981/100%、0.345/32%、0.177/16%、0.106/11%、0.104/10%。
Y11−Y10 的 RMST 差：+0.018、**+0.652、+0.803、+0.827、+0.779**。
**一个危险格就打开分叉，且贯穿整个扫描区间**，3% 不是峰值（2% 与 4% 差不到 0.05）。
`QRM < Y10 < Y11` 在每个非零剂量上单调成立——这正是第三十九步要验证的阶梯。

另修正：第三十八步半初稿把危险格个数写成 2 和 8（那是 8% 试跑的数），实际是 1 和 3，已更正。

## 第四十步：对迁移前审计的回应

审计五项全部独立核实，五项都成立。

**一、QRM 的崩溃是步费造成的，不是结构造成的。** 逐位复现：保留原终奖、只把步费从 −0.01 改成 0，
book 22k/24k/24k、book-and-quill 92k/94k/126k（原协议全删失）。
我第一次复核时自己踩了坑：`reward_success = reward_ratio × cost × maxlen`，
把 `--cost` 归零会连终奖一起归零，必须同时 `--reward-success` 保留原值。

按建议跑完网格：3 任务 × 2 导航 × 2 风险 × **2 协议** × 3 臂 × 6 地图 × 15 种子 = **6,480 次**，
逐运行 JSONL 归档（命令、源码与二进制哈希、全部检查点）。

- **推翻**：三臂阶梯不是协议不变的。原协议下八格全是 `Y11 < Y10 < QRM`，
  零步费下八格全是 `Y11 < QRM < Y10`。**QRM 与 Y10 的相对位置完全由步费决定，四个条件无一例外翻转。**
  昂贵+岩浆下 QRM 从 RMST 997k/达标 1% 变成 42k/100%。
- **站住**：**Y11 优于 Y10 在全部八格、两种协议下都成立**，配对区间都不含零。
- **风险 × 复用的交互站住但缩水**：昂贵条件 orig +0.797 [0.649, 0.924] → nostep **+0.248 [0.104, 0.412]**；
  便宜条件 +0.027 → +0.001。

**二、指标更正。** `run_craftworld_lava.py` 与 `run_dose.py` 把成功率 AUC 写成 RMST。
真 RMST = E[min(T,τ)]，单位是步、**越小越好**。两者都保留并各自定义。

**三、收回的表述**：(1)「QRM 没有机制把安全经验跨状态搬运」——CRM 本来就每步做多状态反事实更新，
它缺的是**共享的目标策略参数**；(2)「QRM→Y10→Y11 阶梯」不能作第三十九步前提；
(3)「本批删失 0%」过宽（Step 36 的 Y00 4.2%、Y01 1.4%；Step 38 的 Y00 昂贵条件 22%；
Y10/Y11/QRM 为 0，所以 3.20× 与 6.28× 不受影响）；(4)「3% 不是峰值」不准确，
+0.827 确是已扫五点最大值，准确说法是 2–4 档相差不到 0.05。

**四、已修代码缺陷**：`event_cost` 把「一步即可触发」记为距离 0，应为 1；
独立 BFS 核验 25/25 格恰好差 1。修正后 C_g 组均值 3.1 / 6.6 / 8.4，排序与结论不变，
合并相关 C_g +0.756、N_g +0.078、C_reuse +0.214。措辞收紧为「当前重复度指标无法解释跨组结果」，
不再说「重复次数几乎不贡献」。

**五、几何与剂量的表述收紧**：safe/lava 物理邻接完全相同，但 36 对里 11 对、270 个起点里 63 个
安全最短路变长；相邻剂量的 72 对里 70 对起点不同、52 对时域不同，剂量扫描是多因素稳健性证据。

**六、回归测试**：新增 `tests/test_step38_mechanisms.py` 覆盖 `perstate`、`meta-safe`
（含 bootstrap 那个缺陷）、safe/lava 配对、C_g 差一、步费陷阱与 RMST 方向。现共 69 项，全通过。

**未补**：第三十八步与剂量扫描的逐种子原始数据已丢失，不补造。从本步起新运行一律 JSONL 逐行归档。

## 第三十九步接口规格：读作者代码得出的五项事实

写在 `STEP39_ARM_SPEC.md`。**读代码得出，尚未运行作者环境。**

1. **作者的默认协议就是我们的 `nostep`。** `05–08-cw-frl-*-exploit` 的
   `pseudoreward_after_step = 0.0`（技能层与 meta 层皆是），`condition_satisfied = 1.0`，
   `deadend = 0.0`。**没有每步惩罚。** 第四十步网格里这一列的排序是 `Y11 < QRM < Y10`。
   所以本地 QRM 惨败的那个协议**不是作者用的协议**。
2. **作者的 CRM 跨自动机状态完全共享参数**：`MinigridCRMDQN` 把状态 one-hot 与卷积嵌入拼接后
   过同一个 MLP。我们的表格 CRM 每个任务状态一套独立表项，完全不共享。
   所以「索引读法没有跨状态共享机制」对作者实现从一开始就不成立，本地 QRM 是更弱的实现。
3. **没有表格版 cross-product**：CRM 与 DQRM 都是 torch 的。原生 QRM 臂必然是神经的，
   与表格 HRL 臂不是同一量级的计算，要么三臂都做成神经的，要么明确声明是跨实现比较。
4. **技能共享比「一目标一表」更宽**：`FormulaBankTabular._q_functions` 按公式条件索引，
   且 `get_q_function` 先走 `FormulaTree.get_root()`，被蕴含的公式共用根节点的 Q 表。
   作者的 HRL 本来就是我们的 Y11。
5. **技能更新是 intra-option 的**，每步更新 `_get_subgoals_to_update()` 采样出的一批子目标，
   与当前在跑哪个 option 无关。Y10 必须保留这条，否则同时改了两件事。

**Y10 没有原生对应，必须自己写**，改动至少覆盖索引、读取（`get_root` 是否保留要声明）、
更新范围、两个计数器、导出导入五处；`_meta_q_functions` 不动（它已按自动机状态索引，是路由）。
改完须先证「只剩预期差异」：把自动机状态折叠成 1 时 Y10 应与 Y11 逐位相同。

safe/lava 配对**只翻 `use_lava` 不够**，要逐项核对物体个数（`max_objs_per_class` 是上限，
生成器 1–2 间采样）、物体/起点/危险格坐标与随机数调用顺序。

## 第三十九步第一阶段：原生环境在本机跑通，不需要换 Linux

建了第四个环境 `.venv-hrm`（Python 3.10.21），完整记录在 `ENVIRONMENT.md`。

**已验证**：三个任务 × 两种网格都能 `gym.make`；子自动机数 3/5/6 与用桩件做的结构审计逐项一致；
随机策略 5000 步下 open_plan 无岩浆 0 次进拒绝态、four_rooms 带岩浆 49 幕全死于岩浆；
`ihsa-hrl`（= Y11）跑满 400 幕、`ihsa-crm`（= QRM）跑满 120 幕均正常结束；
`MinigridCRMDQN` 前向正常、144,147 参数（单一共享网络，再次确认 CRM 跨状态共享参数）。

**与作者 requirements 的五处偏离**（都是必须的）：torch 2.14 代替 `1.10.0+cpu`（ARM 无轮子，API 未漂移）；
**numpy 1.23.5，必须 < 1.24**（源码 5 处用 `np.int`/`np.bool`，1.24 移除了别名；
1.21 在 3.10/ARM 无轮子）；pygame 2.6.1 代替 1.9.6（ARM 不能构建，且只被 WaterWorld 用到）；
matplotlib 3.10；gym 保持 0.15.3（能在 3.10 上从源码构建）。装 numpy 必须放最后，torch 会把它顶到 2.x。

**一行补丁** `docs/environment/hrm_learning_py310_compat.patch`：
`src/ilasp/ilasp_common.py` 的 `os.environ.putenv(...)` → `os.environ[...] = ...`，
`putenv` 在 Python 3.9 被移除；启动器无条件调用它，即使 `handcrafted` 模式用不到 ILASP。

**另记**：`four_rooms` 的尺寸走 `size` 键（须为 ≥11 的奇数），只给 `width`/`height` 会报错。

移植清单与接口规格都已按此更新：**五步全部本机可做**，只有第 5 步的长训练是速度问题而非可行性问题。
下一步是实现 Y10 并做三项等价性检查。

## 第三十九步第二阶段：Y10 已在作者代码里实现并通过等价性检查

**实现方式比原计划省**：不改 bank 内部的键，而是**每个自动机状态各持一整份 bank**，
从父类建好的那份惰性深拷贝。蕴含共享、更新范围、两个计数器、导出格式因此自然分开，
只需一个新类 `ihsa_hrl_tabular_perstate_algorithm.py` 加 `run_algorithm.py` 注册一行。
键取自 `_choose_action` 的 `hierarchy_state`（唯一能拿到它的地方，且同一轮循环里紧接着被
`_update_q_functions` 使用，所以正是动作发生的状态）。算法名 `ihsa-hrl-perstate`。

**三项等价性检查全部通过**：
(1) 键钉成常数（`perstate_collapse: true`）后与 Y11 **逐位相同**，150 幕日志 md5 一致，
不钉死则不同；(2) 每步采样的子目标个数两臂都恒为 4（总调用次数不同只反映各自走了多少步，
那是结果不是预算——初版判据把两者混了，已改正）；(3) 伪奖励取值集合相同，
都是 `{(0.0,False),(0.0,True),(1.0,True)}`，**再次确认作者配置没有每步惩罚**。

150 幕里 Y11 用 10 份 bank（每任务一份），Y10 建了 42 份（每任务每自动机状态一份）。

**已知缺口**：`state_format` 会改变路由——`full_obs` 下 `ihsa-hrl` 走的是 DQN 版。
上述配对用的是 `tabular`，而作者发表用的是 `full_obs`，**神经版的 Y10 尚未实现**。

归档：`docs/environment/y10_perstate_arm.py`、`docs/environment/y10_equivalence_checks.py`、
`docs/environment/hrm_learning_py310_compat.patch`（现含 `putenv` 修复与算法注册两处）。

## 第三十九步第三阶段：神经版 Y10 也已实现并通过检查

作者发表用的是 `state_format: full_obs`，会路由到 `IHSAAlgorithmHRLDQN`，所以补了
`ihsa_hrl_dqn_perstate_algorithm.py`；算法名仍是 `ihsa-hrl-perstate`，按 `state_format` 自动分派。

**一个表格版没有的设计选择，必须声明**：DQN 的 bank 自带回放缓冲，按状态分 bank 会连带分开缓冲。
作者的 `er_start_size = 100000`，一个只占五分之一访问量的状态要五十万步才开始学，
稀少状态可能永远开不了学——那样 Y10 会因记账阈值而输，不是因为不能共享。
**默认让各状态 bank 共用同一个缓冲**：数据流与启动阈值相同，只有网络/目标网络/优化器分开，
恰好隔离被检验的因素，也是这条臂最强的公平形式。`perstate_split_buffer: true` 切到字面读法。
各状态 bank 仍只在自己状态活跃时被更新，两种读法皆然——这是「不共享」的含义，不是混淆。

**四项检查通过**：折叠后与 Y11 逐位相同（60 幕 md5 一致）、每步采样恒为 4、
伪奖励取值相同、梯度更新确实发生（7,112 vs 6,969，差异只来自轨迹长度）。
10 份按状态分开的 bank 共用 **1 个**缓冲对象。检查时须把 `er_start_size` 临时调到 200，
否则短跑里一次梯度更新都不会发生，检查会空过。

归档：`docs/environment/y10_perstate_arm_dqn.py`、`y10_equivalence_checks_dqn.py`，
`hrm_learning_py310_compat.patch` 已刷新（含 putenv 修复与两处算法注册）。

## 第三十九步：原生网格完成，H1 复现且随深度增大

作者代码、作者协议（`pseudoreward_after_step = 0.0`）、CraftWorld four_rooms 13×13。
3 任务 × 2 风险 × 2 臂 × 3 种子 × 10 个任务实例，每次 10,000 幕，逐运行 JSONL 归档。

**风险对照是构造的**：直接翻 `use_lava` 会挪动物体（危险格占位、挤走原物体、消耗随机数，
seed 0 上 workbench 与 squid 都变了），所以两格共用同一张含岩浆网格，
安全格中和层级里的拒绝态（`neutralize_deadends`）。验证：网格/起点/可观测集/可能观测集逐项相同；
随机策略 lava 格 92 幕全死、safe 格踩 226 次不终止。

**H1 复现，且随深度增大**（无岩浆，末段贪心成功率，配对自助按任务实例）：

| 任务 | 深度 | Y11 | Y10 | 差 |
|---|---|---|---|---|
| book | 5 | 1.000 | 0.911 | +0.089 [+0.027, +0.172] |
| book-and-quill | 8 | 1.000 | 0.153 | **+0.847 [+0.743, +0.929]** |
| cake | 9 | 1.000 | 0.088 | **+0.912 [+0.841, +0.969]** |

Y11 三个任务都是 1.000，Y10 随深度从 0.911 掉到 0.088。比本地更强——
本地的复用效应要靠不可逆性才显著放大，这里光是任务变深就够了。

**H2 未测出，是地板不是反向证据**：有岩浆时 book-and-quill 与 cake 上两臂都是 0.000，
所以「差的差」算出 −0.847/−0.912 只反映安全格分开而岩浆格两臂都没离地。
一万幕对带岩浆的深任务远远不够（作者用 20 万幕）。

**四条偏离逐条声明**：(1) 表格而非发表用的 `full_obs`——本机实测神经版慢 110 倍，
20 万幕要 111 小时/次，单机不可行（神经版 Y10 已实现并过同样的等价性检查，只是没预算跑）；
(2) 10,000 幕而非 200,000——无岩浆下 Y11 已饱和，结论不受影响，有岩浆下不够；
(3) 只跑作者协议——book 上 stepcost 的四个格子两臂全是 0.000，继续只会得到更多零，
协议敏感性已由本地 6,480 次网格回答，那 12 次仍在归档里；(4) QRM 缺席，作者没有表格版 cross-product。

本机性能实测：表格臂无岩浆约 0.20 秒/幕，神经臂 0.5 幕/秒。
