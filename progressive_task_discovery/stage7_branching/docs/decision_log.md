# Stage 7 decisions

## Before first run
See protocol.md. Test balanced forced initial histories, deterministic transitions and exact labels/progress. Preserve previous six-stage project.

## Initial development → shared-bridge refinement
1152 runs completed. All count variants failed the balanced90% criterion. Pooled AUC: history replay .8785, hand monitor replay .9299, immediate skills .9361, delayed skills .7681 at epsilon .005. History/learned and dictionary/automaton skill implementations have identical behavior. However, the original task's branch decision immediately follows the cue, so a one-event memory could suffice.

Before confirmation, strengthen the history witness by inserting a common event: A→E versus B→E must return to the same junction at count2 and last-event E, then require different next goals C/D. Add last_event_replay with the same compute budget. Repeat epsilon calibration on these shared-bridge tasks (bridge=1), same development seeds. Keep initial outputs intact. The final hand monitor remembers the cue through the common bridge and forgets it only after the branch-specific goal; this is prior knowledge of the family, not an inferred abstraction. A longer shared bridge (bridge3) may be used as a frozen-parameter structural stress test, without adding feedback or physical noise.

## Metric interpretation
Raw next_goal fields describe the passive learned transition cache even for replay-only agents. In final comparative tables these fields will be reported only for skill methods; a replay-only policy has no explicit next-goal action. Similarly wrong-binding rate is N/A when binding_steps=0. Task-state accuracy is alignment/purity of observed feature partitions, not the accuracy of arbitrary state integers or a minimality claim. The best single-branch evaluation is a diagnostic of why balanced starts matter, not a separately trained freely choosing policy.

## Shared-bridge development → frozen confirmation
1296 additional calibration runs finished. At bridge1, pooled AUC: hand replay .92431, history/learned replay .88003, immediate/history skills .93663, delayed skills .76736. Both count and last-event baselines fail90% for all configurations. Select epsilon by pooled AUC: .005 for count, last-event, hand, history/learned replay, delayed and oracle skills; .02 for immediate/history skills. The immediate-versus-delayed online comparison therefore includes their independently tuned epsilon settings; an equal-epsilon causal check is specified below.

Freeze these values. Main confirmation: seeds400..431, k4/8/12, bridge1, H16k and24k, budget100k/checkpoint1k. Structural stress test: new seeds500..531, bridge3, k8/12, H20k, same parameters and budget. No feedback or action noise. Shared-bridge refinement increases the feature table capacity from64 to128; old snapshot hashes retain the old padding, so source snapshots are authoritative for whole-table hash reproduction.

Add a matched-epsilon comparison on the main held-out tasks: immediate and delayed both use epsilon .005 (the delayed baseline's selected value), same skill code, same relabeling and six updates per interaction. This prevents attributing a benefit of different epsilon choices to the timing of skill use. These online methods still experience different trajectories; shared/frozen physical data is deferred to Stage8.

## Main results → same-map no-ambiguity control
Main held-out runs show reliable history/skill learners, while count and last-event learners do not solve both branches. To isolate task ambiguity from the new corridor layout, retain identical map, starts, horizons, seeds and shortest-route distances; change only B's post-cue required event sequence to match A's. Count is then sufficient. Run five frozen methods on the same32 held-out seeds and lengths/horizons. This is a post-result mechanism control, not an additional hyperparameter search. Its feature-state identities are verified independently.

## Final audit and stopping decision

All six batches completed: 6,096 main runs and 560,640,000 training steps. Count and last-event solve every same-map nonambiguous instance but reach no balanced90% threshold in the ambiguous held-out instances. The 1,280 exact representation pairs pass trajectory, Q-table and checkpoint comparisons. At k12 immediate reuse costs 12–24% less than the privileged hand-monitor replay baseline across the two main horizons; short-task results are mixed or worse. Matched epsilon retains the large immediate-versus-delayed effect. Three shared events retain the long-task effect.

Added per-(prefix,goal) skill diagnostics and replayed 128 k12 immediate/delayed configurations, verifying exact behavior against their original results. These are diagnostic duplicates, not extra independent evidence. Eight semantic tests and the full accounting/edge/policy audit pass. The three final figures were visually inspected.

Stage7 is complete. The data support preserving sufficient history and immediate reuse, but the learned prefix representation equals a full-history dictionary. H3 remains untested. Stop this batch rather than add more seeds to already resolved controls. Stage8's frozen shared-skill-bank intervention is specified in NEXT_STAGE8.md; it separates direct scheduling benefits from changes in online skill-training data. No noisy-feedback, context-dependent, neural or robot experiments were executed.
