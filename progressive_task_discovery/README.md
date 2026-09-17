# Progressive task discovery: mechanism pilot

新增的分支任务 Stage 7 已完成，见 [Stage 7 报告](</Users/yuhang/Downloads/why TL/AUTO/progressive_task_discovery/stage7_branching/REPORT.md>)及[复现说明](</Users/yuhang/Downloads/why TL/AUTO/progressive_task_discovery/stage7_branching/README.md>)。下面保留原六阶段项目说明。

已完成六轮实验。先读 [REPORT.md](</Users/yuhang/Downloads/why TL/AUTO/progressive_task_discovery/REPORT.md>)；主结论是长任务中“部分顺序发现后立即复用导航技能”有效，单独 frontier epsilon 调度没有稳定优势，短任务中普通回放经常更快。

CPU-only C++17 tabular simulator, Python analysis, no GPU dependency. The project is independent of the older `tl_sequence_pilot` directory. No robot, PPO, learned perception, arbitrary automaton inference, or claimed literature novelty is included.

## Run and verify

From this directory, using the existing system Python with NumPy and Matplotlib:

```bash
make test
make smoke
python3 src/verify_reproduction.py
```

`make smoke` prints one complete JSON result. `verify_reproduction.py` replays representative saved configurations and compares exact trajectory hashes, Q-table hashes and evaluation curves. The full suite has 10,008 runs, but each is a tiny tabular experiment; no neural model is trained.

To launch a new suite, copy one of `docs/stage1.json` through `docs/stage6.json`, change its `name`, and run:

```bash
python3 src/run_suite.py docs/your_new_spec.json --workers 2
python3 src/analyze.py your_new_name --baseline count_replay --audit
```

The runner refuses to overwrite an existing result directory. Every stage retains its original specification, source snapshot, manifest and JSONL output. Development and held-out seeds are deliberately separate. Do not tune on the held-out sets.

To regenerate current figures and report from stored data:

```bash
python3 src/plot_results.py
python3 src/write_report.py
```

## Data and algorithm map

* `docs/protocol.md`: initial assumptions, task semantics, controls and accounting.
* `docs/decision_log.md`: observations and the next design recorded before each stage.
* `docs/frozen_selection.json`: configuration selection before held-out evaluation.
* `src/pilot.cpp`: environment, learners, exact evaluation, internal semantic checks.
* `src/run_suite.py`: two-worker experiment orchestration and source provenance.
* `src/analyze.py`: capped outcomes, bootstrap comparisons, optimal feasibility DP.
* `tests/test_pilot.py`: independent backward policy evaluation and interface checks.
* `results/stage*/raw.jsonl`: full per-run data, including final policies.
* `results/stage*/summary.csv`: group outcomes and censoring/stability indicators.
* `results/stage*/paired_vs_*.csv`: seed-paired comparisons; ratio below 1 favors the variant.
* `results/validation_audit.json`: all-run accounting and independent policy checks.
* `figures/`: PNG and SVG charts.

The effective `goal_reuse` learner maintains a progress count, a discovered `next_event[stage]` list and one goal-reaching Q table per observable label. It trains these tables from actual environment transitions, then immediately chooses the appropriate table after a next-event label is confirmed. `delayed_goal` trains exactly the same bank but cannot use it until the first complete task success. `known_goal` trains only goal labels already identified in the prefix. `count_replay` performs eight additional ordinary experience-replay updates per physical step, matching the nine updates of the full goal bank. None of these unknown-task agents receives the hidden target sequence.

The task family assumes an ordered chain, reliable progress feedback, observable labels, and ignore-wrong-events semantics. A progress counter is sufficient memory here. This is prefix identification and skill reuse, not general automaton learning. Fixed physical dynamics make off-policy event-goal relabeling possible. Total training interactions include every prefix traversal; dynamic-programming evaluation is read-only, and its expected rollout length is reported separately.

## Reproducibility details

Observed runtime: Apple system `clang++`, Python 3.9, NumPy 1.24.2, Matplotlib 3.8.2. Runtime versions and hashes are saved in the result metadata. Task and behavior randomness are separate; replay uses independent deterministic draws. Greedy evaluation breaks ties by lowest action index. Q-learning bootstraps at time-limit truncation and not at true success; local skills terminate their own Bellman target on goal entry. Constant alpha=.3 and gamma=.98 are used in the held-out studies.

First90 is the first evaluation checkpoint reaching success probability .9. Unsolved runs are right-censored; the analysis reports a **budget-capped score**, not an uncensored expected solving time. A second metric requires two adjacent successful checkpoints. Raw outcomes, including failures and early development conditions with theoretically unattainable thresholds, are retained.
