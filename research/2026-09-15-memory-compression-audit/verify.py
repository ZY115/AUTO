"""Independent archive/metric audit; no changes to the experiment or its data."""
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "AUTO"
OUT = Path(__file__).resolve().parent
source = REPO / "tools/memory_compression_pilot.py"
blob = (REPO / "results/runs/memory_compression_pilot_5k.jsonl.gz").read_bytes()
raw = gzip.decompress(blob)  # Includes CRC validation.
rows = [json.loads(line) for line in raw.splitlines()]
sha = lambda value: hashlib.sha256(value).hexdigest()
assert sha(blob) == "f701dc69e2c24c5626957f2524ef7c0b40ca8adbfcd1b912b8f3e4cac25d283a"
assert sha(raw) == "1b6b88ace392d022099ee6a17fb56a720b2eb772f3965e479f5c0afdc44bec8b"
code_hash = sha(source.read_bytes())
R1, R2 = "round1_known_compression", "round2_learned_compression"
methods = {R1: ("full_history", "correct_automaton", "correct_handwritten", "progress_count"),
           R2: ("full_history", "immediate_use", "delayed_use", "direct_rule")}
by = {(r["phase"], r["method"], r["n_parts"], r["seed"]): r for r in rows}
expected = {(p, m, n, s) for p in methods for m in methods[p]
            for n in (3, 4, 5) for s in range(400)}
assert len(rows) == len(by) == 9600 and set(by) == expected
points = [0] + list(range(1, 201)) + list(range(205, 1001, 5)) + list(range(1020, 5001, 20))
for r in rows:
    assert r["code_sha256"] == code_hash and r["episodes"] == 5000 and r["epsilon"] == .1
    assert [x[0] for x in r["curve"]] == points
    assert all(0 <= x[1] <= 1 for x in r["curve"])
    last_failure = max((i for i, x in enumerate(r["curve"]) if x[1] < .95), default=-1)
    t95 = None if last_failure == len(points) - 1 else points[last_failure + 1]
    assert r["t95"] == t95
    assert r["q_updates"] == (0 if r["method"] == "direct_rule" else 5000)
    assert r["final_exact_accuracy"] == (.5 if r["method"] == "progress_count" else 1)
    if "candidate_set_final" in r:
        assert r["candidate_set_final"] == ["A<B->X"]
        assert r["identified_episode"] <= 200

def ci(values):
    a = np.array(values, dtype=float)
    rng = np.random.default_rng(9152026)
    draws = a[rng.integers(0, len(a), size=(10000, len(a)))].mean(axis=1)
    return [float(a.mean()), *[float(x) for x in np.quantile(draws, [.025, .975])]]

summary = {"rows": len(rows), "paired_bundles": 1200, "gzip_sha256": sha(blob),
           "raw_sha256": sha(raw), "runner_sha256": code_hash, "results": {}}
for n in (3, 4, 5):
    data = {m: [] for m in ("full", "oracle", "immediate", "delayed", "direct", "ident",
                            "pre_greedy_gain", "realized_reward_gain", "extra_rebuild_evals")}
    post_ident_mismatches = 0
    before_ident_acc_max = 0
    for s in range(400):
        f, a, h, count = (by[R1, m, n, s] for m in methods[R1])
        f2, im, de, di = (by[R2, m, n, s] for m in methods[R2])
        bundle = [f, a, h, count, f2, im, de, di]
        assert len({r["trace_sha256"] for r in bundle}) == 1
        assert a["curve"] == h["curve"] and a["environment_reward_sum"] == h["environment_reward_sum"]
        assert f["curve"] == f2["curve"]
        assert all(x[1] == .5 for x in count["curve"])
        assert im["t95"] == de["t95"]
        assert im["identified_episode"] == de["identified_episode"] == di["identified_episode"]
        assert [[v[:2] for v in r["structure_curve"]] for r in (im, de, di)].count(
            [v[:2] for v in im["structure_curve"]]) == 3
        I = im["identified_episode"]
        assert im["t95"] == max(a["t95"], I)
        pg = sum(im["curve"][t][1] - de["curve"][t][1] for t in range(I))
        post_ident_mismatches += sum(x != y for x, y in zip(im["curve"], de["curve"]) if x[0] >= I)
        before_ident_acc_max = max(before_ident_acc_max, max(x[1] for x in im["curve"] if x[0] < I))
        for label, value in (("full", f["t95"]), ("oracle", a["t95"]), ("immediate", im["t95"]),
                             ("delayed", de["t95"]), ("direct", di["t95"]), ("ident", I),
                             ("pre_greedy_gain", pg),
                             ("realized_reward_gain", im["environment_reward_sum"] - de["environment_reward_sum"]),
                             ("extra_rebuild_evals", im["repartition_rule_evals"] - de["repartition_rule_evals"])):
            data[label].append(value)
    # Independent analytic expectation for the full-history baseline: a state
    # becomes correct upon its first positive reward; unknown Q entries tie.
    H = math.factorial(n)
    K = math.ceil(.9 * H)
    expected_full_exact_time = sum(2 * H / (H - i) for i in range(K))
    summary["results"][n] = {
        "metrics_mean_ci": {k: ci(v) for k, v in data.items()},
        "mean_T95_ratio_full_oracle": statistics.mean(data["full"]) / statistics.mean(data["oracle"]),
        "paired_T95_saving": ci([x-y for x,y in zip(data["full"], data["oracle"])]),
        "pre_gain_epsilon_policy_expected": ci([.9*x for x in data["pre_greedy_gain"]]),
        "post_ident_curve_mismatches": post_ident_mismatches,
        "T95_equals_max_oracle_T95_and_identification_all_400": True,
        "largest_identification_episode": max(data["ident"]),
        "largest_immediate_accuracy_before_identification": before_ident_acc_max,
        "analytic_mean_exact_T95_full": expected_full_exact_time,
        "analytic_mean_exact_T95_oracle": 6,
    }

# Re-run nine complete bundles from current source, comparing every archived
# semantic field, not merely its summary curve or stopping time.
spec = importlib.util.spec_from_file_location("pilot_audit_import", source)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
replayed = 0
for n in (3, 4, 5):
    for s in (0, 17, 399):
        events, trace_hash = mod.make_trace(n, s, 5000)
        for replay in mod.run_seed_bundle((n, s, 5000, .1, ("round1", "round2"), code_hash)):
            stored = by[replay["phase"], replay["method"], n, s]
            assert all(stored[k] == v for k, v in replay.items() if k != "cpu_seconds"), (n, s, replay["method"])
            replayed += 1
        # Independent positive-reward coverage learner for known representations.
        # Does not call original Q table, state mapping, reward, or evaluation.
        for compressed, method in ((False, "full_history"), (True, "correct_automaton")):
            learned = set()
            curve = [[0, .5]]
            reward_total = 0
            for t, e in enumerate(events, 1):
                first = next(x for x in e.history if x in "AB")
                target = int(first == "B")
                key = first if compressed else e.history
                action = int(e.choice_u >= .5) if e.explore_u < .1 or key not in learned else target
                reward_total += int(action == target)
                if action == target:
                    learned.add(key)
                if t in points:
                    curve.append([t, .5 + .5*len(learned)/(2 if compressed else math.factorial(n))])
            stored = by[R1, method, n, s]
            assert np.allclose(curve, stored["curve"], rtol=0, atol=1e-15)
            assert reward_total == stored["environment_reward_sum"]

# Independently enumerate disagreement fractions between distinct candidate
# rules. Before identification, a safely learned positive group must be a
# unanimous group; on disagreements the Q learner cannot yet have a positive.
summary["minimal_candidate_disagreement"] = {}
for n in (3, 4, 5):
    labels = tuple(chr(65+i) for i in range(n))
    histories = list(itertools.permutations(labels))
    rules = list(itertools.permutations(labels, 2))
    predictions = [[int(h.index(a) > h.index(b)) for h in histories] for a,b in rules]
    minimum = min(sum(a != b for a,b in zip(x,y))/len(histories)
                  for x,y in itertools.combinations(predictions, 2))
    summary["minimal_candidate_disagreement"][n] = minimum
summary["replayed_rows_all_semantic_fields_equal"] = replayed
summary["independent_known_compression_replays"] = 18
(OUT / "verification.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
