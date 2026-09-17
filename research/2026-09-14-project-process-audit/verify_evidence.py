"""Read-only rechecks for the process audit. Prints JSON; does not alter runs."""
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    out = {"scope": "Two archived pilot replays; summaries of existing runs, no new training grid."}
    pilot = ROOT / "progressive_task_discovery"
    rows = read_rows(pilot / "results/stage6_early_use/raw.jsonl")
    out["early_online_loop"] = {}
    for method in ("goal_reuse", "delayed_goal"):
        group = [r for r in rows if r["method"] == method and r["k"] == 16
                 and r["condition"] == "noisy"]
        old = next(r for r in group if r["seed"] == 200)
        new = json.loads(subprocess.check_output(
            [str(pilot / "pilot")] + old["command"][1:], text=True))
        checks = {k: old[k] == new[k] for k in ("trace_hash", "q_hash", "checkpoints")}
        assert all(checks.values())
        out["early_online_loop"][method] = {
            "n": len(group), "mean_first90": statistics.mean(r["first90"] for r in group),
            "seed_200_replay": checks, "seed_200_discovery_steps": old["discovery_steps"],
            "seed_200_first_success": old["first_train_success"],
        }

    collections = {}
    out["run_files"] = {}
    for name in ("local_lava_10m.jsonl", "local_sampling_intervention.jsonl"):
        path = ROOT / "AUTO/results/runs" / name
        runs = read_rows(path)
        values = {}
        for run in runs:
            assert run["ok"] and run["stats"]["train_env_steps"] == 10_000_000
            for map_id, curve in run["curves"].items():
                previous_step, previous_value, area = 0, 0.0, 0.0
                for point in curve:
                    step = point[3]
                    assert previous_step <= step <= 10_000_000
                    area += (step - previous_step) * previous_value
                    previous_step, previous_value = step, point[1]
                area += (10_000_000 - previous_step) * previous_value
                values[run["arm"], run["seed"], map_id] = area / 10_000_000
        collections[name] = values
        out["run_files"][name] = {"runs": len(runs), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    old, new = (collections[n] for n in ("local_lava_10m.jsonl", "local_sampling_intervention.jsonl"))
    maps = sorted({key[2] for key in old})
    indices = np.random.default_rng(42).integers(0, len(maps), (50_000, len(maps)))
    out["step_weighted_auc"] = {}
    comparisons = [("old_Y11_minus_Cfull", old, "Y11", old, "Cfull"),
                   ("shared_minus_independent", new, "CfullShared", new, "CfullInd"),
                   ("new_Y11_minus_CfullInd", new, "Y11", new, "CfullInd"),
                   ("new_minus_old_Y11", new, "Y11", old, "Y11")]
    for label, left, la, right, ra in comparisons:
        differences = np.array([np.mean([left[la, s, m] - right[ra, s, m]
                                        for s in range(3)]) for m in maps])
        out["step_weighted_auc"][label] = {
            "mean": float(differences.mean()),
            "map_block_bootstrap_ci": np.quantile(differences[indices].mean(1), [.025, .975]).tolist(),
            "positive_maps": int(sum(differences > 0)), "map_differences": differences.tolist(),
        }
    out["statistical_scope"] = (
        "Descriptive exploratory reanalysis. Map-block bootstrap conditions on these three seeds; "
        "maps share a curriculum within a run. Not a new independent confirmation or a noise-floor estimate."
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
