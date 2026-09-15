#!/usr/bin/env python3
"""Run the controlled history-compression pilot described in the project notes.

The environment is a terminal contextual bandit.  A uniformly sampled
permutation of A..E is observed one symbol at a time; after the workbench is
cleared, the agent chooses X or Y.  X is correct exactly when A preceded B.

Round 1 compares complete histories with two independent implementations of
the known sufficient state.  Round 2 learns an ordered-pair rule version space
and compares immediate safe sharing, sharing only after identification, and a
direct version-space policy.  The implementation uses only the Python standard
library so the raw result is not coupled to the CraftWorld environment.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import random
import socket
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


X, Y = 0, 1
ACTION_NAMES = ("X", "Y")
SCHEMA_VERSION = 1
TRUE_RULE = ("A", "B")
ROUND1_METHODS = ("full_history", "correct_automaton", "correct_handwritten", "progress_count")
ROUND2_METHODS = ("full_history", "immediate_use", "delayed_use", "direct_rule")

History = Tuple[str, ...]
Rule = Tuple[str, str]
Cell = List[float]
StateKey = object


def parts_for(n_parts: int) -> Tuple[str, ...]:
    if n_parts < 3 or n_parts > 8:
        raise ValueError("n_parts must be between 3 and 8")
    return tuple(chr(ord("A") + i) for i in range(n_parts))


def enumerate_histories(n_parts: int) -> Tuple[History, ...]:
    return tuple(itertools.permutations(parts_for(n_parts)))


def correct_action(history: History, true_rule: Rule = TRUE_RULE) -> int:
    pos = {symbol: i for i, symbol in enumerate(history)}
    return X if pos[true_rule[0]] < pos[true_rule[1]] else Y


def reward_for(history: History, action: int) -> int:
    return int(action == correct_action(history))


def all_candidate_rules(parts: Sequence[str]) -> Tuple[Rule, ...]:
    return tuple((a, b) for a in parts for b in parts if a != b)


def predict_action(history: History, rule: Rule) -> int:
    return X if history.index(rule[0]) < history.index(rule[1]) else Y


def prediction_signature(history: History, candidates: Sequence[Rule]) -> Tuple[int, ...]:
    return tuple(predict_action(history, rule) for rule in candidates)


def automaton_state(history: History) -> int:
    """Explicit three-state DFA; terminal output 0 means A-first, 1 B-first."""
    state = "unseen"
    for symbol in history:
        if state == "unseen" and symbol == "A":
            state = "a_first"
        elif state == "unseen" and symbol == "B":
            state = "b_first"
    if state == "unseen":
        raise ValueError("history contains neither A nor B")
    return 0 if state == "a_first" else 1


def handwritten_state(history: History) -> int:
    """Independent program implementing the same sufficient statistic."""
    first_relevant: Optional[str] = None
    for symbol in history:
        if first_relevant is None and symbol in ("A", "B"):
            first_relevant = symbol
    if first_relevant is None:
        raise ValueError("history contains neither A nor B")
    return int(first_relevant == "B")


def full_history_state(history: History) -> History:
    return history


def progress_state(history: History) -> int:
    return len(history)


STATE_FUNCTIONS: Mapping[str, Callable[[History], StateKey]] = {
    "full_history": full_history_state,
    "correct_automaton": automaton_state,
    "correct_handwritten": handwritten_state,
    "progress_count": progress_state,
}


def evaluation_schedule(episodes: int) -> Tuple[int, ...]:
    """Dense near zero, then progressively cheaper exact enumeration."""
    points = {0, episodes}
    points.update(range(1, min(episodes, 200) + 1))
    if episodes > 200:
        points.update(range(205, min(episodes, 1000) + 1, 5))
    if episodes > 1000:
        points.update(range(1020, episodes + 1, 20))
    return tuple(sorted(points))


@dataclass(frozen=True)
class EpisodeEvent:
    history: History
    explore_u: float
    choice_u: float


def make_trace(n_parts: int, seed: int, episodes: int) -> Tuple[List[EpisodeEvent], str]:
    histories = enumerate_histories(n_parts)
    env_rng = random.Random(0xA17E5EED ^ (seed * 0x9E3779B1) ^ n_parts)
    action_rng = random.Random(0xC0FFEE11 ^ (seed * 0x85EBCA77) ^ (n_parts << 16))
    digest = hashlib.sha256()
    events: List[EpisodeEvent] = []
    for _ in range(episodes):
        index = env_rng.randrange(len(histories))
        explore_u = action_rng.random()
        choice_u = action_rng.random()
        digest.update(struct.pack("!Idd", index, explore_u, choice_u))
        events.append(EpisodeEvent(histories[index], explore_u, choice_u))
    return events, digest.hexdigest()


class BanditTable:
    """One-step tabular Q-learning with alpha=1/N(state, action)."""

    def __init__(self) -> None:
        # Each action cell is [reward_sum, visit_count].
        self.stats: Dict[StateKey, List[Cell]] = {}

    def _cells(self, state: StateKey) -> List[Cell]:
        return self.stats.setdefault(state, [[0.0, 0.0], [0.0, 0.0]])

    def q_values(self, state: StateKey) -> Tuple[float, float]:
        cells = self.stats.get(state)
        if cells is None:
            return 0.0, 0.0
        return tuple(cell[0] / cell[1] if cell[1] else 0.0 for cell in cells)  # type: ignore[return-value]

    def choose(self, state: StateKey, explore_u: float, choice_u: float, epsilon: float) -> int:
        if explore_u < epsilon:
            return X if choice_u < 0.5 else Y
        qx, qy = self.q_values(state)
        if qx > qy:
            return X
        if qy > qx:
            return Y
        return X if choice_u < 0.5 else Y

    def update(self, state: StateKey, action: int, reward: int) -> None:
        cell = self._cells(state)[action]
        cell[0] += reward
        cell[1] += 1


def expected_greedy_reward(qx: float, qy: float, history: History) -> float:
    right = correct_action(history)
    if qx > qy:
        return float(right == X)
    if qy > qx:
        return float(right == Y)
    # Evaluation randomises exact ties, without updating or retaining memory.
    return 0.5


def exact_table_accuracy(
    histories: Sequence[History], table: BanditTable, state_fn: Callable[[History], StateKey]
) -> float:
    values = []
    for history in histories:
        values.append(expected_greedy_reward(*table.q_values(state_fn(history)), history))
    return sum(values) / len(values)


def time_to_threshold(curve: Sequence[Sequence[float]], threshold: float = 0.95) -> Optional[int]:
    """First logged point at/above threshold with no later reversal."""
    for i, point in enumerate(curve):
        if point[1] >= threshold and all(later[1] >= threshold for later in curve[i:]):
            return int(point[0])
    return None


def run_round1_method(
    method: str,
    n_parts: int,
    seed: int,
    events: Sequence[EpisodeEvent],
    trace_sha256: str,
    epsilon: float,
    eval_points: Sequence[int],
    code_sha256: str,
) -> dict:
    histories = enumerate_histories(n_parts)
    state_fn = STATE_FUNCTIONS[method]
    table = BanditTable()
    curve: List[List[float]] = [[0, exact_table_accuracy(histories, table, state_fn)]]
    eval_set = set(eval_points)
    start = time.process_time()
    reward_sum = 0
    for episode, event in enumerate(events, 1):
        state = state_fn(event.history)
        action = table.choose(state, event.explore_u, event.choice_u, epsilon)
        reward = reward_for(event.history, action)
        reward_sum += reward
        table.update(state, action, reward)
        if episode in eval_set:
            curve.append([episode, exact_table_accuracy(histories, table, state_fn)])
    cpu_seconds = time.process_time() - start
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "round1_known_compression",
        "method": method,
        "n_parts": n_parts,
        "n_histories": math.factorial(n_parts),
        "seed": seed,
        "episodes": len(events),
        "epsilon": epsilon,
        "q_update": "sample_average_alpha_1_over_n",
        "q_updates": len(events),
        "environment_reward_sum": reward_sum,
        "trace_sha256": trace_sha256,
        "curve": curve,
        "t95": time_to_threshold(curve),
        "final_exact_accuracy": curve[-1][1],
        "final_state_count": len(table.stats),
        "candidate_checks": 0,
        "repartition_rule_evals": 0,
        "repartition_histories_scanned": 0,
        "cpu_seconds": cpu_seconds,
        "code_sha256": code_sha256,
    }


class LearnedPartitionPolicy:
    """Version-space learner backed by lossless per-history sufficient stats.

    Repartitioning aggregates reward sums and visit counts.  It never replays an
    old transition, so every environment episode still causes exactly one Q
    update.  Because alpha is 1/N, aggregation is algebraically lossless.
    """

    def __init__(self, histories: Sequence[History], parts: Sequence[str], mode: str) -> None:
        if mode not in ("immediate", "delayed"):
            raise ValueError(mode)
        self.histories = tuple(histories)
        self.mode = mode
        self.candidates: Tuple[Rule, ...] = all_candidate_rules(parts)
        self.history_stats: Dict[History, List[Cell]] = {}
        self.partition: Dict[History, StateKey] = {}
        self.group_stats: Dict[StateKey, List[Cell]] = {}
        self.candidate_checks = 0
        self.rule_action_predictions = 0
        self.repartition_rule_evals = 0
        self.repartition_histories_scanned = 0
        self.repartition_stat_cells_scanned = 0
        self.partition_rebuilds = 0
        self.class_merges = 0
        self.identified_episode: Optional[int] = None
        self._rebuild_partition(initial=True)

    @staticmethod
    def _blank_cells() -> List[Cell]:
        return [[0.0, 0.0], [0.0, 0.0]]

    def _key(self, history: History) -> StateKey:
        if self.mode == "delayed" and len(self.candidates) > 1:
            return history
        return prediction_signature(history, self.candidates)

    def _rebuild_partition(self, initial: bool = False) -> None:
        old_count = len(set(self.partition.values())) if self.partition else 0
        partition: Dict[History, StateKey] = {}
        if self.mode == "delayed" and len(self.candidates) > 1:
            for history in self.histories:
                partition[history] = history
        else:
            for history in self.histories:
                partition[history] = prediction_signature(history, self.candidates)
            self.repartition_rule_evals += len(self.histories) * len(self.candidates)
        self.repartition_histories_scanned += len(self.histories)
        grouped: Dict[StateKey, List[Cell]] = {}
        for history, cells in self.history_stats.items():
            target = grouped.setdefault(partition[history], self._blank_cells())
            for action in (X, Y):
                target[action][0] += cells[action][0]
                target[action][1] += cells[action][1]
                self.repartition_stat_cells_scanned += 1
        self.partition = partition
        self.group_stats = grouped
        self.partition_rebuilds += 1
        new_count = len(set(partition.values()))
        if not initial and old_count > new_count:
            self.class_merges += old_count - new_count

    def q_values(self, history: History) -> Tuple[float, float]:
        cells = self.group_stats.get(self.partition[history])
        if cells is None:
            return 0.0, 0.0
        return tuple(cell[0] / cell[1] if cell[1] else 0.0 for cell in cells)  # type: ignore[return-value]

    def choose(self, event: EpisodeEvent, epsilon: float) -> int:
        if event.explore_u < epsilon:
            return X if event.choice_u < 0.5 else Y
        qx, qy = self.q_values(event.history)
        if qx > qy:
            return X
        if qy > qx:
            return Y
        return X if event.choice_u < 0.5 else Y

    def observe(self, history: History, action: int, reward: int, episode: int) -> None:
        history_cells = self.history_stats.setdefault(history, self._blank_cells())
        history_cells[action][0] += reward
        history_cells[action][1] += 1
        group_cells = self.group_stats.setdefault(self.partition[history], self._blank_cells())
        group_cells[action][0] += reward
        group_cells[action][1] += 1

        # A singleton version space cannot be narrowed further.  Continuing to
        # "eliminate" the same rule would only inflate the inference-cost audit.
        if len(self.candidates) == 1:
            return
        observed_right = action if reward else 1 - action
        old = self.candidates
        predictions = prediction_signature(history, old)
        self.candidate_checks += len(old)
        self.candidates = tuple(rule for rule, prediction in zip(old, predictions)
                                if prediction == observed_right)
        if TRUE_RULE not in self.candidates:
            raise AssertionError("feedback removed the true rule")
        if len(self.candidates) == 1 and self.identified_episode is None:
            self.identified_episode = episode
        if self.candidates != old:
            if self.mode == "immediate" or (len(old) > 1 and len(self.candidates) == 1):
                self._rebuild_partition()

    def exact_accuracy(self) -> float:
        values = [expected_greedy_reward(*self.q_values(history), history)
                  for history in self.histories]
        return sum(values) / len(values)

    def class_count(self) -> int:
        return len(set(self.partition.values()))


class DirectRulePolicy:
    def __init__(self, histories: Sequence[History], parts: Sequence[str]) -> None:
        self.histories = tuple(histories)
        self.candidates: Tuple[Rule, ...] = all_candidate_rules(parts)
        self.candidate_checks = 0
        self.rule_action_predictions = 0
        self.identified_episode: Optional[int] = None
        self._last_history: Optional[History] = None
        self._last_predictions: Optional[Tuple[int, ...]] = None

    def predictions(self, history: History) -> Tuple[int, ...]:
        return prediction_signature(history, self.candidates)

    def choose(self, history: History, choice_u: float) -> int:
        predictions = self.predictions(history)
        self.rule_action_predictions += len(self.candidates)
        # The same predictions suffice for choosing and eliminating candidates;
        # cache them so the strong baseline is not charged twice for one pass.
        self._last_history = history
        self._last_predictions = predictions
        x_votes = predictions.count(X)
        y_votes = len(predictions) - x_votes
        if x_votes > y_votes:
            return X
        if y_votes > x_votes:
            return Y
        return X if choice_u < 0.5 else Y

    def observe(self, history: History, action: int, reward: int, episode: int) -> None:
        if len(self.candidates) == 1:
            self._last_history = None
            self._last_predictions = None
            return
        observed_right = action if reward else 1 - action
        if self._last_history == history and self._last_predictions is not None:
            predictions = self._last_predictions
        else:
            predictions = self.predictions(history)
        self.candidate_checks += len(self.candidates)
        self.candidates = tuple(rule for rule, prediction in zip(self.candidates, predictions)
                                if prediction == observed_right)
        self._last_history = None
        self._last_predictions = None
        if TRUE_RULE not in self.candidates:
            raise AssertionError("feedback removed the true rule")
        if len(self.candidates) == 1 and self.identified_episode is None:
            self.identified_episode = episode

    def expected_action_reward(self, history: History) -> float:
        predictions = self.predictions(history)
        x_votes = predictions.count(X)
        y_votes = len(predictions) - x_votes
        if x_votes == y_votes:
            return 0.5
        action = X if x_votes > y_votes else Y
        return float(action == correct_action(history))

    def exact_accuracy(self) -> float:
        return sum(self.expected_action_reward(history) for history in self.histories) / len(self.histories)

    def class_count(self) -> int:
        return len({prediction_signature(history, self.candidates) for history in self.histories})


def structure_snapshot(episode: int, candidates: int, classes: int) -> List[int]:
    return [episode, candidates, classes]


def run_round2_partition_method(
    method: str,
    n_parts: int,
    seed: int,
    events: Sequence[EpisodeEvent],
    trace_sha256: str,
    epsilon: float,
    eval_points: Sequence[int],
    code_sha256: str,
) -> dict:
    parts = parts_for(n_parts)
    histories = enumerate_histories(n_parts)
    policy = LearnedPartitionPolicy(histories, parts,
                                    "immediate" if method == "immediate_use" else "delayed")
    curve: List[List[float]] = [[0, policy.exact_accuracy()]]
    structure_curve = [structure_snapshot(0, len(policy.candidates), policy.class_count())]
    eval_set = set(eval_points)
    reward_sum = 0
    start = time.process_time()
    for episode, event in enumerate(events, 1):
        action = policy.choose(event, epsilon)
        reward = reward_for(event.history, action)
        reward_sum += reward
        policy.observe(event.history, action, reward, episode)
        if episode in eval_set:
            curve.append([episode, policy.exact_accuracy()])
            structure_curve.append(structure_snapshot(
                episode, len(policy.candidates), policy.class_count()))
    cpu_seconds = time.process_time() - start
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "round2_learned_compression",
        "method": method,
        "n_parts": n_parts,
        "n_histories": math.factorial(n_parts),
        "n_initial_candidates": n_parts * (n_parts - 1),
        "seed": seed,
        "episodes": len(events),
        "epsilon": epsilon,
        "q_update": "sample_average_alpha_1_over_n",
        "q_updates": len(events),
        "environment_reward_sum": reward_sum,
        "trace_sha256": trace_sha256,
        "curve": curve,
        "structure_curve": structure_curve,
        "t95": time_to_threshold(curve),
        "final_exact_accuracy": curve[-1][1],
        "final_state_count": policy.class_count(),
        "candidate_count_final": len(policy.candidates),
        "candidate_set_final": [f"{a}<{b}->X" for a, b in policy.candidates],
        "identified_episode": policy.identified_episode,
        "candidate_checks": policy.candidate_checks,
        "rule_action_predictions": policy.rule_action_predictions,
        "repartition_rule_evals": policy.repartition_rule_evals,
        "repartition_histories_scanned": policy.repartition_histories_scanned,
        "repartition_stat_cells_scanned": policy.repartition_stat_cells_scanned,
        "partition_rebuilds": policy.partition_rebuilds,
        "class_merges": policy.class_merges,
        "cpu_seconds": cpu_seconds,
        "code_sha256": code_sha256,
    }


def run_round2_direct(
    n_parts: int,
    seed: int,
    events: Sequence[EpisodeEvent],
    trace_sha256: str,
    epsilon: float,
    eval_points: Sequence[int],
    code_sha256: str,
) -> dict:
    histories = enumerate_histories(n_parts)
    policy = DirectRulePolicy(histories, parts_for(n_parts))
    curve: List[List[float]] = [[0, policy.exact_accuracy()]]
    structure_curve = [structure_snapshot(0, len(policy.candidates), policy.class_count())]
    eval_set = set(eval_points)
    reward_sum = 0
    start = time.process_time()
    for episode, event in enumerate(events, 1):
        action = policy.choose(event.history, event.choice_u)
        reward = reward_for(event.history, action)
        reward_sum += reward
        policy.observe(event.history, action, reward, episode)
        if episode in eval_set:
            curve.append([episode, policy.exact_accuracy()])
            structure_curve.append(structure_snapshot(
                episode, len(policy.candidates), policy.class_count()))
    cpu_seconds = time.process_time() - start
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "round2_learned_compression",
        "method": "direct_rule",
        "n_parts": n_parts,
        "n_histories": math.factorial(n_parts),
        "n_initial_candidates": n_parts * (n_parts - 1),
        "seed": seed,
        "episodes": len(events),
        "epsilon": epsilon,
        "q_update": "none",
        "q_updates": 0,
        "environment_reward_sum": reward_sum,
        "trace_sha256": trace_sha256,
        "curve": curve,
        "structure_curve": structure_curve,
        "t95": time_to_threshold(curve),
        "final_exact_accuracy": curve[-1][1],
        "final_state_count": policy.class_count(),
        "candidate_count_final": len(policy.candidates),
        "candidate_set_final": [f"{a}<{b}->X" for a, b in policy.candidates],
        "identified_episode": policy.identified_episode,
        "candidate_checks": policy.candidate_checks,
        "rule_action_predictions": policy.rule_action_predictions,
        "repartition_rule_evals": 0,
        "repartition_histories_scanned": 0,
        "repartition_stat_cells_scanned": 0,
        "partition_rebuilds": 0,
        "class_merges": 0,
        "cpu_seconds": cpu_seconds,
        "code_sha256": code_sha256,
    }


def run_seed_bundle(spec: Tuple[int, int, int, float, Tuple[str, ...], str]) -> List[dict]:
    n_parts, seed, episodes, epsilon, phases, code_sha256 = spec
    events, trace_sha256 = make_trace(n_parts, seed, episodes)
    eval_points = evaluation_schedule(episodes)
    rows: List[dict] = []
    if "round1" in phases:
        for method in ROUND1_METHODS:
            rows.append(run_round1_method(
                method, n_parts, seed, events, trace_sha256, epsilon, eval_points, code_sha256))
    if "round2" in phases:
        # The round-2 full-history row intentionally reuses the same learner and
        # trace as round 1; the checks assert exact curve equality.
        rows.append(run_round1_method(
            "full_history", n_parts, seed, events, trace_sha256, epsilon,
            eval_points, code_sha256) | {"phase": "round2_learned_compression"})
        for method in ("immediate_use", "delayed_use"):
            rows.append(run_round2_partition_method(
                method, n_parts, seed, events, trace_sha256, epsilon, eval_points, code_sha256))
        rows.append(run_round2_direct(
            n_parts, seed, events, trace_sha256, epsilon, eval_points, code_sha256))
    return rows


def row_key(row: Mapping[str, object]) -> Tuple[object, ...]:
    return (row["phase"], row["method"], row["n_parts"], row["seed"],
            row["episodes"], row["epsilon"], row.get("code_sha256"))


def load_completed(path: Path) -> set:
    completed = set()
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            completed.add(row_key(json.loads(line)))
        except (json.JSONDecodeError, KeyError) as exc:
            raise SystemExit(f"{path}:{line_number}: invalid result row: {exc}")
    return completed


def parse_csv_ints(value: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="append-only raw JSONL output")
    parser.add_argument("--parts", default="3,4,5", help="comma-separated part counts")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=400, help="number of consecutive seeds")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--phases", default="round1,round2")
    parser.add_argument("--workers", type=int, default=max(1, min(24, os.cpu_count() or 1)))
    args = parser.parse_args(argv)

    n_values = parse_csv_ints(args.parts)
    phases = parse_csv_strings(args.phases)
    if not n_values or any(n < 3 or n > 8 for n in n_values):
        parser.error("--parts must contain integers from 3 through 8")
    if not set(phases) <= {"round1", "round2"} or not phases:
        parser.error("--phases may contain round1 and/or round2")
    if args.episodes <= 0 or args.seeds <= 0 or args.workers <= 0:
        parser.error("--episodes, --seeds and --workers must be positive")
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error("--epsilon must be in [0, 1]")

    code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(args.out)
    specs = [
        (n, seed, args.episodes, args.epsilon, phases, code_sha256)
        for n in n_values
        for seed in range(args.seed_start, args.seed_start + args.seeds)
    ]
    methods_per_bundle = (len(ROUND1_METHODS) if "round1" in phases else 0) + (
        len(ROUND2_METHODS) if "round2" in phases else 0)
    total_rows = len(specs) * methods_per_bundle
    already = sum(1 for key in completed
                  if key[-1] == code_sha256 and key[4] == args.episodes and key[5] == args.epsilon)
    manifest = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "workers": args.workers,
    }
    print(f"memory-compression pilot: {len(specs)} seed×size bundles, {total_rows} rows; "
          f"workers={args.workers}, existing matching rows={already}", flush=True)
    started = time.monotonic()
    written = 0

    def persist(rows: Iterable[dict]) -> None:
        nonlocal written
        with args.out.open("a", encoding="utf-8") as handle:
            for row in rows:
                key = row_key(row)
                if key in completed:
                    continue
                row["run_manifest"] = manifest
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                completed.add(key)
                written += 1

    if args.workers == 1:
        for index, spec in enumerate(specs, 1):
            persist(run_seed_bundle(spec))
            if index % max(1, len(specs) // 20) == 0 or index == len(specs):
                print(f"  bundles {index}/{len(specs)}; new rows {written}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_seed_bundle, spec): spec[:2] for spec in specs}
            for index, future in enumerate(as_completed(futures), 1):
                persist(future.result())
                if index % max(1, len(specs) // 20) == 0 or index == len(specs):
                    elapsed = time.monotonic() - started
                    print(f"  bundles {index}/{len(specs)}; new rows {written}; {elapsed:.1f}s", flush=True)

    elapsed = time.monotonic() - started
    print(f"done: wrote {written} rows to {args.out} in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
