#!/usr/bin/env python3
"""Analyze raw JSONL from memory_compression_pilot.py using seeds as units."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROUND1 = "round1_known_compression"
ROUND2 = "round2_learned_compression"
METHOD_ORDER = {
    ROUND1: ("full_history", "correct_automaton", "correct_handwritten", "progress_count"),
    ROUND2: ("full_history", "immediate_use", "delayed_use", "direct_rule"),
}
METHOD_LABEL = {
    "full_history": "完整历史",
    "correct_automaton": "正确自动机",
    "correct_handwritten": "等价手写程序",
    "progress_count": "只记进度",
    "immediate_use": "立即使用部分结构",
    "delayed_use": "完整识别后再使用",
    "direct_rule": "候选规则直接决策",
}


def load_rows(paths: Sequence[Path]) -> List[dict]:
    rows = []
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_number}: {exc}")
                if row.get("schema_version") != 1:
                    raise SystemExit(f"{path}:{line_number}: unsupported schema")
                rows.append(row)
    if not rows:
        raise SystemExit("no result rows found")
    return rows


def bootstrap_mean(values: Sequence[float], samples: int = 10000, seed: int = 240915) -> Tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return statistics.mean(values), draws[int(0.025 * samples)], draws[int(0.975 * samples)]


def paired_bootstrap(
    left: Mapping[int, float], right: Mapping[int, float], samples: int = 10000, seed: int = 240916
) -> Tuple[float, float, float, int]:
    seeds = sorted(set(left) & set(right))
    differences = [left[s] - right[s] for s in seeds]
    mean, low, high = bootstrap_mean(differences, samples, seed)
    return mean, low, high, len(seeds)


def regret_area(curve: Sequence[Sequence[float]], horizon: int) -> float:
    """Trapezoidal integral of 1-accuracy, divided by the horizon."""
    if horizon <= 0:
        return 0.0
    total = 0.0
    for left, right in zip(curve, curve[1:]):
        width = min(horizon, int(right[0])) - min(horizon, int(left[0]))
        if width <= 0:
            continue
        total += width * ((1.0 - float(left[1])) + (1.0 - float(right[1]))) / 2.0
        if right[0] >= horizon:
            break
    return total / horizon


def rmst_value(row: Mapping[str, object]) -> float:
    return float(row["episodes"] if row.get("t95") is None else row["t95"])


def pre_identification_expected_success(row: Mapping[str, object]) -> float:
    """Expected correct decisions through the identifying episode.

    The action in episode t uses the policy after feedback t-1.  Evaluation is
    dense at every early episode, so summing accuracies at 0..I-1 gives the
    iid expected number correct for actions 1..I without training-action noise.
    """
    identified = row.get("identified_episode")
    if identified is None:
        return math.nan
    by_episode = {int(point[0]): float(point[1]) for point in row["curve"]}
    needed = range(int(identified))
    missing = [episode for episode in needed if episode not in by_episode]
    if missing:
        raise SystemExit("evaluation schedule is not dense through rule identification")
    return sum(by_episode[episode] for episode in needed)


def fmt_ci(values: Sequence[float], digits: int = 1) -> str:
    mean, low, high = bootstrap_mean(values)
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def validate(rows: Sequence[dict]) -> List[str]:
    notes = []
    keys = [(row["phase"], row["method"], row["n_parts"], row["seed"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate phase/method/n_parts/seed rows")
    configs = {(row["episodes"], row["epsilon"], row["code_sha256"]) for row in rows}
    if len(configs) != 1:
        raise SystemExit(f"mixed formal protocols in one analysis: {configs}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["n_parts"], row["seed"])].append(row)
    for key, group in grouped.items():
        traces = {row["trace_sha256"] for row in group}
        if len(traces) != 1:
            raise SystemExit(f"common-random-number violation at n,seed={key}")

    by = {(row["phase"], row["method"], row["n_parts"], row["seed"]): row for row in rows}
    for n in sorted({row["n_parts"] for row in rows}):
        seeds = sorted({row["seed"] for row in rows if row["n_parts"] == n})
        for seed in seeds:
            auto = by.get((ROUND1, "correct_automaton", n, seed))
            manual = by.get((ROUND1, "correct_handwritten", n, seed))
            if auto and manual and (auto["curve"] != manual["curve"] or
                                    auto["environment_reward_sum"] != manual["environment_reward_sum"]):
                raise SystemExit(f"automaton/handwritten mismatch at n={n}, seed={seed}")
            r1 = by.get((ROUND1, "full_history", n, seed))
            r2 = by.get((ROUND2, "full_history", n, seed))
            if r1 and r2 and r1["curve"] != r2["curve"]:
                raise SystemExit(f"round-1/round-2 history mismatch at n={n}, seed={seed}")
            inferred = [by.get((ROUND2, method, n, seed))
                        for method in ("immediate_use", "delayed_use", "direct_rule")]
            if all(inferred):
                traces = [[point[1] for point in row["structure_curve"]] for row in inferred]
                if not (traces[0] == traces[1] == traces[2]):
                    raise SystemExit(f"candidate trace mismatch at n={n}, seed={seed}")
    notes.append("随机序列逐种子配对；自动机=手写程序、两轮完整历史、三种候选集轨迹均已复核。")
    return notes


def subset(rows: Sequence[dict], phase: str, method: str, n: int) -> List[dict]:
    return [row for row in rows if row["phase"] == phase and row["method"] == method and row["n_parts"] == n]


def make_report(rows: Sequence[dict], source_names: Sequence[str]) -> str:
    validation_notes = validate(rows)
    episodes = int(rows[0]["episodes"])
    epsilon = float(rows[0]["epsilon"])
    ns = sorted({int(row["n_parts"]) for row in rows})
    seed_count = len({int(row["seed"]) for row in rows})
    manifest = rows[0].get("run_manifest", {})
    code_hashes = sorted({str(row["code_sha256"]) for row in rows})
    machine = (f"{manifest.get('platform', 'unknown')}；{manifest.get('cpu_count', '?')} 逻辑 CPU，"
               f"{manifest.get('workers', '?')} workers；Python {manifest.get('python', '?')}")
    lines = [
        "# 记忆压缩 pilot：正式结果",
        "",
        f"- 原始数据：{', '.join(source_names)}",
        f"- 协议：每个规模 {seed_count} 个配对种子，{episodes:,} 幕，常数 ε={epsilon:.2f}；顺序 iid 均匀抽样。",
        f"- 运行环境：{machine}。",
        f"- runner SHA-256：`{code_hashes[0]}`。",
        "- 评估：冻结策略枚举全部 n! 条历史；Q 值并列时按随机并列的期望计分，不把评估经验带回训练。",
        "- 主速度量：首次在已记录评估点达到且以后保持 95% 的幕数；未达到按完整预算计入 RMST。",
        "- 曲线损失：归一化面积 ∫(1-accuracy)/预算，越低越好。",
        f"- 验收：{' '.join(validation_notes)}",
        "",
        "## 第一轮：已知正确压缩",
        "",
        "| 零件 | 方法 | 末点精确准确率（95% CI） | T95/RMST 幕（95% CI） | 曲线损失 | 最终状态数 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for n in ns:
        for method in METHOD_ORDER[ROUND1]:
            group = subset(rows, ROUND1, method, n)
            if not group:
                continue
            finals = [float(row["final_exact_accuracy"]) for row in group]
            times = [rmst_value(row) for row in group]
            regrets = [regret_area(row["curve"], episodes) for row in group]
            states = [float(row["final_state_count"]) for row in group]
            lines.append(
                f"| {n} | {METHOD_LABEL[method]} | {fmt_ci(finals, 3)} | {fmt_ci(times, 1)} | "
                f"{statistics.mean(regrets):.4f} | {statistics.mean(states):.1f} |")

    lines += ["", "### 第一轮的配对效应", "",
              "| 零件 | 完整历史 RMST / 自动机 RMST | 自动机节省幕数（配对 95% CI） | 自动机与手写最大曲线差 |",
              "|---:|---:|---:|---:|"]
    for n in ns:
        full = subset(rows, ROUND1, "full_history", n)
        auto = subset(rows, ROUND1, "correct_automaton", n)
        manual = subset(rows, ROUND1, "correct_handwritten", n)
        if not full or not auto:
            continue
        full_t = {int(r["seed"]): rmst_value(r) for r in full}
        auto_t = {int(r["seed"]): rmst_value(r) for r in auto}
        saved, low, high, _ = paired_bootstrap(full_t, auto_t)
        ratio = statistics.mean(full_t.values()) / statistics.mean(auto_t.values())
        manual_by_seed = {int(r["seed"]): r for r in manual}
        max_gap = max(
            abs(float(a[1]) - float(b[1]))
            for row in auto for a, b in zip(row["curve"], manual_by_seed[int(row["seed"])]["curve"])
        ) if manual else math.nan
        lines.append(f"| {n} | {ratio:.2f}× | {saved:.1f} [{low:.1f}, {high:.1f}] | {max_gap:.3g} |")

    lines += [
        "",
        "## 第二轮：从反馈学习压缩",
        "",
        "| 零件 | 方法 | 末点精确准确率（95% CI） | T95/RMST 幕（95% CI） | 曲线损失 | 识别规则幕数（均值） |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for n in ns:
        for method in METHOD_ORDER[ROUND2]:
            group = subset(rows, ROUND2, method, n)
            if not group:
                continue
            finals = [float(row["final_exact_accuracy"]) for row in group]
            times = [rmst_value(row) for row in group]
            regrets = [regret_area(row["curve"], episodes) for row in group]
            identified = [float(row["identified_episode"]) for row in group
                          if row.get("identified_episode") is not None]
            id_text = f"{statistics.mean(identified):.2f}" if identified else "—"
            lines.append(
                f"| {n} | {METHOD_LABEL[method]} | {fmt_ci(finals, 3)} | {fmt_ci(times, 1)} | "
                f"{statistics.mean(regrets):.4f} | {id_text} |")

    lines += ["", "### 立即使用相对延迟使用", "",
              "“识别前预期多答对”对每个种子累加第 1 幕到唯一识别那一幕的精确策略准确率差；动作发生在反馈和候选排除之前，因此使用曲线的第 0 到 I−1 点。",
              "",
              "| 零件 | 立即法在识别前预期多答对（配对 95% CI） | 延迟 RMST − 立即 RMST（配对 95% CI） | 延迟曲线损失 − 立即曲线损失 | 直接规则 RMST |",
              "|---:|---:|---:|---:|---:|"]
    for n in ns:
        immediate = subset(rows, ROUND2, "immediate_use", n)
        delayed = subset(rows, ROUND2, "delayed_use", n)
        direct = subset(rows, ROUND2, "direct_rule", n)
        if not immediate or not delayed:
            continue
        immediate_t = {int(r["seed"]): rmst_value(r) for r in immediate}
        delayed_t = {int(r["seed"]): rmst_value(r) for r in delayed}
        gap, low, high, _ = paired_bootstrap(delayed_t, immediate_t)
        immediate_pre = {int(r["seed"]): pre_identification_expected_success(r) for r in immediate}
        delayed_pre = {int(r["seed"]): pre_identification_expected_success(r) for r in delayed}
        pre_gap, pre_low, pre_high, _ = paired_bootstrap(immediate_pre, delayed_pre)
        immediate_regret = {int(r["seed"]): regret_area(r["curve"], episodes) for r in immediate}
        delayed_regret = {int(r["seed"]): regret_area(r["curve"], episodes) for r in delayed}
        regret_gap = statistics.mean(delayed_regret[s] - immediate_regret[s]
                                     for s in set(immediate_regret) & set(delayed_regret))
        direct_rmst = statistics.mean(rmst_value(r) for r in direct)
        lines.append(
            f"| {n} | {pre_gap:.3f} [{pre_low:.3f}, {pre_high:.3f}] | "
            f"{gap:.2f} [{low:.2f}, {high:.2f}] | {regret_gap:.6f} | {direct_rmst:.2f} |")

    lines += [
        "",
        "## 推断与重组成本",
        "",
        "这些是训练期的算法操作计数，不把周期性评估算进去。候选唯一后停止排除检查；直接规则法仍每幕评估唯一规则来决策。候选预测可同时用于决策和排除；重组没有重放 TD 更新。",
        "",
        "| 零件 | 方法 | 候选排除检查/次运行 | 决策规则预测/次运行 | 重组规则预测/次运行 | 扫描历史/次运行 | Q 更新/次运行 | CPU 秒/次运行 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n in ns:
        for method in METHOD_ORDER[ROUND2]:
            group = subset(rows, ROUND2, method, n)
            if not group:
                continue
            mean_field = lambda field: statistics.mean(float(row.get(field, 0)) for row in group)
            lines.append(
                f"| {n} | {METHOD_LABEL[method]} | {mean_field('candidate_checks'):.1f} | "
                f"{mean_field('rule_action_predictions'):.1f} | "
                f"{mean_field('repartition_rule_evals'):.1f} | "
                f"{mean_field('repartition_histories_scanned'):.1f} | "
                f"{mean_field('q_updates'):.1f} | {mean_field('cpu_seconds'):.4f} |")

    # Data-dependent but conservative interpretation.
    r1_ratios = []
    for n in ns:
        full = subset(rows, ROUND1, "full_history", n)
        auto = subset(rows, ROUND1, "correct_automaton", n)
        if full and auto:
            r1_ratios.append((n, statistics.mean(rmst_value(r) for r in full) /
                              statistics.mean(rmst_value(r) for r in auto)))
    immediate_gaps = []
    immediate_pre_gaps = []
    for n in ns:
        immediate = subset(rows, ROUND2, "immediate_use", n)
        delayed = subset(rows, ROUND2, "delayed_use", n)
        if immediate and delayed:
            immediate_gaps.append(statistics.mean(
                rmst_value(d) - rmst_value(i) for i, d in zip(
                    sorted(immediate, key=lambda r: r["seed"]), sorted(delayed, key=lambda r: r["seed"]))))
            immediate_pre_gaps.append(statistics.mean(
                pre_identification_expected_success(i) - pre_identification_expected_success(d)
                for i, d in zip(sorted(immediate, key=lambda r: r["seed"]),
                                sorted(delayed, key=lambda r: r["seed"]))))
    direct_is_fastest = all(
        statistics.mean(rmst_value(r) for r in subset(rows, ROUND2, "direct_rule", n)) <=
        min(statistics.mean(rmst_value(r) for r in subset(rows, ROUND2, method, n))
            for method in ("full_history", "immediate_use", "delayed_use"))
        for n in ns if subset(rows, ROUND2, "direct_rule", n)
    )
    lines += [
        "",
        "## 边界内结论",
        "",
        f"第一轮中，完整历史相对正确自动机的 RMST 比从 "
        + "、".join(f"n={n} 时 {ratio:.2f}×" for n, ratio in r1_ratios) + "。",
        "自动机与等价手写程序逐种子、逐评估点完全一致；因此这里测到的是正确状态归类的作用，不是自动机语法本身的额外能力。",
        ("第二轮里立即使用只在规则唯一识别前带来小幅预期收益，T95 没有分开；规则通常在几幕内已经唯一识别，留给“部分结构”的窗口很短。"
         if immediate_gaps and max(immediate_gaps) < 1.0 and max(immediate_pre_gaps, default=0.0) > 0 else
         "第二轮里立即使用没有表现出稳定的识别前收益；在这个任务中部分结构窗口过短，不能据此主张立即使用更好。"
         if immediate_gaps and max(immediate_gaps) < 1.0 else
         "第二轮里立即使用在唯一识别前已经产生可测优势，但它只支持这个受限有序对规则族中的安全共享。"),
        ("候选规则直接决策在所有已运行规模上都是最快的方法；这个确定反馈的小任务直接做候选消除已经足够，再套一层 Q-learning 没有样本效率优势。"
         if direct_is_fastest else
         "候选规则直接决策没有在所有规模占优，但仍是解释结构学习收益所必需的强对照。"),
        "本实验不涉及主动选择检查路线、带噪反馈、长程技能执行或通用自动机归纳，不能据此外推这些能力。",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(argv)
    rows = load_rows(args.paths)
    report = make_report(rows, [path.name for path in args.paths])
    print(report)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
