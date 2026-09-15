#!/usr/bin/env python3
"""Exhaustive and trajectory-level checks for memory_compression_pilot.py."""

from __future__ import annotations

import itertools
import sys

from memory_compression_pilot import (
    ROUND1_METHODS,
    TRUE_RULE,
    X,
    Y,
    all_candidate_rules,
    automaton_state,
    correct_action,
    enumerate_histories,
    handwritten_state,
    parts_for,
    predict_action,
    prediction_signature,
    progress_state,
    reward_for,
    run_seed_bundle,
)


failures = []


def check(label: str, condition: bool, detail: str = "") -> None:
    marker = "  [通过] " if condition else "  [失败] "
    print(marker + label + (("  " + detail) if detail else ""))
    if not condition:
        failures.append(label)


def strip_nonsemantic(row: dict) -> dict:
    ignored = {"phase", "method", "cpu_seconds", "code_sha256"}
    return {key: value for key, value in row.items() if key not in ignored}


def main() -> int:
    expected = {3: 6, 4: 24, 5: 120}
    actual = {n: len(enumerate_histories(n)) for n in expected}
    check("C1. 完整历史数量为 3!=6、4!=24、5!=120", actual == expected, str(actual))

    equivalent = True
    for n in expected:
        for history in enumerate_histories(n):
            if automaton_state(history) != handwritten_state(history):
                equivalent = False
                break
    check("C2. 正确自动机与手写变量在所有完整顺序上逐项等价", equivalent)

    prefix_equivalent = True
    for n in expected:
        for history in enumerate_histories(n):
            for end in range(1, n + 1):
                prefix = history[:end]
                if "A" not in prefix and "B" not in prefix:
                    continue
                if automaton_state(prefix) != handwritten_state(prefix):
                    prefix_equivalent = False
    check("C3. 两种实现对所有已经见到 A/B 的前缀也逐项等价", prefix_equivalent)

    reward_equivalent = True
    group_sizes = {}
    for n in expected:
        grouped = {0: [], 1: []}
        for history in enumerate_histories(n):
            grouped[automaton_state(history)].append(history)
        group_sizes[n] = tuple(len(grouped[state]) for state in (0, 1))
        for histories in grouped.values():
            for action in (X, Y):
                outcomes = {(reward_for(history, action), True) for history in histories}
                reward_equivalent &= len(outcomes) == 1
    check("C4. 压缩组内对 X/Y 的奖励与终止结果严格相同", reward_equivalent,
          f"组大小 {group_sizes}")

    progress_bound = True
    for n in expected:
        histories = enumerate_histories(n)
        progress_bound &= len({progress_state(h) for h in histories}) == 1
        for action in (X, Y):
            progress_bound &= sum(reward_for(h, action) for h in histories) / len(histories) == 0.5
    check("C5. 只记进度时最终只有一个状态，任一固定动作准确率恰为 0.5", progress_bound)

    candidate_counts = {n: len(all_candidate_rules(parts_for(n))) for n in expected}
    check("C6. 第二轮候选规则数为 n(n-1)",
          candidate_counts == {3: 6, 4: 12, 5: 20}, str(candidate_counts))

    feedback_safe = True
    partition_safe = True
    for n in expected:
        histories = enumerate_histories(n)
        rules = all_candidate_rules(parts_for(n))
        for observed in histories:
            label = correct_action(observed)
            remaining = tuple(r for r in rules if predict_action(observed, r) == label)
            feedback_safe &= TRUE_RULE in remaining
            groups = {}
            for history in histories:
                groups.setdefault(prediction_signature(history, remaining), []).append(history)
            partition_safe &= all(len({correct_action(h) for h in group}) == 1
                                  for group in groups.values())
    check("C7. 任一正确反馈都不会排除真实规则", feedback_safe)
    check("C8. 真实规则仍在时，预测签名绝不合并不同正确动作", partition_safe)

    # Run actual learners on identical traces, long enough to exercise all paths.
    rows = run_seed_bundle((5, 17, 600, 0.10, ("round1", "round2"), "check"))
    by_key = {(row["phase"], row["method"]): row for row in rows}
    auto = by_key[("round1_known_compression", "correct_automaton")]
    manual = by_key[("round1_known_compression", "correct_handwritten")]
    check("C9. 自动机与手写程序在同一随机流上学习曲线逐点相同",
          auto["curve"] == manual["curve"] and
          auto["environment_reward_sum"] == manual["environment_reward_sum"])

    r1_full = by_key[("round1_known_compression", "full_history")]
    r2_full = by_key[("round2_learned_compression", "full_history")]
    check("C10. 两轮的完整历史基线使用同一协议并逐点相同",
          strip_nonsemantic(r1_full) == strip_nonsemantic(r2_full))

    inferred = [by_key[("round2_learned_compression", method)]
                for method in ("immediate_use", "delayed_use", "direct_rule")]
    candidate_traces = [[point[:2] for point in row["structure_curve"]] for row in inferred]
    check("C11. 立即、延迟和直接规则法的候选集大小轨迹相同",
          candidate_traces[0] == candidate_traces[1] == candidate_traces[2])
    check("C12. 三种推断法最终都唯一识别 A<B→X",
          all(row["candidate_set_final"] == ["A<B->X"] for row in inferred))

    q_budget_ok = all(row["q_updates"] == 600 for row in rows if row["method"] != "direct_rule")
    direct_budget_ok = by_key[("round2_learned_compression", "direct_rule")]["q_updates"] == 0
    check("C13. 所有 RL 方法每幕恰好一次 Q 更新；直接规则法不做 Q 更新",
          q_budget_ok and direct_budget_ok)

    count_curve = by_key[("round1_known_compression", "progress_count")]["curve"]
    check("C14. 进度诊断的精确评估曲线全程为 0.5",
          all(point[1] == 0.5 for point in count_curve))

    delayed = by_key[("round2_learned_compression", "delayed_use")]
    delayed_structure_ok = True
    for episode, candidates, classes in delayed["structure_curve"]:
        expected_classes = 2 if candidates == 1 else 120
        delayed_structure_ok &= classes == expected_classes
    check("C15. 延迟法在唯一识别前保留 120 条历史，之后才压成 2 类",
          delayed_structure_ok)

    immediate = by_key[("round2_learned_compression", "immediate_use")]
    cost_ok = (immediate["candidate_checks"] > 0 and
               immediate["repartition_rule_evals"] > 0 and
               immediate["repartition_histories_scanned"] > 0 and
               delayed["repartition_rule_evals"] > 0)
    check("C16. 推断检查与旧经验重组成本均被显式计数", cost_ok)

    # A complete enumeration is a deterministic identifying dataset.
    for n in expected:
        candidates = all_candidate_rules(parts_for(n))
        for history in enumerate_histories(n):
            label = correct_action(history)
            candidates = tuple(rule for rule in candidates if predict_action(history, rule) == label)
        check(f"C17.{n}. 枚举全部 {n}! 条历史后候选集唯一", candidates == (TRUE_RULE,),
              str(candidates))

    if failures:
        print(f"\n{len(failures)} 项失败：" + "；".join(failures))
        return 1
    print("\n全部通过：环境、等价类、安全共享、随机流与更新预算满足冻结协议。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
