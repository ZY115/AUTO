# 数据与审计证据

审计只增加观察或构造小型接口例子；没有修复原实现后重新训练整批实验。

| 结果文件 | 记录数 | 处理 |
|---|---:|---|
| [goal_vs_index.jsonl](</Users/yuhang/Downloads/why TL/AUTO/progressive_task_discovery/stage8_compression/results/goal_vs_index.jsonl>) | 1728 | 聚合、记录 SHA256；不同条件不混池 |
| [priorart.jsonl](</Users/yuhang/Downloads/why TL/AUTO/progressive_task_discovery/stage8_compression/results/priorart.jsonl>) | 1536 | 聚合、记录 SHA256；不同条件不混池 |
| [mergerule.jsonl](</Users/yuhang/Downloads/why TL/AUTO/progressive_task_discovery/stage8_compression/results/mergerule.jsonl>) | 1320 | 聚合、记录 SHA256；不同条件不混池 |
| [revision.jsonl](</Users/yuhang/Downloads/why TL/AUTO/progressive_task_discovery/stage8_compression/results/revision.jsonl>) | 576 | 聚合、记录 SHA256；不同条件不混池 |
| [irreversible/goal.jsonl](</Users/yuhang/Downloads/why TL/AUTO/progressive_task_discovery/stage8_compression/results/irreversible/goal.jsonl>) | 900 | 聚合、记录 SHA256；不同条件不混池 |
| [overlap_run2/raw.jsonl](</Users/yuhang/Downloads/why TL/AUTO/progressive_task_discovery/stage8_compression/results/overlap_run2/raw.jsonl>) | 2000 | 聚合、记录 SHA256；不同条件不混池 |
| [irreversible/dose.jsonl](</Users/yuhang/Downloads/why TL/AUTO/progressive_task_discovery/stage8_compression/results/irreversible/dose.jsonl>) | 2160 | 聚合、记录 SHA256；不同条件不混池 |
| [rawhistory.jsonl](</Users/yuhang/Downloads/why TL/AUTO/progressive_task_discovery/stage8_compression/results/rawhistory.jsonl>) | 960 | 聚合、记录 SHA256；不同条件不混池 |

## 可复查文件

- [data_summary.json](</Users/yuhang/Downloads/why TL/AUTO/research/2026-09-10-stage8-review/audit/data_summary.json>)：聚合原始字段：first90_mean 可能含 -1；goal_vs_index 另列 first90_censored_mean 才是报告口径。
- [manifest_check.json](</Users/yuhang/Downloads/why TL/AUTO/research/2026-09-10-stage8-review/audit/manifest_check.json>)：最新五份源码/数据 hash 均匹配。
- [irreversible_witness.json](</Users/yuhang/Downloads/why TL/AUTO/research/2026-09-10-stage8-review/audit/irreversible_witness.json>)：保存完整输出、命令、诊断以及与既有行的对照。
- [goal_replay_witness.json](</Users/yuhang/Downloads/why TL/AUTO/research/2026-09-10-stage8-review/audit/goal_replay_witness.json>)：同一 Raw 在 relabel=false 下改变被更新目标。
- [signature_witness.json](</Users/yuhang/Downloads/why TL/AUTO/research/2026-09-10-stage8-review/audit/signature_witness.json>)：局部细化的语义反例，不是完整 JIRP 运行。
- [corrected_fatal_bootstrap.json](</Users/yuhang/Downloads/why TL/AUTO/research/2026-09-10-stage8-review/audit/corrected_fatal_bootstrap.json>)：地图重采样保留重复次数；没有修复学习数据污染。
- [aalpy_smoke.json](</Users/yuhang/Downloads/why TL/AUTO/research/2026-09-10-stage8-review/audit/aalpy_smoke.json>)：GSM 通过，classic 未通过同样的长词验证。
- [snapshot_manifest.json](</Users/yuhang/Downloads/why TL/AUTO/research/2026-09-10-stage8-review/audit/snapshot_manifest.json>)：保存核心发现的源码/报告快照；不复制全部历史数据。

## 重现入口

从工作区根目录执行：

```bash
python3 research/2026-09-10-stage8-review/tools/audit_existing.py
python3 research/2026-09-10-stage8-review/tools/signature_witness.py
clang++ -std=c++17 -O2 research/2026-09-10-stage8-review/audit/goal_replay_witness.cpp -o research/2026-09-10-stage8-review/audit/goal_replay_witness
research/2026-09-10-stage8-review/audit/goal_replay_witness
/Users/yuhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 research/2026-09-10-stage8-review/tools/smoke_aalpy.py
```

audit_existing.py 会再次执行一个已有 200000 步条件并写入本审计目录；不是新参数扫描。它默认读当前源码，后续若项目变化，应先核对 snapshot hash。目标回放见证通过 include 调用原始 Learner，不更改其逻辑。

## 限制

没有重算全部历史报告的区间；没有证明修复后的效应大小或方向。旧 irreversible_run/run2 与 dose 版本可能不同，本审阅不混成同一批证据。
