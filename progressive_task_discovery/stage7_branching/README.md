# Stage 7: branching task discovery

本阶段已完成：6,096组主运行、6个批次、560,640,000次训练交互；额外128次诊断回放不算独立实验。先读 [中文结果报告](</Users/yuhang/Downloads/why TL/AUTO/progressive_task_discovery/stage7_branching/REPORT.md>)。

分支汇合后的相同位置、进度计数、最后成功事件仍可能对应不同下一目标。完整历史能消除歧义，立即技能复用在长任务中有效；当前前缀树与等价历史字典完全一致，不能据此主张一般自动机推断优势。Stage 8目前只有设计，没有执行。

## 验证与复现

需要C++17编译器、Python 3、NumPy和Matplotlib。所有计算均为CPU上的小型表格学习，没有神经网络。以下命令从本目录运行：

```bash
make test
python3 src/audit_results.py
```

8项测试检查独立任务语义、地图、最短路径、历史歧义、等价表示和确定性/随机策略评价。全数据审计检查预算、正转换、等价运行和抽取的冻结策略。

从保存的原始数据重新生成统计、图和报告：

```bash
python3 src/analyze_stage7.py
python3 src/plot_stage7.py
python3 src/write_report.py
```

统计脚本复用父项目的 `src/analyze.py`，迁移项目时请保留父目录。统计采用配对种子的bootstrap区间，报告中未达标运行保持右删失，不当作恰好100k步成功。

新实验应复制 `docs/heldout.json` 等配置，修改JSON中的 `name` 为新的结果目录名，再运行：

```bash
make
python3 src/run_suite.py docs/your_new_spec.json --workers 2
```

运行器拒绝覆盖现有结果。六批原始配置、运行命令、源码快照和SHA-256都保存在各自结果目录。**早期版本Q表容量64，后续为128；精确复现整张Q表哈希必须编译对应批次的 `branch_snapshot.cpp`。** 将该批 `raw.jsonl` 内的 `command[0]` 替换为快照可执行文件路径，保留其余参数，再对比 `trace_hash`、`q_hash` 和 `checkpoints`；运行时间不参与等价检查。

最终源码增加了按任务上下文统计技能调用的诊断字段。`results/diagnostic_reproduction.json` 记录128次正式配置复跑的行为一致性；对应细分数据在 `results/skill_context_diagnostics.jsonl`。

## 文件组织

- `src/branch.cpp`：环境、任务生成、九种方法、技能调用和评价。
- `tests/test_branch.py`：独立Python参考检查。
- `docs/protocol.md`：最初协议；最终共享事件改动见决策日志。
- `docs/decision_log.md`：每轮观察、后续设计、参数冻结和停止理由。
- `docs/frozen_selection.json`：开发集选定的参数。
- `results/{development,bridge_development,heldout,long_bridge,matched_epsilon,same_map_control}/`：原始JSONL、配置、源码快照、manifest和统计CSV。
- `results/audit.json`、`results/validation.txt`：审计与测试证据。
- `figures/`：PNG预览与可导出的SVG科研图。
- `docs/NEXT_STAGE8.md`：固定技能库的数据控制设计，尚未执行。

正式主验证使用种子400–431；更长共享历史使用500–531；开发使用0–11。不同方法和任务长度重复使用配对种子，6,096次运行不是6,096个独立种子。同地图控制是在看到主结果后添加的机制核查，未用于重新调参。
