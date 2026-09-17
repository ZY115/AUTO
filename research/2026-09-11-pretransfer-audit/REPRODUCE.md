# 复核入口

以下命令从 `/Users/yuhang/Downloads/why TL` 执行。输出只写入审计目录。原始运行器可能覆盖已有结果，本次没有重新运行这些主网格脚本。

```bash
python3 research/2026-09-11-pretransfer-audit/recompute.py
.venv-rm/bin/python research/2026-09-11-pretransfer-audit/structure_checks.py
clang++ -O2 -std=c++17 progressive_task_discovery/stage8_compression/src/compress.cpp -o research/2026-09-11-pretransfer-audit/compress-rebuilt
python3 research/2026-09-11-pretransfer-audit/probes.py
python3 research/2026-09-11-pretransfer-audit/followup_probes.py
```

`recompute.py` 无第三方依赖。`structure_checks.py` 需要现有 `.venv-rm`，并复用仓库的显式 MiniGrid stub；它验证的是作者自动机解释器，不是作者物理环境。

`probes.py` 重放 18 个历史设置，并同时与历史摘要、当前旧二进制比较；另跑 18 个稀疏奖励设置。`followup_probes.py` 隔离 QRM 两个奖励参数，再补 safe 条件。它们不构成总体确认网格；完整命令随每条输出保存。

现有测试从 `progressive_task_discovery/stage8_compression` 执行：

```bash
python3 -m unittest discover -s tests
```

本次结果：59 项通过。110 个项目 Python 文件 AST 解析通过。全报告链接检查覆盖 STEP39_PORTING 与 Step 30–38.5 的 Markdown 文件链接，没有发现缺失的已链接文件；不代表所有文字提及的路径或历史实验均已验收。

主要数据：

| 文件 | 内容 |
|---|---|
| recomputed.json | 真 RMST、AUC、首次/末次达标、灾难、地图配对 bootstrap、独立最短路和剂量配对检查 |
| structure_checks.json | 作者解释器转换核验与 204 个 C_g 少一步的证据 |
| probe_results.json | 18 个历史重放和 18 个零步费/单位终奖探针，含完整 checkpoint 和命令 |
| qrm_reward_isolation.json | 保留终奖只改步费，以及保留步费只改终奖 |
| safe_sparse_probe.json | 零步费/单位终奖探针的 safe 配对 |
| unchanged_optimal_path_subset.json | 最优任务长度未改变的地图子集；事后敏感性分析 |
| older_statistics.json | Step 36 的 3.20×、Step 38 的 3.79×重算及逐臂删失计数 |
| external_repos.json | 五个第三方仓库的 commit、remote、工作区状态 |
| current_snapshot.json | 本次审计时 491 个直接相关文件 SHA256；不是历史运行 manifest |
| static_checks.json / file_inventory.txt | 静态扫描范围和结果 |

`probes.py` 的列表位置映射训练种子，是根据原运行器的有序 `executor.map` 及 seed=0..14 生成顺序恢复的；18 个重放均与该恢复一致。
