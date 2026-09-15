# AUTO — 任务自动机的两种用法对照实验

一个强化学习对照实验的完整交接仓库。克隆下来就能开始跑，不需要其他任何材料。

---

## 这个实验在比什么

有一类方法会给智能体一张「任务流程图」：先做 A，再做 B，然后 C。
我们比较**同一张流程图的两种用法**，两者只差一件事：

| 臂 | 做法 | 代码来源 |
|---|---|---|
| **Y11** | 「去工作台」这个技能**只学一套**，所有阶段共用 | `ihsa-hrl`，原作者实现，未改动 |
| **Y10** | 每个阶段**各学一套**「去工作台」 | `ihsa-hrl-perstate`，本仓库新增 |

要检验两个假设：

- **H1**：Y11 比 Y10 学得快。
- **H2**：地图上有「踩上去就死」的危险格时，这个差距会更大。

**H1 已经复现，而且随任务变长而变大**（见 [已有结果](#已有结果)）。
**H2 还没测出来**——不是反向结果，是训练幕数不够，两臂都还没离开地板。
**H2 就是这台新机器要解决的问题。**

---

## 从哪开始

| 你是 | 看这里 |
|---|---|
| 要跑实验的工程师 | [`docs/SETUP.md`](docs/SETUP.md) → [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) |
| **想知道立项那个问题答到哪一步** | [**`docs/ORIGINAL_QUESTION.md`**](docs/ORIGINAL_QUESTION.md) |
| 想先了解背景 | [`docs/BACKGROUND.md`](docs/BACKGROUND.md) |
| 卡住了 | [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) |
| 要交回结果 | [`results/README.md`](results/README.md) |
| 看「完整历史 vs 任务状态抽象」pilot | [`docs/MEMORY_COMPRESSION_PILOT.md`](docs/MEMORY_COMPRESSION_PILOT.md) |

最短路径，三条命令：

```bash
bash setup.sh /opt/hrm
export HRM_LEARNING=/opt/hrm/hrm-learning HRM_PYTHON=/opt/hrm/venv/bin/python
$HRM_PYTHON tools/verify.py
```

`verify.py` 七项全过之后才开始跑实验。**不要跳过它**——其中两项验证的是
「Y10 这个改动只关掉了共享、没顺手改别的」，过不了这两项，后面所有对比都无效。

---

## 仓库结构

```
setup.sh            一键搭环境：克隆三个上游仓库、打补丁、装依赖
patches/            对上游代码的两个补丁
src/                本项目新增的两个算法实现（Y10 的表格版与神经版）
tools/              verify / run_grid / analyze_grid / 两个深度等价性检查
docs/               搭建、实验、背景、排错
results/
  reference/        已有的结果，供对照
  runs/             ← 你把跑出来的 .jsonl 放这里
```

三个上游仓库**不在本仓库里**，`setup.sh` 会按锁定的提交号克隆：

| 仓库 | 提交 |
|---|---|
| `ertsiger/hrm-formalism-envs` | `929605f` |
| `ertsiger/hrm-learning` | `23d0e25` |
| `ertsiger/hrm-minigrid` | `d0e6763` |

---

## 已有结果

在一台 M1 Pro 上跑的表格版、10,000 幕、作者协议、无岩浆条件：

| 任务 | 任务链条长度 | Y11 | Y10 | 差（95% 区间） |
|---|---|---|---|---|
| book | 5 步 | 1.000 | 0.911 | +0.089 [+0.027, +0.172] |
| book-and-quill | 8 步 | 1.000 | 0.153 | +0.847 [+0.743, +0.929] |
| cake | 9 步 | 1.000 | 0.088 | +0.912 [+0.841, +0.969] |

有岩浆条件下，较深的两个任务上**两臂都是 0.000**。这就是要跑更多幕数的原因。

原始数据在 `results/reference/macos_tabular_10k.jsonl`，48 次运行。
随时可以自己复算：

```bash
python3 tools/analyze_grid.py results/reference/macos_tabular_10k.jsonl
```

如果你跑出来的无岩浆那三行和上表方向一致，说明环境和改动都正确。

---

## 工作流

```
对方                                       我方
──────────────────────────────────────────────────────────
克隆仓库
bash setup.sh
tools/verify.py  ← 七项全过
tools/run_grid.py  ← 跑实验
结果写到 results/runs/*.jsonl
提交并推送                    ──────→   git pull
                                        tools/analyze_grid.py results/runs/*.jsonl
                                        分析、反馈、必要时调整下一轮
```

交回的格式与要求见 [`results/README.md`](results/README.md)。
核心一条：**交原始的 `.jsonl`，不要只交汇总表**。
本项目早期有两批数据只保留了汇总，逐种子的原始结果已经永久丢失，那部分结论至今无法复查。
