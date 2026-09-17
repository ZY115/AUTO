# 本项目安装的一切：软件、环境、依赖、路径、启动方式

安装日期 **2026-09-10**。机器 Apple M1 Pro / 10 核 / 32 GB / macOS arm64 (Darwin 24.3.0)。

本文件记录**为本项目安装的每一项**，包括系统层改动、可逆性，以及如何完整回退。
逐包版本清单另存于 `docs/environment/venv-rl-freeze.txt` 与 `venv-rm-freeze.txt`。

---

## 0. 首要原则与验证

**没有改动 miniconda base，原有流水线零影响。** Stage 8 继续用 `python3`
（`/Users/yuhang/miniconda3/bin/python3`，3.12.9）和 `clang++`。

安装前后逐位一致的验证（三臂轨迹哈希）：

```
history    11680738927923031754
automaton  13939158660656840691
merged     14057955382036670535
```

59 项单元测试安装前后均全部通过。

---

## 1. 系统层（Homebrew）——**唯一在项目目录之外的改动**

主动安装 `python@3.10`，因为 ILASP 的官方二进制动态链接它的 framework。
Homebrew 连带处理了 6 个依赖，**其中两个是升级而非新装**：

| formula | 版本 | 性质 | 说明 |
| --- | --- | --- | --- |
| **python@3.10** | 3.10.21 | 新装，主动请求 | ILASP 的硬依赖 |
| gdbm | 1.26 | 新装，依赖 | |
| mpdecimal | 4.0.1 | 新装，依赖 | |
| sqlite | 3.53.4 | 新装，依赖 | |
| **openssl@3** | 3.6.1 → **3.6.4** | **升级** | 旧版 3.6.1 仍在 Cellar，可回滚 |
| **ca-certificates** | 2025-12-02 → **2026-08-13** | **升级** | 旧版仍在 Cellar，可回滚 |
| **coreutils** | 9.x | 新装，主动请求（2026-09-11） | ISA 调用 GNU `timeout`，macOS 没有 |
| **graphviz** | 16.0.0 | 新装，主动请求（2026-09-11） | ISA 每学到一个自动机就画一张图，需要 `dot` |
| xz / readline | 已存在（2026-03-02） | 未改动 | |

**需要你知道的风险**：`openssl@3` 与 `ca-certificates` 是系统级共享库，升级会影响本机
其他链接 Homebrew openssl 的软件。两者的旧版本目录都还在 `$(brew --prefix)/Cellar` 下，
必要时可 `brew switch` 回去。

安装位置：`$(brew --prefix)/Cellar/`，即 `/opt/homebrew/Cellar/`。

---

## 2. ILASP（ISA 的归纳求解器）

| 项 | 值 |
| --- | --- |
| 版本 | 4.4.1（构建于 2024-11-11），**macOS-M1 原生，不需要 Rosetta** |
| 下载 | `https://github.com/ilaspltd/ILASP-releases/releases/download/v4.4.1/ILASP-4.4.1-macOS-M1.tar.gz` |
| 压缩包 sha256 | `7e9fd1ccbafa3241558c1f70a868b1c3e141de52dbdc76d26396b70f492cd75c` |
| 二进制 sha256 | `08320bd4c20136bcb8803d384e65faedb99371e0960287819ef3257f043406be` |
| 安装路径 | `external/bin/ILASP`（另含 `ClingoLicense.txt`、`examples/`） |
| 副本 | `external/induction-subgoal-automata-rl/src/bin/ILASP`（ISA 期望的位置） |
| 配套 | `external/bin/timeout` → `$(brew --prefix coreutils)/bin/gtimeout` 的符号链接。ISA 的求解器封装无条件调用 `timeout`，macOS 自带的工具里没有它。这是补一个缺失的 GNU 工具，**没有改动 ISA 的任何源码或归纳算法** |
| 运行依赖 | Homebrew `python@3.10` 的 framework，见第 1 节 |

**启动**：

```bash
external/bin/ILASP --version=4 你的任务.las
```

**已验证**：跑通一个含 `#modeh` / `#modeb` / `#pos` / `#neg` 的完整学习任务，
输出预处理、假设空间生成、冲突分析、反例搜索各阶段耗时。注意它没有 `--version` 选项，
`--version=4` 是指定 ILASP 算法版本。

---

## 3. 虚拟环境一：`.venv-rl`（现代 RL 栈）

| 项 | 值 |
| --- | --- |
| 路径 | `/Users/yuhang/Downloads/why TL/.venv-rl` |
| Python | 3.12.9（来自 miniconda 的 `python3 -m venv`） |
| 大小 | 826 MB |
| 完整清单 | `docs/environment/venv-rl-freeze.txt`（32 个包） |

创建与安装命令：

```bash
python3 -m venv .venv-rl
.venv-rl/bin/python -m pip install --upgrade pip
.venv-rl/bin/python -m pip install torch gymnasium minigrid stable-baselines3 aalpy clingo pytest
```

显式安装的顶层包：

| 包 | 版本 | 用途 |
| --- | --- | --- |
| torch | 2.14.0 | MPS 后端可用 |
| gymnasium | 1.3.0 | 标准环境接口 |
| minigrid | 3.1.0 | 82 个已注册环境 |
| stable_baselines3 | 2.9.0 | PPO/DQN 基线 |
| aalpy | 1.6.2 | 被动/主动自动机学习，做 RPNI 独立对照 |
| clingo | 5.8.2 | ASP 求解器 |
| pytest | 9.1.1 | 测试运行器 |
| numpy | 2.5.3 | （由上面带入） |

**启动**：

```bash
.venv-rl/bin/python 你的脚本.py
```

**已验证的实测吞吐**：

| 项目 | 实测 |
| --- | --- |
| MiniGrid DoorKey-6x6 单进程 | 13,479 步/秒 |
| PPO（MlpPolicy, 4 并行环境, n_steps=128） | 4,096 步用时 2.8 秒，约 1,460 步/秒含学习 |
| 由此推算 | 100 万步一次运行约 11 分钟 |
| AALpy `run_RPNI` | 5 条样本得到 3 状态 DFA |
| clingo | 求解 `a. b:-a.` 返回 `b` |

神经网络吞吐（`torch`，批 256）：

| 网络 | CPU | MPS |
| --- | --- | --- |
| MLP 52→8 | 207,616 样本/秒 | 201,238 样本/秒 |
| MiniGrid 卷积 7×7×3 | 10,695 样本/秒 | 146,877 样本/秒 |

---

## 4. 虚拟环境二：`.venv-rm`（奖励机公开域，2020 年代代码）

| 项 | 值 |
| --- | --- |
| 路径 | `/Users/yuhang/Downloads/why TL/.venv-rm` |
| Python | **3.10.21**（来自第 1 节的 Homebrew python@3.10） |
| 大小 | 976 MB |
| 完整清单 | `docs/environment/venv-rm-freeze.txt`（31 个包） |

**为什么必须分开**：`gym-subgoal-automata` 的代码使用 numpy 1.24 已删除的 `np.float`、
`np.bool8`；而 `.venv-rl` 里的 torch 2.14 与 SB3 2.9 期望 numpy 2。一个环境装不下。

创建与安装命令：

```bash
$(brew --prefix python@3.10)/bin/python3.10 -m venv .venv-rm
.venv-rm/bin/python -m pip install --upgrade pip
.venv-rm/bin/python -m pip install "numpy<2" "gym==0.26.2" graphviz pygame
.venv-rm/bin/python -m pip install --no-deps "git+https://github.com/ertsiger/gym-subgoal-automata.git"
.venv-rm/bin/python -m pip install torch scipy matplotlib pandas tqdm
```

`--no-deps` 是必须的：该包钉死 `gym==0.17.2`、`numpy==1.18.1`、`pygame==1.9.6`、
`graphviz==0.14`，这些在 Python 3.10 / arm64 上装不上。它的代码与 gym 0.26 + numpy 1.26 兼容
（除第 5 节那一处）。

| 包 | 版本 |
| --- | --- |
| numpy | 1.26.4 |
| gym | 0.26.2 |
| gym-subgoal-automata | 0.0.2（git 源码安装） |
| graphviz | 0.21 |
| pygame | 2.6.1 |
| torch / scipy / matplotlib / pandas / tqdm | 2.14.0 / 1.15.3 / 3.10.9 / 2.3.3 / 4.70.0 |

**启动**：

```bash
.venv-rm/bin/python 你的脚本.py
```

环境必须先 `import gym_subgoal_automata` 才会注册，并建议 `disable_env_checker=True`
（gym 0.26 的检查器自身也用了 `np.bool8`）：

```python
import gym, gym_subgoal_automata
e = gym.make('OfficeWorldDeliverCoffee-v0',
             params={"generation": "random", "environment_seed": 0},
             disable_env_checker=True)
automaton = e.unwrapped.get_automaton()   # 自带真实自动机，可直接当 oracle
```

**已验证的 42 个注册任务中的代表**：

| 环境 | 步/秒 | 真实自动机 | 观测 |
| --- | --- | --- | --- |
| OfficeWorldDeliverCoffee-v0 | 196,693 | 4 状态 | 表格 |
| OfficeWorldPatrolABC-v0 | 210,233 | 5 状态 | 表格 |
| CraftWorldMakeAxe-v0 | 248,915 | 7 状态 | 表格 |
| WaterWorldRedGreen-v0 | 11,699 | 3 状态 | 52 维连续 |
| WaterWorldRedGreenAndBlueCyan-v0 | 14,979 | 9 状态 | 52 维连续 |

---

## 5. 唯一一处对第三方源码的修改

| 项 | 值 |
| --- | --- |
| 文件 | `.venv-rm/lib/python3.10/site-packages/gym_subgoal_automata/envs/waterworld/waterworld_env.py` |
| 改动 | 三处 `np.float` → `float`（第 46、47、334 行） |
| 原因 | numpy 1.24 删除了 `np.float` 别名；不改则 WaterWorld 全系列无法实例化 |
| 备份 | 同目录 `waterworld_env.py.orig` |
| 补丁 | `docs/environment/waterworld_numpy2.patch` |

其余第三方代码一律未改。

---

## 6. 克隆的代码仓库

| 仓库 | 路径 | commit | 提交日期 | 大小 |
| --- | --- | --- | --- | --- |
| `RodrigoToroIcarte/reward_machines` | `external/reward_machines` | `3116c2fb6db130a3865332afa760f6ab3c3575d8` | 2021-04-05 | **2.8 GB** |
| `ertsiger/induction-subgoal-automata-rl` | `external/induction-subgoal-automata-rl` | `dc498724bb1b46cad91f8240973367bd7f326341` | 2023-08-15 | 8.2 MB |
| `braraki/logical-options-framework` | `external/logical-options-framework` | `5c0dac21491e9e68c8726ad6c24eea7e20830c8c` | 2021-06-07 | 352 KB |

均为 `--depth 1` 浅克隆。

**reward_machines 的 2.8 GB 里有 1.8 GB 是仓库自带的历史结果**（`external/reward_machines/results`），
不需要的话可以直接删除，不影响其中的 CRM/QRM 实现。它的 `requirements.txt` 是一份完整的
anaconda 环境导出（含 `anaconda-navigator` 等），不可直接使用，**我没有按它安装任何东西**。

**ISA 仓库**的 `requirements.txt` 钉在 torch 1.5.0 / numpy 1.18.1 / scipy 1.4.1（2020 年），
**我也没有按它安装**，而是在 `.venv-rm` 里用现代版本。已验证
`ilasp.solver.ilasp_solver` 与 `ilasp.generator.ilasp_task_generator` 可导入。
它的入口是 `src/run_isa.py`，尚未实跑。

---

## 7. 本项目新建的目录一览

```
/Users/yuhang/Downloads/why TL/
├── .venv-rl/                     826 MB   现代 RL 栈
├── .venv-rm/                     976 MB   奖励机公开域
├── external/
│   ├── bin/ILASP                 6.2 MB   ILASP 4.4.1 M1
│   ├── reward_machines/          2.8 GB   （1.8 GB 是可删的历史结果）
│   └── induction-subgoal-automata-rl/  8.2 MB  （含 src/bin/ILASP 副本）
├── docs/environment/
│   ├── venv-rl-freeze.txt
│   ├── venv-rm-freeze.txt
│   └── waterworld_numpy2.patch
└── ENVIRONMENT.md                本文件
```

磁盘占用合计约 4.6 GB。安装后根分区可用 408 GB。

`/tmp` 下有下载中间产物（`ilasp.tar.gz` 等），重启即清，不需要管理。

---

## 8. 完整回退

```bash
# 1. 删掉两个虚拟环境与外部代码（项目目录内，无副作用）
rm -rf ".venv-rl" ".venv-rm" external docs/environment ENVIRONMENT.md

# 2. 卸载系统层的 python@3.10 及其新装依赖
brew uninstall python@3.10
brew uninstall gdbm mpdecimal sqlite coreutils graphviz   # 若无其他软件依赖它们

# 3. 如需把被升级的共享库回滚（旧版本仍在 Cellar）
brew switch openssl@3 3.6.1
brew switch ca-certificates 2025-12-02
```

miniconda base 与 Stage 8 流水线不受上述任何一步影响。

---

## 9. 一句话启动表

| 要做的事 | 命令 |
| --- | --- |
| 原有 Stage 8 实验 | `python3 progressive_task_discovery/stage8_compression/src/…` |
| MiniGrid / SB3 / AALpy | `.venv-rl/bin/python 脚本.py` |
| OfficeWorld / CraftWorld / WaterWorld | `.venv-rm/bin/python 脚本.py` |
| ILASP 归纳 | `external/bin/ILASP --version=4 任务.las` |
| ISA 验收等级 2 自检 | `.venv-rm/bin/python external/isa_level2_check.py` |
| ISA 完整学习运行（等级 3，未跑） | `cd external/induction-subgoal-automata-rl && ../../.venv-rm/bin/python src/run_isa.py …` |


---

## 10. ISA 三级验收：全部通过（2026-09-11）

| 等级 | 内容 | 状态 |
| --- | --- | --- |
| 1 | 求解器跑通自带示例 | **通过**。ILASP 4.4.1 ARM64 跑通 `pets.las`，返回码 0 |
| 2 | ISA 生成的一个归纳任务跑通 | **通过**。`external/isa_level2_check.py` |
| 3 | 原生领域完整学习运行 | **通过**。见下 |

### 等级 3 的证据

命令（`src/config/examples/officeworld/coffee.json` 的官方配置，只缩小了规模）：

```bash
cd external/induction-subgoal-automata-rl
PATH="../../external/bin:$PATH" ../../.venv-rm/bin/python src/run_isa.py qrm 配置.json
```

规模改动：`num_tasks` 50→5，`num_episodes` 10000→3000，`max_episode_length` 250→150，
`ilasp_timeout` 7200→300。**其余参数一律沿用官方示例**，包括
`symmetry_breaking_method=bfs-alternative`、`learn_acyclic_graph=true`、`ilasp_version=2`。

结果：**退出码 0**，生成并求解 12 个归纳任务，画出 12 张自动机图。

学到的自动机（`solution-9.txt`）与环境自带的真实自动机结构一致：

| | 边 |
| --- | --- |
| ISA 学到 | `u0 --f--> u1`，`u1 --g--> u_acc`，`u0 --(n ∧ ¬f)--> u_rej` |
| 环境真值 | `u0 --f&¬g--> u1`，`u1 --g&¬n--> u_acc`，`u0 --n&¬f&¬g--> u_rej`，另有 `u0 --f&g-->` 与 `u1 --n--> u_rej` 两条捷径/死局边 |

即：先拿咖啡（f）再去办公室（g），没拿咖啡时踩到盆栽（n）判负。ISA 学到的是行为上足够、
但比真值略粗的版本（少两条边）。策略侧：5 个任务里 4 个贪心评估稳定拿到奖励 1.0
（步数 36 / 12 / 20 / 18），1 个停在 0.0。

### 为跑通等级 3 所做的适配，全部在此列明

**两处补缺失的系统工具，没有改任何源码**：GNU `timeout`（`external/bin/timeout` →
`gtimeout`），以及 graphviz 的 `dot`。

**四处源码改动，全部在 `docs/environment/isa_compat.patch`，共 2 个文件 11 增 7 删**：

| 改动 | 文件 | 原因 |
| --- | --- | --- |
| `dtype=np.int` → `dtype=int`（2 处）、`np.bool` → `bool`（1 处） | `isa_base_algorithm.py` | numpy 1.24 删除了这些别名 |
| 三处 `gym.make` 加 `disable_env_checker=True` | `run_isa.py` | gym 0.26 的被动检查器在 ISA 所用的 0.17 里并不存在，而检查器自身用了同样被删的 `np.bool8` |
| `binary_folder_name` 由硬编码的 `src/bin` 改为 `None`（可用 `ISA_BINARY_FOLDER` 覆盖） | `run_isa.py` | **ILASP 4.4.1 移除了 `--clingo`**，传目录会让它把 clingo 路径当成 ASP 文件去读，每次归纳都失败 |

**这四处都不触碰归纳算法**：不改任务生成、不改假设空间、不改例子编码、不改解析。
改的是 numpy 别名、gym 的包装器开关、以及求解器的调用方式。

### 两条版本兼容性观察

- ISA 向 ILASP 传 `--simple`。4.4.1 提示该标志已移除，**且 simple 表示现在是默认行为**，语义不变。
- ISA 向 ILASP 传 `--clingo`。4.4.1 已移除，且**不是良性警告**：它会把后面的路径当文件读。
  这是唯一一个会让 ISA 静默失效的不兼容——归纳任务照常生成，但每次求解都失败。

ISA 官方实现用的是 Python 3.6.9 / ILASP 3.6.0 / clingo 5.4.0；这里是 Python 3.10.21 /
ILASP 4.4.1 / ILASP 自带 clingo。**这是"新求解器 + 旧代码"的复现，不是原版复现。**


---

## 11. LOF 三级验收：全部通过，且**一行源码都没改**（2026-09-11）

| 等级 | 内容 | 状态 |
| --- | --- | --- |
| 1 | 依赖齐全、`simulator` 包可导入 | **通过** |
| 2 | 五种方法各自能在官方 Delivery 领域上短跑 | **通过**，`external/lof_level2_check.py` |
| 3 | 官方 satisfaction 实验完整跑一轮 | **通过**，`external/lof_level3_run.py` |

依赖只缺一个 `celluloid`，已装进 `.venv-rm`。仓库的 `environment.yml` 要 Python 3.8.1 /
numpy 1.18.1，实际用 Python 3.10.21 / numpy 1.26.4 直接跑通，**没有任何兼容性补丁**。
与 ISA 形成对比：ISA 需要四处源码适配，LOF 零处。

**等级 3 的结果**（调用 LOF 自己的 `run_experiment`，四个任务 × 五种方法 × 官方的 1,601 回合，
用时 548 秒，退出码 0，产出 20 个 `.npz`）：

末段成功率（最后 10 个记录点）：

| 方法 | composite | sequential | OR | IF |
| --- | --- | --- | --- | --- |
| LOF | 1.00 | 1.00 | 1.00 | 1.00 |
| FSA options | 1.00 | 1.00 | 1.00 | 1.00 |
| Greedy options | 1.00 | 1.00 | 1.00 | 1.00 |
| Reward Machine | 1.00 | 1.00 | 1.00 | 1.00 |
| Flat options | 0.00 | 0.10 | 0.40 | 0.50 |

首次连续三个记录点成功率为 1.0 的回合数（越小越快，— 表示 1,601 回合内没达到）：

| 方法 | composite | sequential | OR | IF |
| --- | --- | --- | --- | --- |
| FSA options | 260 | 320 | 300 | **120** |
| LOF | 300 | 560 | **280** | 220 |
| Greedy options | 420 | 500 | 500 | 200 |
| Reward Machine | 860 | 1400 | 500 | 740 |
| Flat options | — | — | 920 | — |

（结果文件里的 `steps` 是累计计数不是单回合步数，故未列出，以免误读。）

**定位上的硬约束**：LOF **被给定任务规格，它不学习规则**。所以它属于"已知规则"那一列，
只能与本项目的 `--goal-select value_ranked` 这类 oracle 臂比较，**不能与必须学习规则的臂
同列比优劣**。这一点在任何引用这批数字的地方都必须写明。

**启动**：

```bash
cd external
PYTHONPATH="$PWD/logical-options-framework" ../.venv-rm/bin/python lof_level2_check.py
PYTHONPATH="$PWD/logical-options-framework" ../.venv-rm/bin/python lof_level3_run.py
```

---

## `.venv-hrm`：HRM 原生 CraftWorld（第三十九步）

**结论：整套原生栈在本机 macOS ARM 上跑通了**，不需要换 Linux。
环境创建、层级读取、两个算法臂（`ihsa-hrl` 与 `ihsa-crm`）的短训练都完成。

### 仓库

| 路径 | 来源 |
|---|---|
| `external/hrm-formalism-envs` | `ertsiger/hrm-formalism-envs`，环境与层级形式化 |
| `external/hrm-learning` | `ertsiger/hrm-learning`，算法与官方配置 |
| `external/hrm-minigrid` | `ertsiger/hrm-minigrid`，作者的 gym-minigrid 分支，带 11 个 CraftWorld 物体类 |

### 创建与安装

```bash
python3.10 -m venv .venv-hrm
.venv-hrm/bin/python -m pip install --upgrade pip setuptools wheel
.venv-hrm/bin/python -m pip install "gym==0.15.3"
.venv-hrm/bin/python -m pip install -e external/hrm-minigrid
.venv-hrm/bin/python -m pip install --no-deps -e external/hrm-formalism-envs
.venv-hrm/bin/python -m pip install matplotlib pygame torch pandas tqdm xlsxwriter requests
.venv-hrm/bin/python -m pip install "numpy==1.23.5"     # 必须放最后，见下
```

Python 3.10.21。完整冻结见 `docs/environment/venv-hrm-freeze.txt`。

### 偏离作者 requirements 的地方（都是必须的）

| 作者要求 | 实际安装 | 原因 |
|---|---|---|
| `torch==1.10.0+cpu` | torch 2.14.0 | ARM macOS 没有 1.10 的轮子；API 未漂移，CRM 网络前向正常 |
| `numpy==1.21.3` | **numpy 1.23.5** | 1.21 在 3.10/ARM 上没有轮子；**必须 < 1.24**，否则 `np.int`/`np.bool` 别名被移除，源码有 5 处用到 |
| `pygame==1.9.6` | pygame 2.6.1 | 1.9.6 在 ARM 上不能构建；只被 WaterWorld 的 import 链用到，CraftWorld 用不着 |
| `matplotlib==3.4.3` | matplotlib 3.10.9 | 只用于渲染与画自动机图 |
| `gym~=0.15.3` | **gym 0.15.3** | 保持原版，它在 3.10 上能从源码构建 |

装 numpy 要放最后：装 torch 会把 numpy 顶到 2.x。

### 必须打的补丁

`docs/environment/hrm_learning_py310_compat.patch`——一行：
`src/ilasp/ilasp_common.py` 里的 `os.environ.putenv(...)` 改成 `os.environ[...] = ...`。
`os.environ.putenv` 在 Python 3.9 被移除。启动器无条件调用它，即使 `training_mode: handcrafted`
根本用不到 ILASP。

### 启动方式

```bash
cd external/hrm-learning
PYTHONPATH=src ../../.venv-hrm/bin/python src/run_algorithm.py <配置文件绝对路径>
```

配置文件里的 `folder_name` 与 `checkpoint_folder` 要改到仓库外，否则会往仓库里写。
`algorithm` 取 `ihsa-hrl`（= 我们的 Y11）或 `ihsa-crm`（= QRM）。

### 已验证

- `gym.make("CraftWorldBook/BookAndQuill/Cake-v0", params=...)`，两种网格都能创建；
  子自动机数 3 / 5 / 6，与用桩件做的结构审计逐项一致。
- 随机策略 5000 步：open_plan 无岩浆 0 次进入拒绝态；four_rooms 带岩浆 49 幕全部死于岩浆。
- `ihsa-hrl` 跑满 400 幕，`ihsa-crm` 跑满 120 幕，都无异常退出。
- `MinigridCRMDQN((3,7,7), 22, 3, False)` 前向正常，144,147 个参数（单一共享网络）。

### 注意

`four_rooms` 的网格尺寸走 `size` 键，不是 `width`/`height`；且必须是 ≥11 的奇数。
只给 `width`/`height` 会报 `The grid for four rooms must be at least 11x11`。

### 性能实测（第 39 步定规模用）

| 臂 / 条件 | 速度 | 备注 |
|---|---|---|
| 表格，有岩浆 | 54 幕/秒 | 幕短，死得快 |
| 表格，无岩浆 | 5–10 幕/秒 | 幕长到 1000 步上限 |
| 神经（`full_obs`），训练开启 | **0.54 幕/秒** | 比表格慢约 100 倍 |
| 神经 + Apple MPS | **0.40 幕/秒** | **比 CPU 更慢，别用** |

规模不是线性的：cake 带岩浆的表格臂，4 万幕只用 482 秒，而 1 万幕是 185 秒——
**膨胀系数 0.65**，学会之后每幕反而更便宜。

`er_start_size = 100000` 意味着短跑里**一次梯度更新都不会发生**，任何低于约 1000 幕的计时
只测了走路没测训练，不能用来推算。

**MPS 补丁**：`learning_algorithm.py` 的设备选择原本只认 CUDA，在 Apple 芯片上会静默退回 CPU。
已改成也认 MPS（默认仍是 CPU，需 `use_gpu: true` 才生效）。实测更慢，保留只是为了不再重复试。
