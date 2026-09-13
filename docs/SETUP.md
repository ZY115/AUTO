# 环境搭建

目标机器：带 CUDA 的 Linux。预计半小时。

---

## 一条命令

```bash
bash setup.sh /opt/hrm
```

参数是安装根目录，可以换成任何路径。脚本会：

1. 克隆三个上游仓库到锁定的提交号
2. 打两个补丁
3. 把 `src/` 里的两个新算法文件复制进去
4. 建 Python 3.10 虚拟环境
5. 按正确顺序装依赖
6. 自检并打印版本

装完之后：

```bash
export HRM_LEARNING=/opt/hrm/hrm-learning
export HRM_PYTHON=/opt/hrm/venv/bin/python
```

建议把这两行加进 `~/.bashrc`，后面所有命令都要用到。

---

## 依赖为什么这么装

上游的 `requirements.txt` 是 2021 年的，直接照装会失败。下面每一条偏离都是实测踩出来的：

| 上游要求 | 实际要装 | 原因 |
|---|---|---|
| `gym~=0.15.3` | **保持 0.15.3** | 能在 Python 3.10 上从源码构建。**不要升级**，0.2x 的 API 不兼容 |
| `numpy==1.21.3` | **1.23.5，且必须 `< 1.24`** | 上游源码有 5 处用 `np.int` / `np.bool`，numpy 1.24 移除了这些别名 |
| `torch==1.10.0+cpu` | CUDA 版任意近版 | 实测 2.14 无 API 漂移。装 CUDA 版，`cu121` 换成与你驱动匹配的 |
| `pygame==1.9.6` | 任意现代版 | 只被 WaterWorld 的 import 链用到，我们跑的 CraftWorld 不需要它 |
| `matplotlib==3.4.3` | 任意现代版 | 只用于画图 |

**装 numpy 必须放在最后。** 装 torch 会把 numpy 顶到 2.x，顺序错了就前功尽弃。
`setup.sh` 已经处理了这个顺序，手动装的话务必注意。

Python 用 **3.10**。上游的 conda 环境是 3.7，但 3.10 只需要两处兼容补丁，都在 `patches/` 里。

如果 `python3.10` 不在 PATH 里：

```bash
PYBIN=/usr/bin/python3.10 bash setup.sh /opt/hrm
```

---

## 两个补丁改了什么

### `patches/01-hrm-formalism-envs.patch`

给 CraftWorld 环境加了一个 `neutralize_deadends` 选项。**这是整个实验设计的关键**，
原因见 [EXPERIMENTS.md 的「风险对照」一节](EXPERIMENTS.md#风险对照是构造出来的不是开关)。

### `patches/02-hrm-learning.patch`

三处改动：

1. **`src/ilasp/ilasp_common.py`** — `os.environ.putenv(...)` 改成 `os.environ[...] = ...`。
   `putenv` 在 Python 3.9 被移除。启动器无条件调用它，哪怕我们跑的
   `training_mode: handcrafted` 根本用不到 ILASP。
2. **`src/run_algorithm.py`** — 注册新算法名 `ihsa-hrl-perstate`，
   按 `state_format` 自动分派到表格版或神经版。
3. **`src/reinforcement_learning/learning_algorithm.py`** — 设备选择原本只认 CUDA。
   加了 Apple MPS 分支。**在 CUDA 机器上这条没有任何影响**，保留只是为了记录：
   我们在 Apple 芯片上实测过 MPS，比 CPU 还慢，别再试第二次。

补丁是用 `git diff` 生成的，`git apply` 可以直接用。想确认补丁与上游代码匹配：

```bash
git -C /opt/hrm/hrm-learning apply --check -R /path/to/AUTO/patches/02-hrm-learning.patch
```

反向应用能通过，就说明补丁已正确落地。

---

## 新增的两个文件

`src/` 里的两个文件会被 `setup.sh` 复制到 `hrm-learning/src/reinforcement_learning/`：

- `ihsa_hrl_tabular_perstate_algorithm.py` — Y10 的表格版
- `ihsa_hrl_dqn_perstate_algorithm.py` — Y10 的神经版

两个文件开头的注释写了设计取舍，值得读一遍，尤其是神经版那个关于回放缓冲的选择。

---

## 验证

```bash
$HRM_PYTHON tools/verify.py
```

七项检查，每一项都对应一个真实踩过的坑：

| # | 检查 | 失败意味着 |
|---|---|---|
| 1 | `HRM_LEARNING` 指向有效检出 | 环境变量没设或路径错 |
| 2 | 三个任务能创建，子自动机数 3 / 5 / 6 | 环境包或 minigrid 装错了 |
| 3 | safe 与 lava 两格的网格逐项相同 | 补丁 01 没生效 |
| 4 | **表格：Y10 折叠后与 Y11 逐位相同** | **Y10 改动了不该改的东西** |
| 5 | 表格：不折叠时与 Y11 不同 | Y10 改动根本没生效 |
| 6 | **神经：同第 4 项** | 同上 |
| 7 | CUDA 可用 | 驱动或 torch 版本 |

**第 4 和第 6 项是全流程最重要的两项。** 它们把 Y10 的「阶段数」强行压成 1，
这时它必须和原版 Y11 产生**完全相同的日志**。相同，就证明这个改动只关掉了跨阶段共享、
没有顺手改别的；不同，后面所有对比都是被污染的。

想看更细的（每步采样的子目标个数、伪奖励取值、回放缓冲是否真的共享）：

```bash
cd /opt/hrm/hrm-learning
PYTHONPATH=src $HRM_PYTHON /path/to/AUTO/tools/y10_equivalence_checks.py
PYTHONPATH=src $HRM_PYTHON /path/to/AUTO/tools/y10_equivalence_checks_dqn.py
```

不是必须，但如果第 4 或第 6 项失败，这两个脚本能告诉你差在哪一环。
