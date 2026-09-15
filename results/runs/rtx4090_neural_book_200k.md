# rtx4090_neural_book_200k

阶段二神经版 pilot：在 `book` 的 10 张配对地图上，以原作者协议比较 Y11 与 Y10，
风险条件为 safe / lava，每格 3 个种子。

- 机器：NVIDIA GeForce RTX 4090 24 GB / Intel Core i9-14900KF（24 核、32 线程）/ Ubuntu 22.04.5 LTS
- 驱动：NVIDIA 560.35.03；PyTorch 2.5.1+cu121
- 实验运行时 AUTO 提交：`93ac3f618f293b93b7a31df7f9a7597b0f904683`
- 复核分析时 AUTO 提交：`adb13f2`
- 起止时间：2026-09-13 03:27:59–2026-09-15 03:32:11 PDT
- 总墙钟：约 48 小时 4 分钟
- 单 run 墙钟：135,064.6–173,052.0 秒
- 结果：12/12 个 run 成功，无失败或重试
- 原始结果 SHA-256：`496c82ebaf81a5a10656564686b97965eac46dac34e6be7ebdd611c9ea97a010`

## 运行命令

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/auto_nvidia_mps_pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/auto_nvidia_mps_log
export HRM_LEARNING=/home/jerry/AUTO/hrm/hrm-learning
export HRM_PYTHON=/home/jerry/AUTO/hrm/venv/bin/python

$HRM_PYTHON tools/run_grid.py \
    --state-format full_obs \
    --episodes 200000 \
    --seeds 3 \
    --tasks book \
    --workers 12 \
    --work /home/jerry/AUTO/scratch/rtx4090_neural_book_200k \
    --out results/runs/rtx4090_neural_book_200k.jsonl
```

没有传 `--er-start-size`，因此保留发表值 `100000`。12 个 full_obs run 均使用 CUDA，
并通过 NVIDIA MPS 并发执行。

## 运行说明

- 没有修改上游算法、实验超参数、随机种子或数据。
- 本地 `tools/run_grid.py` 当时加入了 `--device` 元数据、长任务优先调度和完成即落盘；
  这些只影响调度、结果元数据与写入顺序，不改变单 run 配置或学习过程。
- MPS 4 / 8 / 12 路短基准确认 12 路仍能扩展后，正式网格使用 12 workers。
- 三个较短的 lava/Y11 完成后，为利用空出的 CPU 资源，只调整了剩余进程的 CPU
  affinity：三个关键 safe/Y10 使用独占 P 核，随后其余进程也分配到互不重叠的物理核。
  进程未重启，配置、模型状态和随机数状态未改变。
- 没有失败或重跑。JSONL 中 12 条记录全部 `ok=true`。
- 本次结果生成后，仓库加入了 `budget_id`、累计训练环境步、密集评估和额外臂。
  本文件属于此前的 200,000 幕协议：记录没有这些新字段，曲线每点为
  `(episode, reward, steps)` 三字段。不得把它冒充或自动续跑成新版环境步预算协议。

## 数据完整性

```text
rows 12
unique tags 12
ok 12/12
task: book 12
arms: Y11 6 / Y10 6
risks: safe 6 / lava 6
seeds: 0, 1, 2 各 4
state_format: full_obs 12
episodes: 200000 12
device: cuda 12
每个 run 的地图实例数: 10
每条实例曲线的评估点数: 2000
三个关键源码哈希在 12 条记录中各只有一个取值
```

## 最新统计复核

以下输出由 `adb13f2` 的 `tools/analyze_grid.py` 产生。自助区间以 10 张地图为独立单位，
先在地图内合并 3 个种子；按 30 个“地图 × 种子”重抽的区间只作诊断。速度使用 RMST，
未达标实例按 200,000 幕计入，达标时刻记在十点窗口末端。

```text
--- full_obs / author ---
任务               风险          Y11      Y10         差          95% 区间（按地图成组）             （按地图×种子，仅诊断）
book             safe      0.991    0.990    +0.002 [-0.000, +0.003] 地图=10    [+0.000, +0.003] n=30
book             lava      0.991    0.986    +0.005 [+0.004, +0.007] 地图=10    [+0.003, +0.007] n=30  *

速度口径  达到 90% 所需幕数（RMST，删失按 200,000 幕计入）
任务               风险             Y11         Y10       倍数        节省幕数         删失
book             safe         1,277       2,513    1.97x       1,237  Y11 0/30 Y10 0/30   另一定义 (4.28x)
book             lava        10,940      39,813    3.64x      28,873  Y11 0/30 Y10 0/30   另一定义 (3.88x)

H2  岩浆是否放大共享的收益
任务                              末段成功率之差             RMST 节省量（加性）            RMST 倍数（乘性）
book             lava +0.005 − safe +0.002 = +0.003                +27,637 幕                 1.849x
```

### 这批数据能说什么

> 在原作者协议、20 万幕预算和 book 的 10 张配对地图上，Y11 在 lava 条件下的末段
> 成功率高于 Y10；Y11 在 safe 与 lava 下都更快达到 90%，且本批数据的加性和乘性
> RMST 交互均显示 lava 下的优势更大。结论严格限定于这个预算、任务、指标与尺度。

末段成功率在两格都接近上限，因此不能单独当效应量。现有 Y11 / Y10 对照同时改变
参数是否跨状态共享和经验如何作用于状态策略，不能把差异无条件命名为“纯参数共享收益”。

## verify.py

在提交前以同一上游环境重新运行：

```text
HRM 交接预检

  [通过] 1. HRM_LEARNING 指向一个 hrm-learning 检出  /home/jerry/AUTO/hrm/hrm-learning
  [通过] 2. 三个任务都能创建，子自动机数为 3 / 5 / 6  {'CraftWorldBook-v0': 3, 'CraftWorldBookAndQuill-v0': 5, 'CraftWorldCake-v0': 6}
  [通过] 3. safe 与 lava 两格的网格/起点/可观测集逐项相同
  [通过] 4. 表格：键钉成常数后与 Y11 逐位相同  39e1494b9874 vs 39e1494b9874
  [通过] 5. 表格：不钉死时与 Y11 不同（改动确实生效）  39e1494b9874 vs 01e626de2ef1
  [通过] 6. 神经：键钉成常数后与 Y11 逐位相同  b435052dc73d vs b435052dc73d
  [通过] 7. CUDA 可用  torch 2.5.1+cu121, NVIDIA GeForce RTX 4090

全部通过。可以开始跑网格。
```

## 上游检出状态

- `hrm-learning`：`23d0e25a367e1e8a559fa815d86696f0183ade43`
- `hrm-formalism-envs`：`929605f46e6ca8a4a3dad2a47ba7d805c2711093`
- `hrm-minigrid`：`d0e676374b011a8a4aeaa2e7141855c32a872ba3`

`git -C hrm/hrm-learning diff --stat`：

```text
 src/ilasp/ilasp_common.py                        |  5 ++++-
 src/reinforcement_learning/learning_algorithm.py |  9 ++++++++-
 src/run_algorithm.py                             | 10 ++++++++++
 3 files changed, 22 insertions(+), 2 deletions(-)
```

另有 setup 按设计复制的两个算法文件：

```text
?? src/reinforcement_learning/ihsa_hrl_dqn_perstate_algorithm.py
?? src/reinforcement_learning/ihsa_hrl_tabular_perstate_algorithm.py
```

`git -C hrm/hrm-formalism-envs diff --stat`：

```text
 .../envs/craftworld/craftworld_env.py | 31 +++++++++++++++++++++-
 1 file changed, 30 insertions(+), 1 deletion(-)
```

## Python 环境

环境与同机第一阶段共用同一个 venv。宿主 ROS 2 包通过 shell 的 Python 路径可见；
它们出现在完整 `pip freeze` 中，但未被本实验导入。项目相关冻结版本如下：

```text
certifi==2026.7.22
charset-normalizer==3.5.1
cloudpickle==1.2.2
contourpy==1.3.2
cycler==0.12.1
filelock==3.32.3
fonttools==4.65.0
fsspec==2026.7.0
future==1.0.0
gym==0.15.3
-e git+https://github.com/ertsiger/hrm-formalism-envs.git@929605f46e6ca8a4a3dad2a47ba7d805c2711093#egg=gym_hierarchical_subgoal_automata
-e git+https://github.com/ertsiger/hrm-minigrid.git@d0e676374b011a8a4aeaa2e7141855c32a872ba3#egg=gym_minigrid
idna==3.19
Jinja2==3.1.6
kiwisolver==1.5.1
MarkupSafe==3.0.3
matplotlib==3.10.9
mpmath==1.3.0
networkx==3.4.2
numpy==1.23.5
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvtx-cu12==12.1.105
packaging==26.3
pandas==2.3.3
pillow==12.3.0
pygame==2.6.1
pyglet==1.3.2
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pytz==2026.3.post1
requests==2.34.2
scipy==1.15.3
six==1.17.0
sympy==1.13.1
torch==2.5.1+cu121
tqdm==4.70.1
triton==3.1.0
typing_extensions==4.16.0
tzdata==2026.4
urllib3==2.7.0
xlsxwriter==3.2.9
```
