# rtx4090_tabular_200k

- 机器：NVIDIA GeForce RTX 4090 24 GB / Intel Core i9-14900KF（24 核、32 线程）/ Ubuntu 22.04.5 LTS
- 驱动：NVIDIA 560.35.03；PyTorch 2.5.1+cu121
- AUTO 提交：`da3b5a00f9dbf2b3515a15ead4754311df243097`
- 起止时间：2026-09-12 18:16:20–23:53:59 PDT
- 总墙钟：约 5 小时 38 分钟
- 结果：36/36 个 run 成功；36 个标签唯一；每个 run 含 10 个任务实例，每条曲线含 2,000 个评估点

## 运行命令

```bash
export HRM_LEARNING=/home/jerry/AUTO/hrm/hrm-learning
export HRM_PYTHON=/home/jerry/AUTO/hrm/venv/bin/python
$HRM_PYTHON tools/run_grid.py \
    --state-format tabular \
    --episodes 200000 \
    --seeds 3 \
    --workers 32 \
    --work /tmp/auto_hrm_tabular_200k \
    --out results/runs/rtx4090_tabular_200k.jsonl
```

## verify.py

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

`git -C hrm/hrm-learning diff --stat`：

```text
 src/ilasp/ilasp_common.py                        |  5 ++++-
 src/reinforcement_learning/learning_algorithm.py |  9 ++++++++-
 src/run_algorithm.py                             | 10 ++++++++++
 3 files changed, 22 insertions(+), 2 deletions(-)
```

另有 setup.sh 按设计复制的两个未跟踪算法文件：

```text
?? src/reinforcement_learning/ihsa_hrl_dqn_perstate_algorithm.py
?? src/reinforcement_learning/ihsa_hrl_tabular_perstate_algorithm.py
```

`git -C hrm/hrm-formalism-envs diff --stat`：

```text
 .../envs/craftworld/craftworld_env.py | 31 +++++++++++++++++++++-
 1 file changed, 30 insertions(+), 1 deletion(-)
```

## 运行说明

- 没有修改上游算法或实验超参数。
- 本地 `tools/run_grid.py` 将长耗时的 safe/deep 任务优先提交，并改为 run 完成后立即写入 JSONL；这只改变调度与落盘顺序，不改变任何单 run 的配置、种子或结果。
- 最初以 24 workers 启动过一次；在没有任何完整 run 产生时停止，并保留到独立诊断目录。正式 32-worker 运行使用全新的工作目录，不含旧日志。
- 正式运行没有失败或重试。
- 三个关键源码哈希在 36 条记录中各自只有一个取值。

## 分析摘要

分析口径在 2026-09-13 的复核中改过四处，下面的数字是改后的。旧版报告里的
`[+0.315, +0.614]` 与「速度 20.11x」都已作废，原因见本节末尾。

```text
--- tabular / author ---
任务               风险          Y11      Y10         差          95% 区间（按地图成组）             （按地图×种子，仅诊断）
book             safe      1.000    1.000    +0.000 [+0.000, +0.000] 地图=10    [+0.000, +0.000] n=30
book             lava      0.851    0.387    +0.464 [+0.218, +0.712] 地图=10    [+0.315, +0.614] n=30  *
book-and-quill   safe      1.000    1.000    +0.000 [+0.000, +0.000] 地图=10    [+0.000, +0.000] n=30
book-and-quill   lava      0.009    0.000    +0.009 [+0.007, +0.010] 地图=10    [+0.007, +0.010] n=30  *
cake             safe      1.000    1.000    +0.000 [+0.000, +0.000] 地图=10    [+0.000, +0.000] n=30
cake             lava      0.003    0.000    +0.003 [+0.002, +0.004] 地图=10    [+0.002, +0.004] n=30  *

速度口径  达到 90% 所需幕数（RMST，删失按 200,000 幕计入）
  达标时刻记在窗口末端（首次确认）；另一种定义的数字附在括号里，供判断敏感度
任务               风险             Y11         Y10       倍数        节省幕数         删失
book             safe         1,013       5,110    5.04x       4,097  Y11 0/30 Y10 0/30   另一定义 (37.15x)
book             lava       103,457     152,107    1.47x      48,650  Y11 0/30 Y10 18/30   另一定义 (1.48x)
book-and-quill   safe         1,103      22,577   20.46x      21,473  Y11 0/30 Y10 0/30   另一定义 (106.61x)
book-and-quill   lava       200,000     200,000    1.00x           0  Y11 30/30 Y10 30/30   另一定义 (1.00x)
cake             safe         1,123      29,543   26.30x      28,420  Y11 0/30 Y10 0/30   另一定义 (128.25x)
cake             lava       200,000     200,000    1.00x           0  Y11 30/30 Y10 30/30   另一定义 (1.00x)

H2  岩浆是否放大共享的收益
  三个统计量，三种尺度。它们可以给出不同方向而彼此不矛盾，
  所以每一条都要连尺度一起报，任何一条都不能单独当成 H2 的答案。
任务                              末段成功率之差             RMST 节省量（加性）            RMST 倍数（乘性）
book             lava +0.464 − safe +0.000 = +0.464                +44,553 幕                 0.292x
book-and-quill   lava +0.009 − safe +0.000 = +0.009                -21,473 幕                 0.049x  ← 两臂都在地板上，下列数字照报，但不要当效应量
cake             lava +0.003 − safe +0.000 = +0.003                -28,420 幕                 0.038x  ← 两臂都在地板上，下列数字照报，但不要当效应量

  读法：加性看「共享省下多少幕」，乘性看「共享让它快几倍」。
  book 上前者在岩浆下更大、后者在岩浆下更小，两句都对。
  但注意两条各自的限制：末段成功率在 safe 格封顶（两臂都 1.000，差只能是 0）；
  加性量受量纲支配（岩浆下两臂本来就要十万幕级）；乘性量受 tau 截断
  （岩浆下 Y11 已用掉大半预算，倍数上限只有 200,000/Y11 那么大）。
  能成立的结论是「交互方向依赖所选指标与尺度」，
  不能成立的是不加限定的「不可逆性普遍放大复用收益」。
```

### 这批数据能说什么

> 在原作者协议、20 万幕预算和 book 的 10 张配对地图上，Y11 在 safe 与 lava 下
> 均显示优势；**优势的大小及风险交互方向依赖所选指标与尺度**。现有对照
> **尚不能区分参数共享与跨状态经验传播各自的贡献**。

book-and-quill 与 cake 的 lava 条件下两臂都在地板上（0.009 / 0.003 对 0.000）。
数字照报，但那不是效应量；也不要据此推断「加预算有没有用」——曲线形状不足以支撑
那个推断，本报告的上一版做过，已撤回。

### 复核改了什么

1. **自助重抽单位**：从 30 个「地图 × 种子」改为 **10 张地图**（先在地图内对种子
   取平均）。一个任务只有 10 张不同的地图，把 30 个配对差当独立单位是伪重复。
   book/lava 因此从 `[+0.315, +0.614]` 放宽到 `[+0.218, +0.712]`。
2. **删失**：速度口径原本丢掉「从未达标」的实例——lava/book 下 Y10 有 **18/30**
   被丢，然后报出一个只在成功者身上成立的倍数。改用 RMST，删失按 20 万幕计入。
3. **达标时刻的定义**：原代码在找到连续十个评估点达标后，把时刻记在**窗口起点**，
   即用了随后 900 幕的信息却把时间记回窗口开始。现默认记**窗口末端**（首次确认，
   在线可得），另一种定义并排打出。这不是细节：book/safe 的 Y11 从 113 幕变成
   1,013 幕，共享的加速倍数从 37.15x 变成 **5.04x**。
4. **交互项按三种尺度分别报告**，不再只给一个数：末段成功率之差、RMST 节省量
   （加性）、RMST 倍数（乘性）。三者在 book 上分别是 `+0.464`、`+44,553 幕`、
   `0.292x`——**加性说岩浆下共享省得更多，乘性说岩浆下相对加速更小，两句都对**，
   因为加性与乘性交互本就是不同的量。

## pip freeze

环境由一个已有 Python 3.10 Conda 解释器创建 venv；宿主 ROS 2 环境通过 shell 的 Python 路径可见，故 `pip freeze` 同时列出 ROS 包。实验要求的关键版本为 `torch==2.5.1+cu121`、`numpy==1.23.5`、`gym==0.15.3`、`pygame==2.6.1`。

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

宿主 ROS 2 包也出现在原始 `pip freeze` 中，但未在上面的项目依赖摘录中重复列出；它们未被本实验导入。
