# RTX 4090：小网络 DQN 的 MPS 并发测速

这组探针用于回答：144,147 参数、batch size 32 的 `full_obs` DQN 在 RTX 4090 上，
能否通过 NVIDIA MPS 让多个小 kernel 的 run 真正并发。

- GPU：NVIDIA GeForce RTX 4090 24 GB
- CPU：Intel Core i9-14900KF，24 核 / 32 线程
- 驱动：NVIDIA 560.35.03
- PyTorch：2.5.1+cu121
- AUTO 基线：`93ac3f618f293b93b7a31df7f9a7597b0f904683`
- 任务：`book-and-quill`
- 协议：author；`full_obs`；safe / lava；Y11 / Y10
- 每个 run：200 幕、10 张地图；`er_start_size=3000`

`er_start_size` 特意从发表值 100,000 降到 3,000，使短跑跨过 replay 预热并实际执行
梯度更新。这只是吞吐探针，不是实验结果。CUDA、CPU、MPS4 用 seed 0；MPS8 用
seed 0–1；MPS12 用 seed 0–2。

## 汇总

时间单位均为秒。范围是同一风险条件下所有 arm / seed 的单 run 时间；“混合批次墙钟
代理”取该批最慢 run，未另计很小的父进程启动/收尾开销。“混合吞吐”只适用于本探针
固定的 1:1 safe/lava 组合。

| 模式 | 并发 run | safe 中位数（范围） | lava 中位数（范围） | 混合批次墙钟代理 | 混合吞吐（run/h） |
|---|---:|---:|---:|---:|---:|
| 普通 CUDA | 4 | 1,353.25（1,347.0–1,359.5） | 243.70（237.5–249.9） | 1,359.5 | 10.59 |
| CPU | 4 | 2,711.45（2,678.8–2,744.1） | 285.90（277.0–294.8） | 2,744.1 | 5.25 |
| MPS | 4 | 915.00（901.1–928.9） | 114.40（110.6–118.2） | 928.9 | 15.50 |
| MPS | 8 | 1,060.05（1,050.8–1,068.4） | 164.70（145.0–170.1） | 1,068.4 | 26.96 |
| MPS | 12 | 1,066.60（1,044.5–1,071.3） | 173.05（164.2–182.6） | 1,071.3 | 40.32 |

逐 run 原数见 [`timings.csv`](timings.csv)。

### 扩展效率

- 普通 CUDA4 → MPS4：混合吞吐 `10.59 → 15.50 run/h`，提高 **46.3%**。
- MPS4 → MPS8：`15.50 → 26.96 run/h`，提高 **73.9%**；并发翻倍时仍有明显扩展。
- MPS8 → MPS12：`26.96 → 40.32 run/h`，提高 **49.6%**；任务数增加 50%，
  批次墙钟几乎不变（1,068.4 对 1,071.3 秒）。
- MPS12 相对普通 CUDA4 的混合吞吐为 **3.81x**。

当时的 `nvidia-smi` 单点采样（不是时间平均）：普通 CUDA4 约 99% GPU、116 W、
3.1 GiB；MPS8 约 94%、143 W、5.4 GiB；MPS12 在 12 个客户端都存活时约 98%、
158 W、7.7 GiB。功耗没有接近 450 W 上限，因为瓶颈是 Python 环境步进与小 kernel
发射，不是大矩阵算力或显存容量。

## 运行方式

普通 CUDA 基线：

```bash
$HRM_PYTHON tools/run_grid.py \
    --state-format full_obs --episodes 200 --seeds 1 --workers 4 \
    --tasks book-and-quill --er-start-size 3000 \
    --work /tmp/auto_hrm_neural_benchmark_200 \
    --out /tmp/auto_hrm_neural_benchmark_200.jsonl
```

CPU 对照在当时的本地 runner 上加 `--device cpu`。MPS 客户端使用：

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/auto_nvidia_mps_pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/auto_nvidia_mps_log

$HRM_PYTHON tools/run_grid.py \
    --state-format full_obs --episodes 200 --seeds <并发数/4> --workers <并发数> \
    --tasks book-and-quill --er-start-size 3000 \
    --work /tmp/auto_hrm_neural_mps<N>_benchmark_200 \
    --out /tmp/auto_hrm_neural_mps<N>_benchmark_200.jsonl
```

## 当时本地 runner 的差异

[`run_grid_93ac3f6_local.patch`](run_grid_93ac3f6_local.patch) 是当时在
`93ac3f6` 工作树上运行以下命令得到的差异内容：

```bash
git diff 93ac3f6 -- tools/run_grid.py
```

它只包含设备覆盖、设备元数据、长任务优先排序和 `as_completed` 即时落盘。**没有把
当时的 `run_grid.py` 整个文件复制回来，也没有覆盖此后新增的配置指纹、环境步预算、
密集评估、C-full 和采样干预等改动。** 当前新版 runner 应继续作为主文件；这个补丁
只用于解释测速和神经 pilot 是如何启动、调度与落盘的。

## 限制

- 200 幕探针只用于比较硬件吞吐，不能作为 200,000 幕的学习结果。
- 一次 run 内含 10 张地图；这里的“幕”是配置中的每地图幕数，不要把单 run 秒数
  错除以 2,000 后再与现有文档的每地图进度口径混用。
- 负载包含一半 safe、一半 lava；全 safe 或其他任务的扩展曲线可能不同。
- 功耗与显存是运行中的单点观察，不是带误差条的系统测量。
