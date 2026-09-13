# 文件清单与上游锁定

## 上游仓库（不在本仓库内，由 `setup.sh` 克隆）

| 仓库 | 提交 | 作用 |
|---|---|---|
| `ertsiger/hrm-formalism-envs` | `929605f` | CraftWorld 环境与任务层级 |
| `ertsiger/hrm-learning` | `23d0e25` | 算法实现与官方配置 |
| `ertsiger/hrm-minigrid` | `d0e6763` | 作者魔改的 gym-minigrid |

上游如有更新，以这里的提交号为准。补丁是针对这些提交生成的。

## 本仓库文件

| 路径 | 说明 |
|---|---|
| `setup.sh` | 一键搭环境 |
| `patches/01-hrm-formalism-envs.patch` | 新增 `neutralize_deadends` 选项 |
| `patches/02-hrm-learning.patch` | Python 3.10 兼容、注册新臂、MPS 设备分支 |
| `src/ihsa_hrl_tabular_perstate_algorithm.py` | Y10 表格版 |
| `src/ihsa_hrl_dqn_perstate_algorithm.py` | Y10 神经版 |
| `tools/verify.py` | 七项预检，跑实验前必须通过 |
| `tools/run_grid.py` | 网格 runner，支持断点续跑 |
| `tools/analyze_grid.py` | 配对自助分析，带地板保护 |
| `tools/y10_equivalence_checks.py` | 表格版深度等价性检查 |
| `tools/y10_equivalence_checks_dqn.py` | 神经版深度等价性检查 |
| `results/reference/macos_tabular_10k.jsonl` | 48 次已有运行，供对照 |

## 交付前验证过的项

在 macOS / Apple M1 Pro 上，**从空目录完整跑了一遍 `setup.sh`，
再用那个全新环境跑 `verify.py`**，证明「克隆下来就能用」：

- `setup.sh` 的依赖顺序与版本，逐条来自实际安装记录
- 两个补丁都能干净地**反向**应用到对应仓库，证明与上游代码匹配
- `verify.py` 七项里六项通过，唯一失败的是 CUDA 那项（Mac 上本应如此）
- 全新环境跑出的等价性日志指纹（`39e1494b9874` / `b435052dc73d`）
  与开发时的工作副本**完全一致**，证明打包的补丁与文件能复现同样的行为
- `run_grid.py` 脱离原项目目录独立跑通
- `analyze_grid.py` 能读新格式，也能读 `results/reference/` 里的旧格式，
  且复现出的数字与研究报告一致
- `analyze_grid.py` 的地板保护正确触发：岩浆条件下两臂都是 0 时，
  报「无法判读」而不是打印一个会被误读成反向证据的负数
