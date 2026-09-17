# 工具适配评估

2026-09-10 核对。这里把“已在本机执行”“已核对官方接口”“只宜借算法”分开；不把公开仓库链接视作成功复现。只有 AALpy 安装在本审阅目录的 `tools/vendor/`，原项目依赖未改变。

| 工具/来源 | 可复用的部分 | 需要提供的信息 | 本轮状态与限制 |
|---|---|---|---|
| [AALpy](https://github.com/DES-Lab/AALpy) | RPNI/GSM、Mealy/DFA，被动状态合并的独立实现 | prefix-closed 的输入词与输出反馈 | **本地 GSM smoke 通过**；1.6.2 classic 同例子未压缩，不能当已验证基准 |
| [ISA](https://github.com/ertsiger/induction-subgoal-automata-rl) | 自动机归纳、共享公式技能、元控制器、修订保留 | 标签、成功/失败轨迹、ILASP 假设空间 | 官方仓库/论文核对；未跑。旧栈涉及 Python 3.6、ILASP 3.6、clingo 5.4，M1 适配需另验 |
| [HRM](https://github.com/ertsiger/hrm-learning) | 选项栈、子机调用、课程、强 HRL 基准 | 标签、任务组、归纳配置 | 未安装。其 JIRP baseline 部分专注模型学习，不能直接当完整 JIRP 策略基准 |
| [Reward Machines](https://github.com/RodrigoToroIcarte/reward_machines) | 标准 CRM、HRM、塑形更新的语义参考 | 正确 RM 和标注函数 | 未安装；旧 Python/TF/Baselines 依赖不宜直接混进当前项目 |
| [PyCRM](https://github.com/TristanBester/pycrm) / [包](https://pypi.org/project/pyrewardmachines/) | Gymnasium 乘积环境、counterfactual generation、SB3 集成 | ground environment、labeler、RM | 官方接口与软件论文已核对，未安装。包名 `pyrewardmachines`，包页面要求 Python >=3.10,<3.13 |
| [LOF](https://github.com/braraki/logical-options-framework) | 离散选项与高层规划；对照 greedy | 给定 FSA、选项成本/终止模型 | 已确认作者仓库；未运行。最优性有适用条件 |
| [SF-FSA-VI regression](https://github.com/timtimtim3/sf-fsa-vi-regression) | 多个目标出口、successor features、规划 | 已知 FSA/事件映射，足够技能覆盖 | 未运行；适合研究多实例目标造成的后缀成本 |
| [TALearner](https://github.com/dkhyland/TALearner) | 乘积辨识、cone lumping、环境偏差分析 | 标签化环境轨迹，HMM/乘积估计假设 | C++ 代码库已确认，未编译；观测接口与当前 progress 反馈不同 |
| [C-PREP](https://github.com/CLAIR-LAB-TECHNION/C-PREP) | RM 上下文表示、预规划、迁移包装 | 上下文到 RM 的生成器 | 未运行；需要其 conda 环境和 SB3 修改版 |
| [FORM](https://github.com/leoardon/form) | 对象数量变化下的一阶规则归纳 | 对象谓词/量词语义与事件标签 | 未运行；官方列 Python 3.10.12、ILASP 4.4 Ubuntu 包与 Ray patch，不能直接保证 M1 可用 |
| [Noisy RM](https://github.com/andrewli77/reward-machines-noisy-environments) | Naive/IBU/TDM，历史到 RM 状态推断 | 带噪标签模型，部分实现需真实 RM 状态训练目标 | 未运行；不能直接解未知结构后验问题 |
| [Shielding](https://github.com/safe-rl/safe-rl-shielding) | 安全博弈/动作屏蔽上界 | 可靠的安全规范与环境抽象 | 未运行；不把未知任务机当正确安全模型 |

Future Dependent Options 的作者稿提供代码网盘链接；本轮未核对其中内容，因此目前只列为算法参考，未列为可运行工具。

## AALpy 的实际试验

使用 Python **3.12.14**，库版本 **1.6.2**，单独安装目录。系统 Python 3.9 导入时报 `typing.defaultdict` 错误；没有修改库源码来掩盖兼容性问题。

任务：先 A 后 B；B 先出现会失败；输出区分 ignored、progress、success、failure。用 A/B 字母表、长度 1–5 全部 62 个词构造 prefix-closed 数据；独立枚举长度 1–7 共 254 个词核对完整输出串。

| 实现 | 状态数 | 未定义测试词 | 输出不一致总数 |
|---|---:|---:|---:|
| 默认 GSM | 4 | 0 | 0 |
| classic | 63 | 192 | 192 |

这是接口/语义 smoke，**不说明 GSM 在 Stage 8 上优于项目学习器**。classic 的原因尚未定位；不能只挑通过者就宣称“已完成 RPNI 文献基准”。用官方库对照当前项目的下一项必要工作是冻结同一批观测，而不是直接换训练策略后比较两批不同数据。

从工作区根目录重跑：

```bash
/Users/yuhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 research/2026-09-10-stage8-review/tools/smoke_aalpy.py
```

默认保持未观测边为 unknown。不要为了满足接口而把 unknown 自动设为 self-loop；那会混淆“未观察到”和“已确认忽略”。本例得到完整 GSM 是数据覆盖的结果，不是补全策略。

## 接入前应固定的最小接口

1. 物理数据：`state, action, next_state, event`。
2. 真实反馈：奖励、progress、success、failure，以及哪些项确实可见。
3. 执行记录：实际目标、选项起止、episode 与模型版本。
4. 学习器输出：已知转移、未知转移、终止语义和证据来源。
5. 使用器：独立选择索引 Q、目标技能、高层规划或反事实更新。

这样可以替换学习器而不同时改变反馈，替换使用器而不隐式赠送地图或任务答案。它是接口审阅建议，尚未修改原实现。
