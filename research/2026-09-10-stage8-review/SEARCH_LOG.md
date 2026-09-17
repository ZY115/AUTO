# 检索范围与筛选记录

日期：2026-09-10。先以本地 STEP8–16 的问题为入口作宽搜索，再沿论文方法、引用和作者代码追踪。采用论文原文、正式会议/期刊页面、作者预印本和作者仓库作为论据来源。聚合站只用于发现，不作为技术结论依据。

## 搜索主题与实际查询示例

- 任务结构与技能：`hierarchical reinforcement learning reward machines subgoal automata goal conditioned policies counterfactual experience transfer unknown automata`；`automata guided reinforcement learning subgoal options hindsight experience replay JIRP HRM`。
- 不确定结构：`reward machines uncertain learned model counterfactual experience negative transfer reward machine learning 2024 2025 2026`；`"Reward Machines" "uncertain" "2025"`；`reward machines uncertainty confidence counterfactual model errors`。
- 迁移：`"reward machines" "transfer" "learned" 2025 2026`；`"reward machines" "transfer" "2025" learning`；FORM、C-PREP、TALearner 定向核对。
- 不可逆与安全：`"reward machines" "hierarchical" "safety" learned`；`"reward machines" "uncertainty" automaton structure safe exploration`。
- 新工具和近期工作：`"Reward Machines" "2026" learning`；`"PyCRM" reward machines`；`reward machines survey 2025 2026 learning transfer uncertainty`。
- 子目标的短视与多实例：`"Logical Options Framework" reinforcement learning`；`"Learning Spatially Refined Sub-Policies"`；`"Generalization of Temporal Logic Tasks" "Future" Options`；`"future-dependent options" reinforcement learning`。
- 复现代码：`"Joint Inference of Reward Machines" github JIRP`；`"Logical Options Framework" github araki`；`"Active Finite Reward Automaton" github`；`"Reinforcement Learning with Symbolic Reward Machines" github`。

## 纳入规则

优先下载能直接改变以下判断的论文：最近已有方法是什么；我们使用了哪些额外信息；能否提供独立算法或语义实现；是否已有对短视、任务迁移和结构不确定性的处理。核心材料均保存 PDF、提取文本、版本与 SHA256，并写单独笔记。精读集中于问题定义、关键算法、保证前提与实验接口，不声称重证所有定理。

## 广搜后未作为核心依据的材料

| 材料 | 本轮处理 | 原因 |
|---|---|---|
| [ARM-FM, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/68f24fad8f460d4b7bbac402c252ebbb-Abstract-Conference.html) | 页面筛选，未下载精读 | 基础模型相关先验会改变本项目信息预算，非当前最小基准 |
| [Physics-Informed Reward Machines](https://arxiv.org/abs/2508.14093) | 搜索线索，未精读 | 可留作物理模型假设的后续阅读，不据此作技术主张 |
| [Noise-robust RM induction, Scientific Reports 2026](https://www.nature.com/articles/s41598-026-54889-z) | 页面/摘要筛选，未下载精读 | 概率标注与遗传搜索相关，但当前标签精确；没有验证其可复现性 |
| [Noisy Symbolic Abstractions, 2022](https://arxiv.org/abs/2211.10902) | 前序工作筛选 | 本轮优先阅读同方向更完整的 P10 |
| [Reward Models in Deep RL: A Survey, IJCAI 2025](https://www.ijcai.org/proceedings/2025/1199) | 宽搜索排除 | 范围偏一般奖励建模，不是任务自动机基准 |
| LLM reward-model survey、金融/基础设施的“future options”结果 | 排除 | 关键词碰撞，与任务无关 |

## 下载与版本处理

P21 的 OpenReview 直链下载失败后改用 BNAIC 官方会议网站，保留原发现链接。P23 期刊页面为订阅预览，使用作者公开的 EasyChair 稿，没有绕过付费限制。P03 下载的是 arXiv 扩展稿，P11 的本地文件名不作为出版年份证据。P14、P15、P22 的新颖性声明按作者预印本主张处理。

这是一轮问题驱动的宽搜索和定向精读，不是系统综述：没有虚构 PRISMA 数量、穷尽性或“已无其他相关工作”的结论。各项潜在改进只表示当前值得检验的空间，不是已经证明无人做过。
