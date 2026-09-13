# 结果怎么交回

## 目录约定

```
results/
├── reference/          已有结果，只读，不要动
└── runs/               ← 你把跑出来的放这里
```

## 文件命名

```
results/runs/<机器标识>_<状态格式>_<幕数>.jsonl
```

例如：

```
results/runs/a100_tabular_200k.jsonl
results/runs/a100_neural_pilot_200k.jsonl
```

一次网格一个文件。分几批跑的就用同一个 `--out`，`run_grid.py` 会追加并自动跳过已完成的。

## 交什么

**交原始的 `.jsonl`，不要只交汇总表或图。**

`run_grid.py` 每完成一次运行就追加一行，里面已经包含：

- 完整命令行
- 三个关键源码文件的哈希
- 任务、风险条件、协议、臂、随机种子、幕数
- 墙钟耗时
- **每个任务实例的完整贪心评估曲线**（`幕数;奖励;步数`）

有了这些，任何分析都能重做。只交汇总的话就不行了——
本项目早期有两批数据只保留了汇总，逐种子的原始结果已经永久丢失，
那部分结论至今无法复查。这是硬性要求的由来。

## 一起交的东西

在 `results/runs/` 里放一个同名的 `.md`，写清楚：

```markdown
# a100_tabular_200k

- 机器：8× A100 80GB / 64 核 / Ubuntu 22.04
- 跑的命令：（原样粘贴）
- 起止时间与总耗时
- verify.py 的输出（七项全过的截图或文本）
- pip freeze 的输出
- git -C hrm-learning diff --stat 与 git -C hrm-formalism-envs diff --stat
- 有没有改过任何东西？改了什么、为什么
- 中途有没有失败重跑的运行？
```

`pip freeze` 和两个 `git diff --stat` 是用来确认环境和补丁状态的，
出现异常结果时第一时间要查这两样。

## 提交

```bash
git add results/runs/
git commit -m "结果：A100 表格版 200k，36 次运行"
git push
```

**不要提交** `--work` 目录里的训练产物（几百兆的中间文件）。
`.gitignore` 已经挡掉了常见位置，但如果你把 `--work` 指到了仓库里面，请自己确认。

如果 `.jsonl` 超过 100 MB（36 次运行的表格版大约 1 MB，正常不会超），
用 Git LFS 或者分文件。

## 跑完自己先看一眼

```bash
python3 tools/analyze_grid.py results/runs/*.jsonl
```

对照 [根目录 README 的「已有结果」](../README.md#已有结果)：
如果你的**无岩浆**那三行和参考表方向一致（Y11 全是 1.000，
Y10 随任务变长从 0.911 掉到 0.088），说明环境和改动都正确。

如果差得很远，先别急着交，去
[TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) 走一遍排查顺序。

## 我方怎么同步

```bash
git pull
python3 tools/analyze_grid.py results/runs/*.jsonl results/reference/*.jsonl
```

`analyze_grid.py` 接受多个文件，会按「状态格式 × 协议」自动分组，
所以新旧结果可以放在一起看。
