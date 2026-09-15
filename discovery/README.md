# discovery —— 支撑「最初那个问题」的原始数据与脚本

[`docs/ORIGINAL_QUESTION.md`](../docs/ORIGINAL_QUESTION.md) 引用的数字，原本散在
另一个代码库 `progressive_task_discovery/stage8_compression/` 里，本仓库无法复现。
这个目录把**支撑那七条所需的部分**搬了过来。

**不是整个搬。** 原目录 1.2 GB，其中结果 1.1 GB。这里只有 2.4 MB——挑出七条用得到的，
大文件无损 gzip。没搬的包括 `conversion.jsonl`(256M)、`generalisation.jsonl`(287M)、
`reward_grid.jsonl`(66M) 等，它们支撑的是别的结论。

## 先验证，再引用

```bash
python3 discovery/src/reproduce.py
```

它从这里的归档重算文档里的每一个数，和文档写的并排打出来。**当前 18 项全部对上。**

对不上时先查口径，不要改数去凑。写这个脚本时就因为口径吃过三次亏：

| 踩的坑 | 真相 |
|---|---|
| 按 `method` 分组 | 应按 **`arm`**。`method` 会把 `+shape` 的另外两臂混进来 |
| 「从未达到」当成小数值 | `first90 <= 0`（常见是 **−1**）才是未达到，不是 200000 |
| 找 `shape=count` | 文档里的「标量进度计数器」在数据里叫 **`frontier`** |

状态数的字段是 **`features`**，不是 `nodes`（`nodes` 恒为 1）。

## 目录

```
src/
  compress.cpp          模拟器。clang++ -O3 -std=c++17 -o compress src/compress.cpp
  run_verify.py         第一、二条：中途反馈的预算曲线
  depth_vs_graph.py     第三、四条：信号强度与「学出来 vs 真值」
  run_query.py          查询策略（21% 提问节省）
  run_audit.py          归属天花板（2.04x）
  reproduce.py          重算并核对上面全部数字

results/
  verify.jsonl.gz                 1,584 行   第一、二条
  rawhistory.jsonl.gz               960 行   第五条
  query.jsonl.gz                  4,608 行   21% 提问节省
  audit.jsonl.gz                  2,304 行   归属天花板
  attribution2.jsonl.gz           4,032 行   签名合并规则
  occam_merge_n3.json                        第七条 · 离线
  online_occam_n3*.json                      第七条 · 实时
  depth_vs_graph/
    final12.jsonl.gz              1,536 行   第三、四条主结果
    final_product.jsonl.gz        1,920 行   乘积臂
    final_controls.jsonl.gz       3,072 行   对照：shuffled / wrongproduct / mapdist / taskstep
    manifest*.json rungs12.json              协议与任务定义
  SHA256SUMS                                 全部归档文件的校验和

reports/
  REPORT_STEP23.md  24  26  33               对应四条的原始报告
  DECISIONS.md                               决策日志，含各步的预先登记与撤回
```

## 那些对照臂为什么值得留

`final_controls.jsonl.gz` 里的四个臂，是用来证明「任务图 × 地图的乘积」不是假象的：

- **`shuffled`** —— 把乘积的值打乱。若它也赢，说明赢的是塑形的量级不是内容
- **`wrongproduct`** —— 用错误的任务图算乘积
- **`mapdist`** —— 只有地图距离，没有任务结构
- **`taskstep`** —— 只有任务转移数，没有地图

**没有这四个，1,128 那个数说明不了任何机制。**

## 没搬进来的，和为什么

| | 为什么不搬 |
|---|---|
| 1.1 GB 的其余结果 | 支撑的是别的结论，不在那七条里 |
| `figures/` | 都能从数据重画 |
| `stage7_branching/` 及更早 | 那些阶段的结论多数已被撤回或被后续取代 |
| 编译好的 `compress` 二进制 | arm64 Mach-O，换平台要重编，源码在这里 |

要完整的原始树，在本机 `progressive_task_discovery/`。**那个目录不在任何 git 仓库里**
——这是一处真实风险，删了就没了。
