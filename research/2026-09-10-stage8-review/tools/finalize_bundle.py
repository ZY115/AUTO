from pathlib import Path
import json, shutil, hashlib, re

R = Path(__file__).resolve().parents[1]
S = R.parents[1] / 'progressive_task_discovery/stage8_compression'
snap = R / 'audit/snapshot'
files = ['src/compress.cpp', 'src/run_priorart.py', 'src/run_goal_irreversible.py',
         'src/goal_report.py', 'src/run_revision.py', 'REPORT_STEP15.md', 'REPORT_STEP16.md',
         'results/irreversible/taskR_00_fatal.task', 'results/depth_vs_graph/gap00.task']
meta = []
for name in files:
    src, dst = S / name, snap / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    meta.append({'source': str(src), 'snapshot': str(dst.relative_to(R)),
                 'sha256': hashlib.sha256(dst.read_bytes()).hexdigest()})
(R / 'audit/snapshot_manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
summary = json.loads((R / 'audit/data_summary.json').read_text())
lines = ['# 数据与审计证据', '',
         '审计只增加观察或构造小型接口例子；没有修复原实现后重新训练整批实验。', '',
         '| 结果文件 | 记录数 | 处理 |', '|---|---:|---|']
for name, d in summary.items():
    lines.append(f'| [{name}](<{S / "results" / name}>) | {d["rows"]} | 聚合、记录 SHA256；不同条件不混池 |')
lines += ['', '## 可复查文件', '']
evidence = {
    'data_summary.json': '聚合原始字段：first90_mean 可能含 -1；goal_vs_index 另列 first90_censored_mean 才是报告口径。',
    'manifest_check.json': '最新五份源码/数据 hash 均匹配。',
    'irreversible_witness.json': '保存完整输出、命令、诊断以及与既有行的对照。',
    'goal_replay_witness.json': '同一 Raw 在 relabel=false 下改变被更新目标。',
    'signature_witness.json': '局部细化的语义反例，不是完整 JIRP 运行。',
    'corrected_fatal_bootstrap.json': '地图重采样保留重复次数；没有修复学习数据污染。',
    'aalpy_smoke.json': 'GSM 通过，classic 未通过同样的长词验证。',
    'snapshot_manifest.json': '保存核心发现的源码/报告快照；不复制全部历史数据。',
}
for name, desc in evidence.items():
    lines.append(f'- [{name}](<{R / "audit" / name}>)：{desc}')
lines += ['', '## 重现入口', '', '从工作区根目录执行：', '', '```bash',
          'python3 research/2026-09-10-stage8-review/tools/audit_existing.py',
          'python3 research/2026-09-10-stage8-review/tools/signature_witness.py',
          '/Users/yuhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 research/2026-09-10-stage8-review/tools/smoke_aalpy.py',
          '```', '',
          'audit_existing.py 会再次执行一个已有 200000 步条件并写入本审计目录；不是新参数扫描。它默认读当前源码，后续若项目变化，应先核对 snapshot hash。目标回放见证通过 include 调用原始 Learner，不更改其逻辑。', '',
          '## 限制', '',
          '没有重算全部历史报告的区间；没有证明修复后的效应大小或方向。旧 irreversible_run/run2 与 dose 版本可能不同，本审阅不混成同一批证据。']
(R / 'AUDIT.md').write_text('\n'.join(lines) + '\n')
readme = f'''# Stage 8 审阅资料包 · 2026-09-10

**先读 [总审阅](<{R / 'MAIN_REVIEW.md'}>)。** 方向可继续，但当前最强主张需要收紧，尤其失败态语义、JIRP 对照与重标记消融。

- [23 篇论文与逐篇笔记](<{R / 'LITERATURE_INDEX.md'}>)：PDF、正文、方法假设、与实验的对应关系、潜在改进空间。
- [工具评估](<{R / 'TOOLS.md'}>)：能直接借用什么，以及实际安装/测试到什么程度。
- [数据审计与复现证据](<{R / 'AUDIT.md'}>)：11,180 条记录聚合、5 份清单核对、独立见证。
- [检索范围与排除记录](<{R / 'SEARCH_LOG.md'}>)：宽搜索后如何收敛；不声称穷尽。
- [下载清单](<{R / 'paper_manifest.json'}>)：来源、路径、字节数和 SHA256。

建议阅读顺序：总审阅 → P01 ISA → P03 JIRP → P16 LOF → P23 Future Dependent Options → P05 TALearner / P19 FORM → 工具表。

本轮做审阅、已有条件复现与小型工具核对。原代码、原报告、原数据未修改。研究空间供之后讨论，没有开始新一轮算法或实验设计执行。
'''
(R / 'README.md').write_text(readme)
for p in R.rglob('*.md'):
    if 'vendor' in p.parts or 'snapshot' in p.parts:
        continue
    t = p.read_text()
    for name in ['LITERATURE_INDEX.md', 'TOOLS.md']:
        t = t.replace(f']({name})', f'](<{R / name}>)')
    t = re.sub(r'\]\((/[^\n)]+)\)', r'](<\1>)', t)
    p.write_text(t)
print('rows', sum(d['rows'] for d in summary.values()), 'pdfs', len(list((R/'papers').glob('*.pdf'))),
      'notes', len(list((R/'notes').glob('*.md'))))
