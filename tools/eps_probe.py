"""同上，但跑 book —— 也就是曲线分叉真正发生的那个任务。

一个臂一个进程，各自把计数器写成 JSON，跑完再合并。分开跑是为了并行：
C-full 每个环境步贵约 8 倍，串起来要等它一个。
"""
import json, os, sys, tempfile
from collections import defaultdict
from pathlib import Path

ARM = sys.argv[1]
OUT = Path(sys.argv[2])
BUDGET = 4_000_000

HRM = Path(os.environ['HRM_LEARNING']); os.chdir(HRM); sys.path.insert(0, 'src')
from ilasp.ilasp_common import set_ilasp_env_variables
set_ilasp_env_variables('src')
from reinforcement_learning.ihsa_hrl_tabular_algorithm import IHSAAlgorithmHRLTabular
from reinforcement_learning.ihsa_hrl_tabular_crossstate_algorithm import (
    IHSAAlgorithmHRLTabularCrossState)

BASE = 'src/config/examples/ihsa/07-cw-frl-bq-exploit-flat/config.json'
d = json.loads((HRM / BASE).read_text())
d.update(debug=False, num_episodes=10**9, state_format='tabular', use_gpu=False,
         env_step_budget=BUDGET, seed=25101993)
# 这一行是上一版漏掉的：基础配置默认是 book-and-quill，网格跑的是 book。
d['environments'] = [dict(d['environments'][0], name='book')]
d['grid_params'] = dict(d['grid_params'], use_lava=True)
d['neutralize_deadends'] = False

w = Path(tempfile.mkdtemp(prefix=f'epsbook_{ARM}_'))
d['folder_name'] = str(w); d['checkpoint_folder'] = str(w)
cls = IHSAAlgorithmHRLTabular if ARM == 'Y11' else IHSAAlgorithmHRLTabularCrossState
if ARM != 'Y11':
    d['algorithm'] = 'ihsa-hrl-crossstate'
a = cls(d); a.run()

counts = defaultdict(list)
if ARM == 'Y11':
    for b in a._formula_banks:
        for g, n in b._q_function_step_counter.items():
            counts[str(g)].append(n)
else:
    for tb in a._perstate_banks.values():
        for b in tb.values():
            for g, n in b._q_function_step_counter.items():
                counts[str(g)].append(n)
OUT.write_text(json.dumps({'arm': ARM, 'task': 'book', 'budget': BUDGET,
                           'steps': a.train_env_steps,
                           'counts': {k: v for k, v in counts.items()}}, indent=1))
print(f'{ARM} 完成：{a.train_env_steps:,} 步，{len(counts)} 个子目标 → {OUT}')
