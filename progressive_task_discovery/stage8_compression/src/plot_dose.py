"""Step 38.5 robustness: the hazard density was tuned, so here is the whole range.

The claim under attack is that 3% was picked to flatter the sharing arm. What the
sweep shows is a regime, not a point: the separation opens at a single hazard
cell and holds across every dose tried, while both arms are easy at zero and
both degrade at the top.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC',
                                          'Arial Unicode MS', 'PingFang SC']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / 'results/dose_sweep.json').read_text())
DOSES = ['0.00', '0.01', '0.02', '0.03', '0.04']
HAZ = [0, 1, 2, 3, 4]
ARMS = ['QRM', 'Y10 no-reuse', 'Y11 goal readout', 'Y01 meta (safe)']
LAB = {'QRM': 'QRM（结构进价值表）', 'Y10 no-reuse': '仅路由（每状态独立技能）',
       'Y11 goal readout': '目标读法（技能共享）',
       'Y01 meta (safe)': '仅技能 + 安全目标提示'}
COL = {'QRM': '#805ad5', 'Y10 no-reuse': '#3182ce',
       'Y11 goal readout': '#2f855a', 'Y01 meta (safe)': '#dd6b20'}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, key, ylab, logy in (
        (axes[0], 'rmst', 'RMST', False),
        (axes[1], 'attain', '达到 90% 的运行比例', False),
        (axes[2], 'fail90', '达标前累计不可逆失败（对数轴）', True)):
    for a in ARMS:
        y = [D[f'{d}|{a}'][key] for d in DOSES]
        xs = [h for h, v in zip(HAZ, y) if v is not None]
        ys = [v for v in y if v is not None]
        if logy:
            xs = [x for x, v in zip(xs, ys) if v and v > 0]
            ys = [v for v in ys if v and v > 0]
        ax.plot(xs, ys, '-o', color=COL[a], lw=1.9, ms=6, label=LAB[a])
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('危险格数（104 个自由格）')
    ax.set_ylabel(ylab)
    ax.set_xticks(HAZ)
    ax.grid(alpha=.2, which='both')
axes[0].axvspan(0.6, 4.4, color='#2f855a', alpha=.06)
axes[0].annotate('分叉区间', xy=(2.5, .55), fontsize=9, color='#2f855a', ha='center')
axes[1].legend(fontsize=8, loc='center right')
fig.suptitle('第三十八步半 稳健性：分叉不是一个点，一个危险格就出现并贯穿整个扫描区间',
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .93])
out = ROOT / 'results/step38_5_dose.png'
fig.savefig(out, dpi=160)
print(out)
