"""Why the learned merger fails: evidence coverage, and what that does to the arm."""
import collections, json, statistics, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main(name='confirm'):
    ctrl = json.loads((ROOT / 'results/merge_control.json').read_text())
    rows = [json.loads(l) for l in (ROOT / 'results' / name / 'raw.jsonl').open()]
    metas = json.loads((ROOT / 'results/family_maps/rungs.json').read_text())
    g = collections.defaultdict(list)
    for r in rows:
        g[(r['rung'], r['arm'])].append(r)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    byk = collections.defaultdict(list)
    for r in ctrl:
        byk[r['keep']].append(r['precision'])
    keeps = sorted(byk)
    ax[0].plot(keeps, [statistics.mean(byk[k]) for k in keeps], 'o-', c='#1b6ca8',
               label='merge precision on complete-to-partial evidence')
    for k in keeps:
        ax[0].scatter([k] * len(byk[k]), byk[k], s=14, c='#8fb8d6', zorder=2)
    reached = statistics.mean(x['structural_coverage'] for x in rows if x['arm'] == 'merged')
    ax[0].axvline(reached, color='#c05640', ls='--', lw=1.2,
                  label=f'coverage a trained agent reaches ({reached:.2f})')
    ax[0].set_xlabel('fraction of (state, event) pairs actually tried')
    ax[0].set_ylabel('merge precision')
    ax[0].set_title('The rule is exact on full evidence and collapses without it')
    ax[0].set_ylim(-.03, 1.05)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.25)

    order = sorted(metas, key=lambda m: m['honest_ratio'])
    x = list(range(len(order)))
    for arm, lbl, col in [('automaton', 'minimal automaton (oracle)', '#1b6ca8'),
                          ('history', 'full history', '#333333'),
                          ('merged', 'learned merging', '#c05640')]:
        ax[1].plot(x, [statistics.mean(v['auc'] for v in g[(m['tag'], arm)]) for m in order],
                   'o-', c=col, label=lbl)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"{m['honest_ratio']:.1f}" for m in order])
    ax[1].set_xlabel('design-time compression ratio')
    ax[1].set_ylabel('success AUC over training')
    ax[1].set_title('Merging on achievable evidence is worse than not merging')
    ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.25)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        plt.savefig(ROOT / f'figures/merging.{ext}', dpi=150)
    print('wrote figures/merging.png and .svg')


if __name__ == '__main__':
    main(*sys.argv[1:])
