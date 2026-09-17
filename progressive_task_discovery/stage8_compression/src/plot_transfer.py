"""Does the learned task automaton survive a change of map, and what is it worth?"""
import collections, json, statistics, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLOURS = {'cold_history': '#333333', 'cold_merged': '#6f9c4e',
           'transfer': '#c05640', 'oracle': '#1b6ca8'}
LABELS = {'cold_history': 'remember everything, no structure',
          'cold_merged': 'learn the structure here, from scratch',
          'transfer': 'bring the structure from the other map',
          'oracle': 'given the true structure'}


def main():
    curves = [json.loads(l) for l in (ROOT / 'results/transfer/curves.jsonl').open()]
    amort = [json.loads(l) for l in (ROOT / 'results/transfer/amortise.jsonl').open()]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))

    g = collections.defaultdict(list)
    for r in curves:
        g[r['arm']].append(r)
    marks = [500 * k for k in range(1, 81)]
    for arm in ('cold_history', 'cold_merged', 'transfer', 'oracle'):
        ys = []
        for m in marks:
            vals = []
            for r in g[arm]:
                seen = [c[1] for c in r['checkpoints'] if c[0] <= m]
                vals.append(seen[-1] if seen else 0.0)
            ys.append(statistics.mean(vals))
        ax[0].plot(marks, ys, c=COLOURS[arm], lw=1.8, label=LABELS[arm])
    ax[0].set_xlabel('practice steps on the new map')
    ax[0].set_ylabel('share of starting points it can finish from')
    ax[0].set_title('A carried structure is a head start, not a shortcut', fontsize=11)
    ax[0].legend(fontsize=8, loc='lower right')
    ax[0].grid(alpha=.25)
    ax[0].set_xlim(0, 40000)

    by = collections.defaultdict(list)
    for r in amort:
        by[r['source_budget']].append(r)
    # The reference is the cold arm from the same batch the amortisation numbers
    # are quoted against, not the curve batch, so the figure and the table agree.
    reference = [json.loads(l) for l in (ROOT / 'results/transfer_run/raw.jsonl').open()]
    cold = statistics.mean((x['first90'] if x['first90'] > 0 else x['budget'])
                           for x in reference if x['arm'] == 'cold_merged')
    budgets = sorted(by)
    payback = []
    for b in budgets:
        target = statistics.mean((x['first90'] if x['first90'] > 0 else x['budget']) for x in by[b])
        saved = cold - target
        payback.append(b / saved if saved > 0 else None)
    good = [(b, p) for b, p in zip(budgets, payback) if p]
    ax[1].plot([b for b, _ in good], [p for _, p in good], 'o-', c='#c05640')
    for b, p in good:
        ax[1].annotate(f'{p:.1f}', (b, p), textcoords='offset points', xytext=(6, 5), fontsize=8)
    bad = [(b, p) for b, p in zip(budgets, payback) if p is None]
    for b, _ in bad:
        ax[1].scatter([b], [0], marker='x', c='#9a9a9a', s=50)
        ax[1].annotate('never pays\nback', (b, 0), textcoords='offset points',
                       xytext=(6, 8), fontsize=8, color='#9a9a9a')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('practice steps spent on the first map before moving on')
    ax[1].set_ylabel('new maps needed before it pays for itself')
    ax[1].set_title('Cheapest at about ten thousand steps', fontsize=11)
    ax[1].grid(alpha=.25)
    for ext in ('png', 'svg'):
        plt.tight_layout()
        plt.savefig(ROOT / f'figures/transfer.{ext}', dpi=150)
    print('wrote figures/transfer.png and .svg')


if __name__ == '__main__':
    main()
