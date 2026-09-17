"""Two panels: the cost law with its representation-independent intercept, and the ladder."""
import collections, json, statistics, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLOURS = {'automaton': '#1b6ca8', 'bag': '#c05640', 'history': '#333333', 'count': '#9a9a9a'}


def main(name='ladder32'):
    rows = [json.loads(l) for l in (ROOT / 'results' / name / 'raw.jsonl').open()]
    metas = json.loads((ROOT / 'results/family_maps/rungs.json').read_text())
    g = collections.defaultdict(list)
    for r in rows:
        g[(r['rung'], r['arm'])].append(r)
    cap = lambda rs: [(x['first90'] if x['first90'] > 0 else x['budget']) for x in rs]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    pts = []
    seen = set()
    for (tag, arm), rs in sorted(g.items()):
        solved = all(x['first90'] > 0 for x in rs)
        f, c = statistics.mean(x['features'] for x in rs), statistics.mean(cap(rs))
        col = COLOURS.get(arm, '#6f9c4e')
        label = None
        key = (arm if arm in COLOURS else 'window', solved)
        if key not in seen:
            seen.add(key)
            label = (arm if arm in COLOURS else 'sliding window') + ('' if solved else ' (censored)')
        ax[0].scatter(f, c, c=col, marker='o' if solved else 'x', s=44, zorder=3, label=label)
        if solved:
            pts.append((f, c))
    mx, my = statistics.mean(p[0] for p in pts), statistics.mean(p[1] for p in pts)
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)
    inter = my - slope * mx
    ss = sum((p[1] - my) ** 2 for p in pts)
    r2 = 1 - sum((p[1] - (inter + slope * p[0])) ** 2 for p in pts) / ss
    xs = [0, max(p[0] for p in pts) * 1.05]
    ax[0].plot(xs, [inter + slope * x for x in xs], '--', c='#1b6ca8', lw=1.2,
               label=f'cost = {inter:.0f} + {slope:.0f} x states  (R2 = {r2:.2f})')
    ax[0].axhline(inter, color='#c05640', lw=.9, ls=':',
                  label=f'representation-independent floor ({inter:.0f})')
    ax[0].set_xlabel('task-feature states actually visited in training')
    ax[0].set_ylabel('environment steps to 90% balanced success')
    ax[0].set_title('Cost is affine in representation size')
    ax[0].legend(fontsize=7.5, loc='upper left')
    ax[0].grid(alpha=.25)

    order = sorted(metas, key=lambda m: m['honest_ratio'])
    x = list(range(len(order)))
    for arm, lbl, style in [('history', 'full history', 'o-'),
                            ('automaton', 'minimal automaton', 'o-'),
                            ('count', 'progress count (never solved)', 's--')]:
        ax[1].plot(x, [statistics.mean(cap(g[(m['tag'], arm)])) for m in order],
                   style, c=COLOURS[arm], label=lbl)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"{m['honest_ratio']:.1f}" for m in order])
    ax[1].set_xlabel('design-time compression ratio')
    ax[1].set_ylabel('environment steps to 90% balanced success')
    ax[1].set_title('Six rungs, matched map, length and updates')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.25)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        plt.savefig(ROOT / f'figures/compression_cost.{ext}', dpi=150)
    print('wrote figures/compression_cost.png and .svg')


if __name__ == '__main__':
    main(*sys.argv[1:])
