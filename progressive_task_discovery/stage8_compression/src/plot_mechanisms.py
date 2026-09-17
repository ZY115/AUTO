"""Three separable mechanisms and what each does to the cost law."""
import collections, json, statistics, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main(name='full'):
    rows = [json.loads(l) for l in (ROOT / 'results' / name / 'raw.jsonl').open()]
    metas = json.loads((ROOT / 'results/family_maps/rungs.json').read_text())
    g = collections.defaultdict(list)
    for r in rows:
        g[(r['rung'], r['arm'])].append(r)
    cap = lambda rs: statistics.mean((x['first90'] if x['first90'] > 0 else x['budget']) for x in rs)
    mean = lambda arm: statistics.mean(cap(g[(m['tag'], arm)]) for m in metas)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))

    order = [('history', 'full history, no aid', '#333333'),
             ('history+random', '+ undirected detours', '#8a8a8a'),
             ('automaton', '+ oracle compression', '#1b6ca8'),
             ('history+scarcest', '+ directed structural trials', '#6f9c4e'),
             ('history+shape', '+ automaton shaping', '#c9902b'),
             ('history+shape+scarcest', '+ shaping and trials', '#a2543c'),
             ('automaton+shape+scarcest', '+ all three', '#c05640'),
             ('merged+scarcest', 'learned merging on top', '#7d5ba6')]
    y = list(range(len(order)))[::-1]
    ax[0].barh(y, [mean(a) for a, _, _ in order], color=[c for _, _, c in order])
    for yy, (arm, _, _) in zip(y, order):
        ax[0].text(mean(arm) + 500, yy, f'{mean(arm):,.0f}', va='center', fontsize=8)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([l for _, l, _ in order], fontsize=8.5)
    ax[0].set_xlabel('environment steps to 90% balanced success, mean over six rungs')
    ax[0].set_title('Three separable mechanisms, and they compose')
    ax[0].grid(alpha=.25, axis='x')

    conditions = [('history', 'automaton', 'no aid', '#333333'),
                  ('history+scarcest', 'automaton+scarcest', 'directed trials', '#6f9c4e'),
                  ('history+shape+scarcest', 'automaton+shape+scarcest', 'shaping + trials', '#c05640')]
    for ha, aa, label, col in conditions:
        pts = []
        for m in metas:
            for arm in (ha, aa):
                pts.append((statistics.mean(x['features'] for x in g[(m['tag'], arm)]),
                            cap(g[(m['tag'], arm)])))
        mx = statistics.mean(p[0] for p in pts)
        my = statistics.mean(p[1] for p in pts)
        slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)
        inter = my - slope * mx
        xs = [0, max(p[0] for p in pts) * 1.05]
        ax[1].scatter([p[0] for p in pts], [p[1] for p in pts], c=col, s=30, zorder=3)
        ax[1].plot(xs, [inter + slope * x for x in xs], '--', c=col, lw=1.2,
                   label=f'{label}: {inter:.0f} + {slope:.0f} x states')
    ax[1].set_xlabel('task-feature states visited in training')
    ax[1].set_ylabel('environment steps to 90% balanced success')
    ax[1].set_title('The aids flatten the slope; the floor barely moves')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.25)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        plt.savefig(ROOT / f'figures/mechanisms.{ext}', dpi=150)
    print('wrote figures/mechanisms.png and .svg')


if __name__ == '__main__':
    main(*sys.argv[1:])


def shaping_panel(name='depth_vs_graph/final_product.jsonl', out='figures/shaping.png'):
    """Four potentials, twelve tasks: the task graph alone against the product."""
    import collections, json, statistics
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads(l) for l in (root / 'results' / name).open()]
    metas = json.loads((root / 'results/depth_vs_graph/rungs12.json').read_text())
    g = collections.defaultdict(list)
    for r in rows:
        g[(r['rung'], r['shape'])].append(r)
    cap = lambda rs: statistics.mean((x['first90'] if x['first90'] > 0 else x['budget']) for x in rs)
    order = [('none', 'no shaping', '#333333'),
             ('frontier', 'progress depth', '#6f9c4e'),
             ('learned', 'task graph, learned', '#c9902b'),
             ('oracle', 'task graph, true', '#a2543c'),
             ('product', 'task graph x map', '#1b6ca8')]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    y = list(range(len(order)))[::-1]
    vals = [statistics.mean(cap(g[(m['tag'], a)]) for m in metas) for a, _, _ in order]
    ax[0].barh(y, vals, color=[c for _, _, c in order])
    for yy, v in zip(y, vals):
        ax[0].text(v + 400, yy, f'{v:,.0f}', va='center', fontsize=8.5)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([l for _, l, _ in order], fontsize=9)
    ax[0].set_xlabel('environment steps to 90%, mean over twelve tasks')
    ax[0].set_title('Task graph alone buys nothing, times the map 3.9x', fontsize=11)
    ax[0].grid(alpha=.25, axis='x')
    gaps = sorted({m['gap'] for m in metas})
    for arm, lbl, col in [('learned', 'task graph, learned', '#c9902b'),
                          ('oracle', 'task graph, true', '#a2543c'),
                          ('product', 'task graph x map', '#1b6ca8')]:
        ys = []
        for gap in gaps:
            tags = [m['tag'] for m in metas if m['gap'] == gap]
            ys.append(statistics.mean(cap(g[(t, 'frontier')]) / cap(g[(t, arm)]) for t in tags))
        ax[1].plot(gaps, ys, 'o-', c=col, label=lbl)
    ax[1].axhline(1, color='#888888', lw=.9, ls=':')
    ax[1].set_xticks(gaps)
    ax[1].set_xlabel('how far depth and task distance disagree, in transitions')
    ax[1].set_ylabel('speedup over the progress-depth potential')
    ax[1].set_title('Disagreement hurts the graph, not the product', fontsize=11)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.25)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        plt.savefig(root / f'figures/shaping.{ext}', dpi=150)
    print('wrote figures/shaping.png and .svg')
