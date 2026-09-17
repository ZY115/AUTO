"""One task, drawn: the map, the moment that matters, and what each method scores."""
import collections, json, statistics, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from emit_tasks import make_layout

GOAL, MARK, START, PLAIN, HUB = '#1b6ca8', '#c9902b', '#6f9c4e', '#e8e8e8', '#555555'


def main(tag='gap01'):
    meta = {m['tag']: m for m in json.loads(
        (ROOT / 'results/depth_vs_graph/rungs12.json').read_text())}[tag]
    layout = make_layout(meta)
    cells = [tuple(c) for c in layout['cells']]
    label_of = {i: g for i, g in layout['labels']}
    starts = set(meta['starts'])
    n = meta['n']
    nm = lambda g: 'ABC'[g] if g < n else 'XYZ'[g - n]

    fig = plt.figure(figsize=(13, 5.6))
    ax = fig.add_axes([.01, .04, .45, .88])
    for i, (x, y) in enumerate(cells):
        if i in label_of:
            g = label_of[i]
            col, txt = (GOAL, nm(g)) if g < n else (MARK, nm(g))
        elif i in starts:
            col, txt = START, 'S'
        elif (x, y) == (0, 0):
            col, txt = HUB, 'H'
        else:
            col, txt = PLAIN, ''
        ax.add_patch(Rectangle((x - .45, -y - .45), .9, .9, facecolor=col,
                               edgecolor='white', linewidth=1.4))
        if txt:
            ax.text(x, -y, txt, ha='center', va='center', fontsize=11,
                    color='white', fontweight='bold')
    zc = [c for i, c in enumerate(cells) if label_of.get(i) == n + 2][0]
    ax.add_patch(Circle((zc[0], -zc[1]), .62, fill=False, edgecolor='#c05640',
                        linewidth=2.2, linestyle='--'))
    ax.text(zc[0] + .95, -zc[1], 'the moment that matters:\nboth histories stand here,\n'
            'both have done 6 things,\none owes 1 more, the other 4',
            fontsize=8.5, va='center', color='#c05640')
    xs, ys = [c[0] for c in cells], [-c[1] for c in cells]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 7.5)
    ax.set_ylim(min(ys) - 4.4, max(ys) + 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('One task: 28 walkable cells, three corridors and one exit', fontsize=11)
    for col, lab, row in [(GOAL, 'A B C   collect all three, in any order', 0),
                          (MARK, 'X Y Z   the marks that come afterwards', 1),
                          (START, 'S   the six forced starting points', 2)]:
        yy = min(ys) - 1.5 - row * 1.05
        ax.add_patch(Rectangle((min(xs) - .4, yy - .4), .8, .8,
                               facecolor=col, edgecolor='white'))
        ax.text(min(xs) + .7, yy, lab, fontsize=8.5, va='center')

    bx = fig.add_axes([.55, .13, .43, .78])
    rows = [json.loads(l) for l in (ROOT / 'results/depth_vs_graph/final_controls.jsonl').open()]
    metas = json.loads((ROOT / 'results/depth_vs_graph/rungs12.json').read_text())
    g = collections.defaultdict(list)
    for r in rows:
        g[(r['rung'], r['shape'])].append(r)
    cap = lambda rs: statistics.mean((x['first90'] if x['first90'] > 0 else x['budget']) for x in rs)
    order = [('shuffled', 'right numbers, wrong places', '#9a9a9a'),
             ('mapdist', 'only where you are standing', '#9a9a9a'),
             ('none', 'no hint at all', '#333333'),
             ('oracle', 'only what the task still needs', '#c9902b'),
             ('frontier', 'only how many steps done', '#6f9c4e'),
             ('wrongproduct', 'both, but the ending judged wrong', '#a2543c'),
             ('product', 'both, judged right', '#1b6ca8')]
    vals = [statistics.mean(cap(g[(m['tag'], a)]) for m in metas) for a, _, _ in order]
    y = list(range(len(order)))[::-1]
    bx.barh(y, vals, color=[c for _, _, c in order])
    for yy, v in zip(y, vals):
        bx.text(v + 700, yy, f'{v:,.0f}', va='center', fontsize=8.5)
    bx.set_yticks(y)
    bx.set_yticklabels([l for _, l, _ in order], fontsize=9)
    bx.set_xlim(0, max(vals) * 1.18)
    bx.set_xlabel('practice steps needed before it can do the job, twelve tasks averaged')
    bx.set_title('What each kind of hint is worth', fontsize=11)
    bx.grid(alpha=.25, axis='x')
    for ext in ('png', 'svg'):
        plt.savefig(ROOT / f'figures/worked_example.{ext}', dpi=150)
    print('wrote figures/worked_example.png and .svg')


if __name__ == '__main__':
    main(*sys.argv[1:])
