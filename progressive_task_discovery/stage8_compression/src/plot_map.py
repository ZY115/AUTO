"""Draw one rung: the physical map, and the task rule that runs on it."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from emit_tasks import make_layout

GOAL = '#1b6ca8'
MARK = '#c9902b'
START = '#6f9c4e'
PLAIN = '#e8e8e8'
HUB = '#555555'


def main(tag='rung_1.8'):
    meta = {m['tag']: m for m in json.loads((ROOT / 'results/family_maps/rungs.json').read_text())}[tag]
    layout = make_layout(meta)
    cells = [tuple(c) for c in layout['cells']]
    labels = {c: g for c, g in layout['labels']}
    starts = set(meta['starts'])
    n = meta['n']
    nm = lambda g: 'ABCDE'[g] if g < n else 'XYZ'[g - n]

    fig = plt.figure(figsize=(12, 5.2))
    ax = fig.add_axes([.03, .06, .60, .86])
    for i, (x, y) in enumerate(cells):
        if i in labels:
            g = labels[i]
            col, txt = (GOAL, nm(g)) if g < n else (MARK, nm(g))
        elif i in starts:
            col, txt = START, 'S'
        elif (x, y) == (0, 0):
            col, txt = HUB, ''
        else:
            col, txt = PLAIN, ''
        ax.add_patch(Rectangle((x - .45, -y - .45), .9, .9, facecolor=col,
                               edgecolor='white', linewidth=1.4))
        if txt:
            ax.text(x, -y, txt, ha='center', va='center', fontsize=11,
                    color='white', fontweight='bold')
    ax.text(0, 0, 'H', ha='center', va='center', fontsize=10, color='white')
    junction = cells[layout['junction']]
    ax.add_patch(Circle((junction[0], -junction[1]), .58, fill=False,
                        edgecolor='#c05640', linewidth=2, linestyle='--'))
    ax.text(junction[0] + .9, -junction[1], 'the only way out\nevery history passes here',
            fontsize=8.5, va='center', color='#c05640')
    xs = [c[0] for c in cells]
    ys = [-c[1] for c in cells]
    ax.set_xlim(min(xs) - 1.4, max(xs) + 5.2)
    ax.set_ylim(min(ys) - 2.4, max(ys) + 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'{tag}: 31 walkable cells, three labelled corridors and one exit',
                 fontsize=11)
    for col, lab, xoff in [(GOAL, 'A B C D  the four goals, any order', 0),
                           (MARK, 'X Y Z  the later marks', 7.6),
                           (START, 'S  the five forced starting points', 13.2)]:
        ax.add_patch(Rectangle((min(xs) - .5 + xoff, min(ys) - 1.6), .8, .8,
                               facecolor=col, edgecolor='white'))
        ax.text(min(xs) + .6 + xoff, min(ys) - 1.2, lab, fontsize=8.5, va='center')

    bx = fig.add_axes([.66, .06, .32, .86])
    bx.axis('off')
    bx.set_xlim(0, 10)
    bx.set_ylim(0, 10)
    bx.set_title('the hidden rule the agent is never told', fontsize=11)
    steps = [(8.9, 'press A, B, C, D', 'in any order you like', GOAL),
             (7.0, 'the environment remembers', 'which of A or C you pressed first', '#7d5ba6'),
             (5.1, 'then press X, Y, Z', 'the same for both branches', MARK),
             (3.2, 'then press X', 'only if A came before C', '#c05640'),
             (1.9, 'or press Y', 'only if C came before A', '#c05640'),
             (0.4, 'reward arrives once, at the end', 'nothing before that', HUB)]
    for y, head, sub, col in steps:
        bx.add_patch(Rectangle((.3, y - .05), 9.2, 1.1, facecolor=col, alpha=.13,
                               edgecolor=col, linewidth=1.2))
        bx.text(.6, y + .68, head, fontsize=10, fontweight='bold', color=col)
        bx.text(.6, y + .22, sub, fontsize=8.6, color='#444444')
    for y0, y1 in [(8.9, 8.1), (7.0, 6.2), (5.1, 4.3)]:
        bx.add_patch(FancyArrowPatch((5, y0), (5, y1), arrowstyle='-|>',
                                     mutation_scale=11, color='#999999', lw=1.1))
    for ext in ('png', 'svg'):
        plt.savefig(ROOT / f'figures/task_map.{ext}', dpi=150)
    print('wrote figures/task_map.png and .svg')


if __name__ == '__main__':
    main(*sys.argv[1:])
