"""Write the selected layouts as compact task files the C++ learner reads.

One file per ladder rung. Everything the learner needs is pre-computed here so
the task semantics live in one place: the transition table over task states,
the Myhill-Nerode class of each state used by the oracle arm, the map, the
balanced start set, and the exact optimal length from each start.
"""
import itertools, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from family_maps import Task, evaluate_layout, product_shortest
from search_maps import build, funnel_check

ROOT = Path(__file__).resolve().parents[1]


def make_layout(cfg):
    """Rebuild the selected layout from its recorded parameters."""
    if cfg.get('forked'):
        from spoke_maps import build_forked, funnel_ok
        task = Task(cfg['n'], tuple(cfg['pair']), cfg['d'], cfg['m'], cfg.get('m2'))
        layout = build_forked(cfg['n'], cfg['L'], cfg['seed'], arm=cfg['arm'],
                              bridge=task.bridge, tails=task.tails)
        assert layout is not None and funnel_ok(layout), 'junction is not a cut vertex'
        return layout
    if 'arm' in cfg:
        from spoke_maps import build_spokes, funnel_ok
        layout = build_spokes(cfg['n'], cfg['L'], cfg['seed'], arm=cfg['arm'])
        assert layout is not None and funnel_ok(layout), 'junction is not a cut vertex'
        return layout
    layout = build(cfg['n'], cfg['L'], cfg['seed'], side=cfg['side'])
    assert funnel_check(layout), 'junction is not a cut vertex'
    return layout


def emit(cfg, outdir):
    n, d, m = cfg['n'], cfg['d'], cfg['m']
    layout = make_layout(cfg)
    task = Task(n, tuple(cfg['pair']), d, m, cfg.get('m2'), fatal=bool(cfg.get('fatal')))
    info = evaluate_layout(layout, task, 24)
    assert info is not None and info['count_aliased']
    if 'ratio' in cfg:
        assert abs(info['honest_ratio'] - cfg['ratio']) < 1e-9, (info['honest_ratio'], cfg['ratio'])
    dist, N = product_shortest(layout, task)
    starts = info['starts']
    lengths = [dist[s] for s in starts]
    horizon = 2 * max(lengths)
    tag = cfg.get('tag') or f"rung_{cfg['target']:g}"
    lines = [f"{n} {d} {m} {task.pair[0]} {task.pair[1]} {task.alphabet}",
             f"{len(task.states)}",
             ' '.join(str(task.minimal[i]) for i in range(len(task.states))),
             ' '.join('1' if i in task.accepting else '0' for i in range(len(task.states))),
             ' '.join('1' if i in task.failing else '0' for i in range(len(task.states)))]
    for row in task.trans:
        lines.append(' '.join(map(str, row)))
    lines.append(str(N))
    for row in layout['moves']:
        lines.append(' '.join(f'{ns} {ev}' for ns, ev in row))
    lines.append(str(len(starts)))
    lines.append(' '.join(map(str, starts)))
    lines.append(' '.join(map(str, lengths)))
    lines.append(f"{horizon} {max(lengths)} {info['min_sufficient_window']}")
    # Distance in the product of the task machine and the map: environment steps
    # still owed from (task state, cell). A task-only potential cannot express
    # this, and these layouts have physical routes of very different lengths.
    # A failing state can never reach acceptance, so it has no finite distance.
    # It is written as a value beyond any real route rather than as zero, which
    # would read as "already finished".
    far = 4 * max(lengths) + 4
    clean = [d if d >= 0 else far for d in dist]
    assert all(clean[q * N + s] >= 0 for q in range(len(task.states)) for s in range(N))
    for q in range(len(task.states)):
        if q in task.failing:
            for s in range(N):
                clean[q * N + s] = far
    lines.append(' '.join(str(clean[q * N + s]) for q in range(len(task.states))
                          for s in range(N)))
    # The same distance under a task whose branch assignment is inverted: same
    # alphabet, same map, same lengths, wrong answer. A potential that does as
    # well with this table is not being helped by the task structure.
    wrong = Task(n, (task.pair[1], task.pair[0]), d, m, cfg.get('m2'),
                 fatal=bool(cfg.get('fatal')))
    wdist, wN = product_shortest(layout, wrong)
    assert wN == N
    wclean = [d if d >= 0 else far for d in wdist]
    lines.append(' '.join(str(wclean[q * N + s]) for q in range(len(wrong.states))
                          for s in range(N)))
    # The same numbers attached to the wrong task states. Structured, correct in
    # its marginals, and meaningless as a map of where the agent stands. If a
    # potential still helps here, nothing about the task is doing the work.
    rng = random.Random(20260910)
    order = list(range(len(task.states)))
    rng.shuffle(order)
    lines.append(' '.join(str(dist[order[q] * N + s]) for q in range(len(task.states))
                          for s in range(N)))
    (outdir / f'{tag}.task').write_text('\n'.join(lines) + '\n')
    meta = dict(cfg, tag=tag, horizon=horizon, lengths=lengths, starts=starts,
                m2=task.m2, tail_lengths=[len(t) for t in task.tails],
                fatal=bool(cfg.get('fatal')), failing_states=len(task.failing),
                history_states=info['history_states'], minimal_states=info['minimal_states'],
                denominator=info['denominator'], denominator_states=info['denominator_states'],
                min_sufficient_window=info['min_sufficient_window'],
                honest_ratio=info['honest_ratio'], orders=info['orders'],
                feature_sizes=info['feature_sizes'], feature_sufficient=info['feature_sufficient'],
                optimal_histories=info['optimal_histories'],
                task_states=len(task.states), minimal_total=task.n_minimal,
                bridge=task.bridge, tails=task.tails)
    (outdir / f'{tag}.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


def main(selection='results/family_maps/ladder.json'):
    outdir = ROOT / 'results/family_maps'
    outdir.mkdir(parents=True, exist_ok=True)
    seen, metas = set(), []
    for cfg in json.loads((ROOT / selection).read_text()):
        key = (cfg.get('arm'), cfg.get('side'), cfg['L'], cfg['seed'], tuple(cfg['pair']))
        if key in seen:
            continue
        seen.add(key)
        meta = emit(cfg, outdir)
        metas.append(meta)
        print(f"{meta['tag']}: ratio={meta['honest_ratio']:.2f} orders={meta['orders']} "
              f"|H|={meta['history_states']} |Q|={meta['minimal_states']} "
              f"denom={meta['denominator']}({meta['denominator_states']}) "
              f"minw={meta['min_sufficient_window']} H={meta['horizon']} len<={max(meta['lengths'])}")
    (outdir / 'rungs.json').write_text(json.dumps(metas, indent=2) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
