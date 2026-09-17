"""Convert a learner's tree dump into the plain table the learner can read back.

Only the map-independent part travels: which event advanced the task after which
history, which histories ended the task, and how often each event was tried
there. Nothing indexed by a cell is written.
"""
import json, sys
from pathlib import Path


def convert(dump_path, out_path):
    tree = json.loads(Path(dump_path).read_text())['tree']
    nodes, alphabet = tree['nodes'], tree['alphabet']
    lines = [f'{alphabet} {len(nodes)}']
    for nd in nodes:
        lines.append(' '.join(map(str, [nd['visits'], nd['accepting']]
                                  + nd['advanced'] + nd['ignored']
                                  + nd.get('fatal', [0] * alphabet)
                                  + nd['succ'])))
    Path(out_path).write_text('\n'.join(lines) + '\n')
    return len(nodes), sum(1 for nd in nodes if nd['accepting'])


if __name__ == '__main__':
    n, acc = convert(sys.argv[1], sys.argv[2])
    print(f'{n} nodes, {acc} accepting')
