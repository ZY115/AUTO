"""Independent bounded BFS versus the existing DFS; does not change task code."""
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'progressive_task_discovery/stage8_compression'
sys.path.insert(0, str(ROOT / 'src'))
from taskfile import TaskFile


def bfs(t):
    seen = {(s, ()) for s in t.starts}
    queue = collections.deque((s, (), 0) for s in t.starts)
    out = {()}
    while queue:
        s, h, d = queue.popleft()
        if d >= t.horizon:
            continue
        q = t.run(h)
        if t.accepting[q] or t.failing[q]:
            continue
        for a in range(4):
            ns, e = t.moves[s][a], t.events[s][a]
            nh = h
            if e >= 0:
                nq = t.trans[q][e]
                if t.failing[nq]:
                    continue
                if nq != q:
                    nh = h + (e,)
            key = (ns, nh)
            if key in seen:
                continue
            seen.add(key)
            out.add(nh)
            queue.append((ns, nh, d+1))
    return out


if __name__ == '__main__':
    rows = []
    for folder in ['overlap', 'irreversible']:
        for path in sorted((ROOT/'results'/folder).glob('*.task')):
            t = TaskFile(path)
            a, b = t.reachable_histories(), bfs(t)
            rows.append(dict(path=str(path), dfs=len(a), bfs=len(b),
                             missing=len(b-a), extra=len(a-b),
                             witness=list(min(b-a)) if b-a else None))
    summary = dict(maps=len(rows), affected=sum(r['missing']>0 for r in rows),
                   dfs_total=sum(r['dfs'] for r in rows),
                   bfs_total=sum(r['bfs'] for r in rows),
                   source_sha256=hashlib.sha256((ROOT/'src/compress.cpp').read_bytes()).hexdigest())
    Path(__file__).with_suffix('.json').write_text(json.dumps(dict(summary=summary, maps=rows), indent=2)+'\n')
    print(json.dumps(summary))
