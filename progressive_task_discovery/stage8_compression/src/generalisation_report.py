import json, random, statistics
from collections import defaultdict

rows = [json.loads(l) for l in open('results/generalisation.jsonl')]
ARTS = ['dictionary', 'machine', 'oracle_merge']
SIZES = ['少量', '中等', '较充分']


def share(sel):
    c = defaultdict(int)
    for r in sel:
        c[r['outcome']] += 1
    n = sum(c.values()) or 1
    return c['correct'] / n, c['wrong'] / n, c['unknown'] / n, n


print('目标地图可达、源数据未见过的历史（这是问题的核心集合）')
print(f"{'源预算':>7} {'构件':>13} {'回答正确':>8} {'回答错误':>8} {'回答不了':>8} {'问题数':>8}")
for size in SIZES:
    for a in ARTS:
        c, w, u, n = share([r for r in rows if r['novel'] and r['size'] == size
                            and r['artefact'] == a])
        print(f"{size:>7} {a:>13} {c:>8.3f} {w:>8.3f} {u:>8.3f} {n:>8}")
    print()

print('源数据见过的历史（对照：这里字典本来就该会）')
print(f"{'源预算':>7} {'构件':>13} {'回答正确':>8} {'回答错误':>8} {'回答不了':>8}")
for size in SIZES:
    for a in ARTS:
        c, w, u, n = share([r for r in rows if not r['novel'] and r['size'] == size
                            and r['artefact'] == a])
        print(f"{size:>7} {a:>13} {c:>8.3f} {w:>8.3f} {u:>8.3f}")
    print()

print('按顺序重叠度分档，新历史上，源预算=较充分')
print(f"{'重叠':>6} {'构件':>13} {'正确':>7} {'错误':>7} {'答不了':>7} {'问题数':>7}")
for band in (1.0, 0.5, 0.33, 0.0):
    for a in ARTS:
        c, w, u, n = share([r for r in rows if r['novel'] and r['size'] == '较充分'
                            and r['artefact'] == a and abs(r['overlap'] - band) < .01])
        print(f"{band:>6} {a:>13} {c:>7.3f} {w:>7.3f} {u:>7.3f} {n:>7}")
    print()

# Paired bootstrap over map pairs: how much of the dictionary's blind spot does
# the machine cover, and is anything it says there wrong?
pairs = sorted({r['pair'] for r in rows})
by = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r['novel']:
        by[(r['size'], r['artefact'])][r['pair']].append(r['outcome'])


def rate(size, art, sample, kind):
    tot = hit = 0
    for p in sample:
        v = by[(size, art)][p]
        tot += len(v)
        hit += sum(1 for x in v if x == kind)
    return hit / tot if tot else 0.0


rnd = random.Random(73)
print('新历史上的作答率与错误率，块=地图对，25 块')
print(f"{'源预算':>7} {'构件':>13} {'作答率':>22} {'错误率(占全部问题)':>24}")
for size in SIZES:
    for a in ARTS:
        pt_c = rate(size, a, pairs, 'correct')
        pt_w = rate(size, a, pairs, 'wrong')
        dc, dw = [], []
        for _ in range(2000):
            s = [rnd.choice(pairs) for _ in pairs]
            dc.append(rate(size, a, s, 'correct'))
            dw.append(rate(size, a, s, 'wrong'))
        dc.sort(); dw.sort()
        print(f"{size:>7} {a:>13} {pt_c:>8.3f} [{dc[50]:.3f}, {dc[1950]:.3f}]"
              f"   {pt_w:>8.3f} [{dw[50]:.3f}, {dw[1950]:.3f}]")
    print()
