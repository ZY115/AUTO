from search_maps import *

def witnesses(layout):
    n=layout['n'];N=len(layout['cells']);J=layout['junction']
    seen={(s,()) for s in layout['starts']};queue=deque(seen);groups=[{} for _ in range(n)];found={}
    while queue:
        s,h=queue.popleft()
        if s==J and len(h)<n:
            mask=sum(1<<e for e in h)
            for w in range(n):
                key=(len(h),h[-w:] if w else ())
                previous=groups[w].get(key)
                if previous and previous[0]!=mask:found[w]=dict(position=J,depth=len(h),window=w,first_history=previous[1],second_history=h,first_mask=previous[0],second_mask=mask)
                groups[w][key]=(mask,h)
        if len(h)==n:continue
        for ns,e in layout['moves'][s]:
            nh=h+(e,) if 0<=e<n and e not in h else h
            if (ns,nh) not in seen:seen.add((ns,nh));queue.append((ns,nh))
    assert n-1 not in found
    assert all(w in found for w in range(n-1)), ('missing short-window witness',layout['target_bin'],found.keys())
    return {'reachable_history_position_pairs':len(seen),'short_window_witnesses':found,'window_n_minus_1_sufficient':True}

def optimal_funnel(layout):
    dist,_=product(layout,layout['suffix']);N=len(layout['cells']);full=(1<<layout['n'])-1
    seen=set(layout['starts']);queue=deque(seen);hits=set()
    while queue:
        z=queue.popleft();q,s=divmod(z,N)
        if s==layout['junction']:
            assert q==full, ('optimal funnel not at completed unordered set',q,full)
            hits.add(q)
        if dist[z]==0:continue
        for ns,event in layout['moves'][s]:
            dest=nextq(q,event,layout['n'],layout['suffix'])*N+ns
            if dist[dest]==dist[z]-1 and dest not in seen:seen.add(dest);queue.append(dest)
    assert hits=={full}
    assert funnel_check(layout)
    return {'all_optimal_junction_states':[full],'optimal_product_states':len(seen),'cut_vertex_verified':True}

def main():
    paths=sorted((ROOT/'results/map_search').glob('layout_*.json'))+list((ROOT/'results/map_search_extended').glob('layout_*.json'))
    results=[];pack=ROOT/'docs/layouts';pack.mkdir(exist_ok=True)
    for path in paths:
        l=json.loads(path.read_text());r=summarize(l,l['m']);assert r['optimal_histories']==[tuple(h) for h in l['optimal_histories']]
        assert r['compression']==l['compression'];v=dict(target=l['target_bin'],**witnesses(l),**optimal_funnel(l));results.append(v)
        # Text format for dependency-free C++ ingestion; JSON is authoritative.
        N=len(l['cells']);label=dict(l['labels']);lines=[f"{N} {l['n']} {l['m']} {len(l['starts'])} {max(l['optimal_lengths'])}", ' '.join(map(str,l['suffix'])),' '.join(map(str,l['starts']))]
        for s,row in enumerate(l['moves']):lines.append(' '.join(map(str,[label.get(s,-1)]+[v for pair in row for v in pair])))
        (pack/f"layout_{l['target_bin']:g}.txt").write_text('\n'.join(lines)+'\n')
        (pack/f"layout_{l['target_bin']:g}.json").write_text(json.dumps(l,indent=2)+'\n')
        print('validated bin',l['target_bin'],'ratio',r['compression'],'orders',r['orders'],'max path',max(r['optimal_lengths']),flush=True)
    (ROOT/'results/map_validation.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
