from search_maps import *

def main():
    out=ROOT/'results/map_search_extended';out.mkdir(exist_ok=False)
    start=time.monotonic();best=None;records=[]
    for side in [13,17]:
        for L in [4,8,12,16]:
            for rep in range(100):
                seed=91000+side*1000+L*100+rep;layout=build(4,L,seed,side);s=summarize(layout,8)
                rec=dict(side=side,n=4,L=L,seed=seed,m=8,orders=s['orders'],compression=s['compression'],max_length=max(s['optimal_lengths'],default=0));records.append(rec)
                if s['orders']>1 and 5.4<=s['compression']<=6.6 and (best is None or (rec['max_length'],abs(s['compression']-6))<best[0]):best=((rec['max_length'],abs(s['compression']-6)),layout,s)
            print(side,L,'max ratio',max(r['compression'] for r in records),'bin6',bool(best),'seconds',round(time.monotonic()-start,1),flush=True)
    (out/'candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    if best:
        _,layout,s=best;assert funnel_check(layout);(out/'layout_6.json').write_text(json.dumps({**layout,**s,'target_bin':6,'funnel_verified':True},indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(layouts=len(records),max_compression=max(r['compression'] for r in records),reached_bin6=bool(best),seconds=time.monotonic()-start),indent=2)+'\n')
if __name__=='__main__':main()
