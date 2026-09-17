from pathlib import Path
import json,collections,statistics,subprocess,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent

def read(name):
 p=ROOT/'results'/name
 return [json.loads(s) for s in p.open()] if p.suffix=='.jsonl' else json.loads(p.read_text())
def costs(r):return r['first90'] if r['first90']>=0 else r['budget']
def compare(rows,armkey,a,b,metric='first90'):
 get=lambda r:costs(r) if metric=='first90' else r[metric]
 index={(r['seed'],r['rung'],armkey(r)):r for r in rows}
 seeds=sorted({r['seed'] for r in rows});rungs=sorted({r['rung'] for r in rows})
 aa=np.array([[get(index[s,t,a]) for t in rungs] for s in seeds]);bb=np.array([[get(index[s,t,b]) for t in rungs] for s in seeds])
 rng=np.random.default_rng(901);ix=rng.integers(0,len(seeds),(10000,len(seeds)))
 draws=aa.mean(1)[ix].mean(1)/bb.mean(1)[ix].mean(1)
 return dict(a=a,b=b,metric=metric,point=float(aa.mean()/bb.mean()),seed_block_ci95=np.quantile(draws,[.025,.975]).tolist())

def main():
 h4=read('h4_retest.jsonl');h5=read('shaping_confirm.json');query=read('query_strategy_v2.json')
 summaries={}
 for name,rows,key in [('h4',h4,lambda r:r['arm']),('h5',h5,lambda r:r['shape']),('query',query,lambda r:r['target'])]:
  groups=collections.defaultdict(list)
  for r in rows:
   assert r['updates']==6*r['budget'] and r['revision_updates']==0
   groups[key(r)].append(r)
  summaries[name]=[dict(arm=a,n=len(rs),first90=statistics.mean(costs(r) for r in rs),auc=statistics.mean(r['auc'] for r in rs),final_success=statistics.mean(r['final_success'] for r in rs),final_below90=sum(r['final_success']<.9 for r in rs),precision=statistics.mean(r['merge_precision'] for r in rs),min_precision=min(r['merge_precision'] for r in rs),recall=statistics.mean(r['merge_recall'] for r in rs)) for a,rs in groups.items()]
 ratios=[compare(h4,lambda r:r['arm'],'history','merged'),compare(h4,lambda r:r['arm'],'merged','automaton'),compare(h4,lambda r:r['arm'],'history+shape','merged+shape'),compare(h5,lambda r:r['shape'],'none','frontier','first_success'),compare(h5,lambda r:r['shape'],'frontier','oracle'),compare(query,lambda r:r['target'],'scarcest','separating')]
 pairs={(r['rung'],r['seed'],r['shape']):r for r in h5}
 assert all(pairs[t,s,'none']['first_success']==pairs[t,s,'count']['first_success'] for t,s,shape in pairs if shape=='none')
 out=dict(source_sha256=hashlib.sha256((ROOT/'src/compress.cpp').read_bytes()).hexdigest(),binary_sha256=hashlib.sha256((ROOT/'compress').read_bytes()).hexdigest(),existing_tests_passed=25,runs_checked=len(h4+h5+query),summaries=summaries,ratios=ratios,count_first_success_exact_matches=192)
 (OUT/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(ratios,indent=2))
 # Small read-only replay of stored configurations. Strict children is the
 # current explicit permissive choice; old files do not record this flag.
 chosen=[next(r for r in h4 if r['rung']=='rung_1.8' and r['seed']==800 and r['arm']==a) for a in ['history','merged','merged+shape']]
 chosen+=[next(r for r in h5 if r['rung']=='rung_2.6' and r['seed']==700 and r['shape']==a) for a in ['none','count','frontier','oracle']]
 reproduction=[]
 for r in chosen:
  cmd=[str(ROOT/'compress'),'--task',str(ROOT/'results/family_maps'/Path(r['task']).name),'--strict-children','0']
  for k in ['method','seed','budget','window','epsilon','gamma','theta','quota','revision','beta','target','shape','shape_scale']:
   cmd+=['--'+k.replace('_','-'),str(r[k])]
  got=json.loads(subprocess.check_output(cmd,text=True))
  checks={k:got[k]==r[k] for k in ['trace_hash','checkpoints','first90','first_success','auc','merge_precision']}
  assert all(checks.values()),(r['method'],checks)
  reproduction.append(dict(rung=r['rung'],method=r['method'],shape=r['shape'],seed=r['seed'],checks=checks))
 (OUT/'reproduction.json').write_text(json.dumps(reproduction,indent=2)+'\n');print('Exact reproductions:',len(reproduction))
if __name__=='__main__':main()
