from pathlib import Path
import json, hashlib, statistics as st, random, subprocess, collections
ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT.parents[1]/'progressive_task_discovery/stage8_compression'
files=['goal_vs_index.jsonl','priorart.jsonl','mergerule.jsonl','revision.jsonl','irreversible/goal.jsonl','overlap_run2/raw.jsonl','irreversible/dose.jsonl','rawhistory.jsonl']
summary={}
for name in files:
 p=STAGE/'results'/name
 if not p.exists(): continue
 rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
 arms=collections.defaultdict(list)
 for r in rows: arms[r.get('arm',r.get('rev',r.get('method','?')))].append(r)
 report={}
 for a,rs in arms.items():
  out={'n':len(rs)}
  if name=='goal_vs_index.jsonl':out['first90_censored_mean']=st.mean(r['first90'] if r['first90']>0 else 100000 for r in rs)
  for k in ['first90','auc','final','final_success','failures','doomed','features','nodes','updates','route']:
   vs=[r[k] for r in rs if isinstance(r.get(k),(int,float))]
   if vs: out[k+'_mean']=st.mean(vs)
  out['final_ge_0.9']=sum(r.get('final',r.get('final_success',0))>=.9 for r in rs)
  report[a]=out
 summary[name]={'rows':len(rows),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'arms':report}
(ROOT/'audit/data_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
# Correct map-block bootstrap: repeated sampled maps retain their multiplicity.
rows=[json.loads(l) for l in (STAGE/'results/irreversible/goal.jsonl').read_text().splitlines()]
maps=sorted({r['map'] for r in rows}); means={(a,m):st.mean(r['first90'] for r in rows if r['arm']==a and r['map']==m) for a in {r['arm'] for r in rows} for m in maps}
rng=random.Random(23); result=[]
for x,y in [('goal_value','automaton'),('goal_learned','goal_value'),('goal_learned','merged'),('goal_learned','history')]:
 def rat(ms): return sum(means[y,m] for m in ms)/sum(means[x,m] for m in ms)
 ds=sorted(rat(rng.choices(maps,k=len(maps))) for _ in range(10000))
 result.append({'denominator':x,'numerator':y,'ratio':rat(maps),'ci95':[ds[250],ds[9750]]})
(ROOT/'audit/corrected_fatal_bootstrap.json').write_text(json.dumps(result,indent=2)+'\n')
# Pure observational instrumentation, no learning/control changes.
src=(STAGE/'src/compress.cpp').read_text()
patches=[
 ('long long failures=0,doomed_steps=0;', 'long long audit_entries=0,audit_sink_steps=0,audit_sink_events=0,audit_sink_conflicts=0;long long failures=0,doomed_steps=0;'),
 ('bool fell=T.failing[nq]!=0;', 'bool fell=T.failing[nq]!=0; if(fell&&!T.failing[qstate])audit_entries++; if(T.failing[qstate])audit_sink_steps++;'),
 ('if(L.tracking&&ev>=0&&!progress){', 'if(L.tracking&&ev>=0&&!progress){ if(T.failing[qstate]){audit_sink_events++;if(L.tree.advanced[node][ev]>0)audit_sink_conflicts++;}'),
 ('// Diagnostics compare the learner', 'std::cerr<<"{\\"failure_entries\\":"<<audit_entries<<",\\"post_failure_steps\\":"<<audit_sink_steps<<",\\"post_failure_events_written_to_healthy_node\\":"<<audit_sink_events<<",\\"contradictions_with_prior_progress\\":"<<audit_sink_conflicts<<"}\\n";\n    // Diagnostics compare the learner')]
for old,new in patches:
 assert src.count(old)==1,(old,src.count(old))
 src=src.replace(old,new)
copy=ROOT/'audit/compress_observed.cpp';copy.write_text(src)
subprocess.run(['clang++','-std=c++17','-O2',str(copy),'-o',str(ROOT/'audit/compress_observed')],check=True)
subprocess.run(['clang++','-std=c++17','-O2',str(STAGE/'src/compress.cpp'),'-o',str(ROOT/'audit/compress_original')],check=True)
entry=json.loads((STAGE/'results/irreversible/maps.json').read_text())[0]
args=['--task',str(STAGE/'results/irreversible'/f"{entry['fatal']['tag']}.task"),'--seed','5000','--budget','200000','--every','500','--doom-continues','1','--method','goal','--goal-select','learned']
a=subprocess.run([str(ROOT/'audit/compress_observed')]+args,text=True,capture_output=True,check=True)
b=subprocess.run([str(ROOT/'audit/compress_original')]+args,text=True,capture_output=True,check=True)
assert json.loads(a.stdout)==json.loads(b.stdout),'instrumentation changed output'
out={'command_args':args,'source_sha256':hashlib.sha256((STAGE/'src/compress.cpp').read_bytes()).hexdigest(),'instrumented_equals_original_all_output_fields':True,'diagnostics':json.loads(a.stderr),'original_result':json.loads(b.stdout)}
original=next(r for r in rows if r['map']==entry['fatal']['tag'] and r['seed']==5000 and r['arm']=='goal_learned')
out['stored_result']=original
out['stored_fields_reproduced']={k:original[k]==out['original_result'][key] for k,key in [('auc','auc'),('final','final_success'),('failures','failures'),('doomed','doomed_steps'),('nodes','nodes')]}
(ROOT/'audit/irreversible_witness.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='original_result'},indent=2))
print('data summaries and corrected bootstrap written')
