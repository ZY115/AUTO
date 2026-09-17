from pathlib import Path
import sys,itertools,json,importlib.metadata
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/vendor'))
from aalpy.learning_algs import run_RPNI
# Input events; output distinguishes ignored/progress/success/failure.
# Goal: A then B; B before A is fatal, terminal outputs stay terminal.
def oracle(word):
 q=0;out=[]
 for e in word:
  nq=q
  if q==0:nq=1 if e=='A' else (3 if e=='B' else 0)
  elif q==1:nq=2 if e=='B' else 1
  o='failure' if nq==3 else ('success' if nq==2 else ('progress' if nq!=q else 'ignored'))
  out.append(o);q=nq
 return out
alphabet=['A','B']
data=[(w,oracle(w)[-1]) for n in range(1,6) for w in itertools.product(alphabet,repeat=n)]
results=[]
for alg in ['classic','gsm']:
 model=run_RPNI(data,'mealy',algorithm=alg,input_completeness=None,print_info=False)
 mismatches=0;checked=0;undefined=0
 for n in range(1,8):
  for w in itertools.product(alphabet,repeat=n):
   model.reset_to_initial()
   try: got=[model.step(e) for e in w]
   except KeyError: got=None;undefined+=1
   mismatches+=got!=oracle(w);checked+=1
 results.append({'algorithm':alg,'states':len(model.states),'training_words':len(data),'checked_words':checked,'mismatches':mismatches,'undefined_words':undefined,'input_complete':model.is_input_complete()})

out={'aalpy_version':importlib.metadata.version('aalpy'),'purpose':'Independent library API and deterministic feedback-encoding smoke test, NOT Stage 8 performance benchmark','results':results}
(ROOT/'audit/aalpy_smoke.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
