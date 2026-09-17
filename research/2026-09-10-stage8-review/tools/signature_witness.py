from pathlib import Path
import json
# Same finite-depth information as the project's three refinements, represented
# by tuples instead of hashes so accidental hash collisions are not involved.
def sig(accept):
 h=[(int(i in accept),1,0) for i in range(6)]
 for _ in range(3): h=[(h[i],h[min(i+1,5)]) for i in range(6)]
 return h[0]
# A accepts after aaaa, B accepts after aaaaa. Both advance on a before terminal.
a,b={4,5},{5}
# Only root radius <= 3 matters: neither acceptance nor terminal boundary seen.
assert sig(a)==sig(b)
r={'three_round_root_signatures_equal':True,'distinguishing_suffix':'aaaa','A_accepts':True,'B_accepts':False,'purpose':'Finite-depth semantic counterexample, not a benchmark of the project JIRP arm'}
Path(__file__).resolve().parents[1].joinpath('audit/signature_witness.json').write_text(json.dumps(r,indent=2)+'\n')
