"""Native end-to-end collapse check and uncollapsed multi-state smoke run.

All runs are recorded separately. This checks implementation equivalence, not
statistical learning superiority. Model/log outputs here are test artifacts.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from native_hrm.y10 import IHSAAlgorithmHRLPerState, StateScopedFormulaBank
from native_hrm.tests.test_y10 import RecordingBank
from reinforcement_learning.ihsa_hrl_tabular_algorithm import IHSAAlgorithmHRLTabular


class Traced:
    def __init__(self,params):
        self.trace=[]
        self.transitions=[]
        self.update_trace=[]
        self.episodes_trace=[]
        super().__init__(params)

    def _init_formula_q_functions(self):
        for bank in self._formula_banks:
            bank.__class__=RecordingBank
            bank.selections=[];bank.rewards=[]
        super()._init_formula_q_functions()

    def _choose_action(self,domain,task,state,hierarchy,hs):
        action=super()._choose_action(domain,task,state,hierarchy,hs)
        self.trace.append((bool(self.training_enable),domain,task,int(state),int(action),
                           hs.automaton_name,hs.state_name))
        self.pre_context=(hs.automaton_name,hs.state_name)
        return action

    def _update_q_functions(self,domain,task,state,action,next_state,terminal,obs):
        b=self._get_policy_bank(task)
        if isinstance(b,StateScopedFormulaBank):
            expected=('*','*') if b.collapse else self.pre_context
            assert b.context==expected
            current=b.current_bank
        else:current=b
        start=len(current.rewards)
        super()._update_q_functions(domain,task,state,action,next_state,terminal,obs)
        self.update_trace.append((current.selections[-1],current.rewards[start:]))
        self.transitions.append((domain,task,int(state),int(action),int(next_state),
                                 bool(terminal),sorted(obs)))

    def _run_episode(self,domain,task):
        result=super()._run_episode(domain,task)
        self.episodes_trace.append((bool(self.training_enable),result))
        return result


class NativeTraced(Traced,IHSAAlgorithmHRLTabular):pass
class ScopedTraced(Traced,IHSAAlgorithmHRLPerState):pass


def table_hash(table):
    return hashlib.sha256(table.tobytes()).hexdigest()


def bank_snapshot(bank):
    return dict(tables={str(g):table_hash(q) for g,q in bank._q_functions.items()},
                steps={str(g):v for g,v in bank._q_function_step_counter.items()},
                updates={str(g):v for g,v in bank._q_function_update_counter.items()},
                calls=bank._num_update_calls)


def snapshot(a):
    banks=[]
    for b in a._formula_banks:
        if isinstance(b,StateScopedFormulaBank):
            banks.append({str(k):bank_snapshot(v) for k,v in b.banks.items()})
        else:banks.append(bank_snapshot(b))
    meta=[]
    for tid,automata in a._meta_q_functions.items():
        for name,states in automata.items():
            for (q,ctx),table in states.items():meta.append((tid,name,q,str(ctx),table_hash(table)))
    return dict(banks=banks,meta=sorted(meta))


def config(task,folder,episodes,seed):
    c=json.loads((ROOT/'external/hrm-learning/src/config/examples/ihsa/07-cw-frl-bq-exploit-flat/config.json').read_text())
    c.update(debug=False,folder_name=str(folder),checkpoint_folder=str(folder),
             state_format='tabular',num_environment_tasks=1,num_episodes=episodes,
             max_episode_length=120,seed=seed,training_mode='handcrafted',use_flat_hierarchy=True,
             formula_update_sel_num=2,greedy_evaluation_frequency=20,
             exploration_rate_subgoal_steps=2000,exploration_rate_automaton_steps=1000,
             learning_rate=.1,meta_learning_rate=.1)
    c['environments']=[dict(name=task,automaton_name='m0',hierarchy_level=1,
                            dependencies=[],starting_seed=seed)]
    c['grid_params'].update(grid_type='open_plan',width=7,height=7,size=7,use_lava=False,
                            use_lava_walls=False,max_objs_per_class=1)
    return c


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--episodes',type=int,default=160);p.add_argument('--seed',type=int,default=17)
    args=p.parse_args();out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for task in ('book','book-and-quill','cake'):
        runs=[]
        for mode in ('native','collapsed','perstate'):
            c=config(task,out/task/mode,args.episodes,args.seed)
            c.update(algorithm='ihsa-hrl' if mode=='native' else 'ihsa-hrl-perstate',
                     skill_state_collapse=mode=='collapsed')
            (out/f'{task}_{mode}.json').write_text(json.dumps(c,indent=2)+'\n')
            cls=NativeTraced if mode=='native' else ScopedTraced
            a=cls(c);a.run(False)
            snap=snapshot(a)
            payload=dict(config=c,actions=a.trace,transitions=a.transitions,updates=a.update_trace,
                         episodes=a.episodes_trace,snapshot=snap)
            path=out/f'{task}_{mode}_trace.json'
            path.write_text(json.dumps(payload)+'\n')
            runs.append(a)
        native,collapsed,perstate=runs
        assert native.trace==collapsed.trace,(task,'actions')
        assert native.transitions==collapsed.transitions,(task,'transitions')
        assert native.update_trace==collapsed.update_trace,(task,'subgoal sets / rewards')
        assert native.episodes_trace==collapsed.episodes_trace,(task,'episodes')
        ns,cs=snapshot(native),snapshot(collapsed)
        assert ns['meta']==cs['meta'],(task,'meta values')
        assert ns['banks'][0]==cs['banks'][0][str(('*','*'))],(task,'skill values / counters')
        b=perstate._formula_banks[0]
        assert len(b.banks)>1,(task,'vacuous single-state smoke')
        assert b.template._num_update_calls==0
        assert sum(v._num_update_calls for v in b.banks.values())==len(perstate.transitions)
        item=dict(task=task,episodes=args.episodes,seed=args.seed,
                  collapsed_actions_equal=True,collapsed_updates_rewards_equal=True,
                  collapsed_skill_and_meta_tables_equal=True,
                  action_steps=len(native.trace),training_steps=len(native.transitions),
                  perstate_contexts=len(b.banks),perstate_training_steps=len(perstate.transitions))
        summary.append(item);print(json.dumps(item),flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
