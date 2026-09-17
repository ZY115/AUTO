import collections
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pilot_statistics',ROOT.parent/'src/analyze.py')
statistics=importlib.util.module_from_spec(spec);spec.loader.exec_module(statistics)


def read(name):
    return [json.loads(x) for x in (ROOT/'results'/name/'raw.jsonl').read_text().splitlines()]


def summarize_stage(name):
    rows=read(name);out=ROOT/'results'/name
    summary=statistics.summarize(rows)
    groups=collections.defaultdict(list)
    for r in rows:groups[r['condition'],r['variant'],r['k']].append(r)
    for s in summary:
        group=groups[s['condition'],s['variant'],s['k']]
        s['state_accuracy']=np.mean([x['state_accuracy'] for x in group])
        s['junction_state_accuracy']=np.mean([x['junction_state_accuracy'] for x in group])
        s['feature_states']=np.mean([x['feature_states'] for x in group])
        s['true_states']=np.mean([x['true_states'] for x in group])
        has_skills=any(x['binding_steps']>0 for x in group)
        for key in ['next_goal_accuracy','next_goal_coverage']:
            s[key]=np.mean([x[key] for x in group]) if has_skills else ''
        s['wrong_binding_rate']=sum(x['wrong_binding_steps'] for x in group)/sum(x['binding_steps'] for x in group) if has_skills else ''
        s['skill_target_success_rate']=sum(x['skill_successes'] for x in group)/sum(x['skill_attempts'] for x in group) if has_skills else ''
        s['skill_other_progress_rate']=sum(x['skill_other_progress'] for x in group)/sum(x['skill_attempts'] for x in group) if has_skills else ''
        costs=[];prefix=[];frontier=[]
        for r in group:
            if min(r['first_branch_success'])>=0:
                i=int(np.argmax(r['first_branch_success']))
                costs.append(r['first_branch_success'][i]);prefix.append(r['prefix_at_branch_success'][i]);frontier.append(r['frontier_at_branch_success'][i])
        s['both_branches_discovered']=len(costs)
        s['steps_until_both_success']=np.mean(costs) if costs else ''
        s['prefix_until_both_success']=np.mean(prefix) if prefix else ''
        s['frontier_until_both_success']=np.mean(frontier) if frontier else ''
        s['best_single_branch_success']=np.mean([max(x['final_branch_success']) for x in group])
        for key in ['stochastic_final_success_e005','stochastic_final_success_e02']:
            s[key]=np.mean([x[key] for x in group]) if key in group[0] else ''
    statistics.write_csv(out/'summary.csv',summary)
    for base in ['count_replay','history_replay','hand_replay','delayed_skill','immediate_skill','delayed_e005']:
        pairs=statistics.paired(rows,base)
        if pairs:statistics.write_csv(out/f'paired_vs_{base}.csv',pairs)
    return summary


def main():
    names=['development','bridge_development','heldout','long_bridge','matched_epsilon','same_map_control']
    for name in names:
        data=summarize_stage(name)
        print('\n'+name)
        for row in data:
            if name not in ['development','bridge_development'] and row['k']==max(x['k'] for x in data):
                print(row['condition'],row['variant'],round(row['mean_capped_first90']),f"{row['solved']}/{row['n']}",round(row['auc'],4),round(row['junction_state_accuracy'],3))


if __name__=='__main__':main()
