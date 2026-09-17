import collections
import csv
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import read, capped

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold','savefig.dpi':180})
COLORS={'count':'#687888','progressive':'#aa7889','learned_qrm':'#8971c4','count_replay':'#d89435',
        'goal_reuse':'#008d85','oracle_goal':'#26343e','delayed_goal':'#ac6778','known_goal':'#5896b6'}
LABELS={'count':'Count Q','progressive':'Frontier schedule','learned_qrm':'Known-edge reuse',
        'count_replay':'Replay (9 updates)','goal_reuse':'Immediate skill reuse','oracle_goal':'Oracle skills',
        'delayed_goal':'Use skills only after full discovery','known_goal':'Train only discovered goals'}
CONDITIONS={'benign':'Deterministic | H = 24k','deadline':'Tighter deadline | H = 16k','noisy':'20% action replacement | H = 20k'}


def interval(values):
    values=np.asarray(values,dtype=float)
    ix=np.random.default_rng(667).integers(len(values),size=(5000,len(values)))
    bs=values[ix].mean(axis=1)
    return values.mean(),*np.quantile(bs,[.025,.975])


def save(fig,name):
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/(name+'.png'),bbox_inches='tight',facecolor='white')
    fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def scaling():
    rows=read('stage5_heldout')
    methods=['count','progressive','learned_qrm','count_replay','goal_reuse','oracle_goal']
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),sharey=True)
    largest_upper=0
    for ax,(condition,title) in zip(axes,CONDITIONS.items()):
        for method in methods:
            lengths=sorted({x['k'] for x in rows})
            stats=np.array([interval([capped(x)/1000 for x in rows if x['condition']==condition and x['variant']==method and x['k']==k]) for k in lengths])
            largest_upper=max(largest_upper,float(stats[:,2].max()))
            style='--' if method in ['progressive','oracle_goal'] else '-'
            ax.plot(lengths,stats[:,0],style,marker='o',markersize=3.6,lw=1.8,color=COLORS[method],label=LABELS[method])
            ax.fill_between(lengths,stats[:,1],stats[:,2],alpha=.1,color=COLORS[method],linewidth=0)
        ax.set_title(title,fontsize=11);ax.set_xlabel('Sequence length k');ax.set_xticks(lengths);ax.grid(axis='y',alpha=.17)
    axes[0].set_ylim(0,largest_upper*1.06)
    axes[0].set_ylabel('Training steps to 90% success (thousands)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.13),ncol=3,frameon=False)
    fig.text(.5,-.02,'32 held-out seeds per point; shaded paired-seed bootstrap 95% intervals. First-hit costs capped at 100k. All prefix execution is counted.',ha='center',fontsize=9,color='#56606b')
    fig.tight_layout();save(fig,'heldout_scaling')


def ablation():
    rows=read('stage6_early_use');methods=['delayed_goal','count_replay','known_goal','goal_reuse']
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),gridspec_kw={'width_ratios':[1.35,1]})
    conditions=list(CONDITIONS)
    width=.19;xs=np.arange(3)
    for j,method in enumerate(methods):
        stats=np.array([interval([capped(x)/1000 for x in rows if x['k']==16 and x['condition']==cond and x['variant']==method]) for cond in conditions])
        axes[0].bar(xs+(j-1.5)*width,stats[:,0],width,color=COLORS[method],label=LABELS[method])
        axes[0].errorbar(xs+(j-1.5)*width,stats[:,0],yerr=np.array([stats[:,0]-stats[:,1],stats[:,2]-stats[:,0]]),fmt='none',ecolor='#334155',capsize=2,lw=.8)
    axes[0].set_xticks(xs,['Deterministic','Deadline','Noisy']);axes[0].set_ylabel('Training steps to 90% (thousands)');axes[0].set_title('Early use is the consequential decision | k = 16',fontsize=11)
    short={'delayed_goal':'Delayed use','count_replay':'Replay','known_goal':'Known goals only','goal_reuse':'Immediate use'}
    for j,method in enumerate(methods):
        subset=[x for x in rows if x['k']==16 and x['condition']=='noisy' and x['variant']==method]
        prefix=np.mean([x['prefix_steps_at_first_success'] for x in subset])/1000
        frontier=np.mean([x['frontier_steps_at_first_success'] for x in subset])/1000
        axes[1].barh(j,prefix,color='#547d98',label='Known-prefix steps' if j==0 else None)
        axes[1].barh(j,frontier,left=prefix,color='#7fc4bd',label='Unknown-frontier steps' if j==0 else None)
        axes[1].text(prefix+frontier+.4,j,f'{prefix+frontier:.1f}k',va='center',fontsize=9)
    axes[1].set_yticks(range(4),[short[x] for x in methods]);axes[1].invert_yaxis();axes[1].set_xlim(0,43)
    axes[1].set_xlabel('Steps before first complete training success (thousands)');axes[1].set_title('Where the saved interactions come from | noisy',fontsize=11)
    axes[1].legend(loc='lower right',fontsize=8,frameon=False)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.1),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,-.025,'32 new seeds per condition. Immediate and delayed use both train the same 8 auxiliary skill tables: 9 updates per interaction.',ha='center',fontsize=9,color='#56606b')
    for ax in axes:ax.grid(axis='x' if ax==axes[1] else 'y',alpha=.13)
    fig.tight_layout(w_pad=3);save(fig,'early_use_ablation')


def learning():
    rows=read('stage5_heldout');methods=['count','progressive','count_replay','goal_reuse','oracle_goal']
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),sharey=True)
    for ax,k in zip(axes,[8,16]):
        for method in methods:
            subset=[x for x in rows if x['condition']=='noisy' and x['k']==k and x['variant']==method]
            y=np.array([[z[1] for z in x['checkpoints']] for x in subset]);steps=np.array([z[0] for z in subset[0]['checkpoints']])/1000
            ix=np.random.default_rng(291).integers(len(y),size=(1000,len(y)))
            boot=y[ix].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975],axis=0)
            ax.plot(steps,y.mean(axis=0),label=LABELS[method],color=COLORS[method],lw=1.5,ls='--' if method in ['progressive','oracle_goal'] else '-')
            ax.fill_between(steps,lo,hi,color=COLORS[method],alpha=.1,lw=0)
        ax.axhline(.9,color='#9ca3af',lw=.7,ls=':');ax.set_title(f'Noisy navigation | k = {k}');ax.set_xlabel('Training interactions (thousands)');ax.set_ylim(0,1.03);ax.grid(alpha=.12)
    axes[0].set_ylabel('Exact greedy-policy success probability')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.1),ncol=3,frameon=False)
    fig.tight_layout();save(fig,'noisy_learning_curves')


if __name__=='__main__':
    scaling();ablation();learning()
