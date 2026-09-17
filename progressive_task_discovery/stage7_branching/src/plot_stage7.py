import csv
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from analyze_stage7 import read

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
COLORS={'history_replay':'#657888','hand_replay':'#be8c42','immediate_skill':'#008c82','delayed_skill':'#b36f85','oracle_skill':'#263842'}
LABELS={'history_replay':'Full history + replay','hand_replay':'Hand monitor + replay','immediate_skill':'Immediate skill reuse','delayed_skill':'Delayed skill reuse','oracle_skill':'Oracle + skills'}


def save(fig,name):
    (ROOT/'figures').mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def witness():
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),gridspec_kw={'width_ratios':[1,1.5]})
    grid=np.zeros((13,13));grid[1,4:9]=1;grid[1:13,6]=1;grid[6,:]=1
    axes[0].imshow(grid,cmap=ListedColormap(['#e2e8ed','#ffffff']),origin='upper',vmin=0,vmax=1)
    events={'A':(5,1),'B':(7,1),'C':(0,6),'D':(12,6),'E':(6,12)}
    palette={'A':'#407ab3','B':'#905ca5','C':'#148a74','D':'#d28c36','E':'#427d88'}
    for e,(x,y) in events.items():
        axes[0].scatter(x,y,s=220,c=palette[e],zorder=3,marker='s')
        axes[0].text(x,y,e,ha='center',va='center',color='white',fontweight='bold',zorder=4)
    for x in [4,8]:axes[0].scatter(x,1,s=60,c='#243541',marker='^',zorder=3)
    axes[0].scatter(6,6,s=230,marker='*',c='#cf9b13',zorder=3)
    axes[0].text(6,5.15,'J',ha='center',fontweight='bold')
    axes[0].annotate('',xy=(10.9,6),xytext=(6.5,6),arrowprops={'arrowstyle':'->','color':palette['A'],'lw':2})
    axes[0].annotate('',xy=(1.1,6),xytext=(5.5,6),arrowprops={'arrowstyle':'->','color':palette['B'],'lw':2})
    axes[0].set_title('Same physical junction, different required goals',fontsize=11,fontweight='bold')
    axes[0].set_xticks([]);axes[0].set_yticks([])
    axes[1].axis('off')
    axes[1].text(.03,.9,'A real task from seed 400',fontsize=14,fontweight='bold',transform=axes[1].transAxes)
    axes[1].text(.03,.74,'History A:    A → E    | next D',fontsize=16,color=palette['A'],transform=axes[1].transAxes)
    axes[1].text(.03,.60,'History B:    B → E    | next C',fontsize=16,color=palette['B'],transform=axes[1].transAxes)
    axes[1].text(.03,.38,'At junction J, both have:\nposition = (6, 6)    progress count = 2\nlast successful event = E',fontsize=12,linespacing=1.7,transform=axes[1].transAxes,bbox={'boxstyle':'round,pad=.65','fc':'#f0f4f7','ec':'none'})
    axes[1].text(.03,.13,'The earlier cue must survive the shared event.\nBoth starts are tested; every traversal costs steps.',fontsize=11,linespacing=1.6,transform=axes[1].transAxes)
    fig.tight_layout(w_pad=2);save(fig,'ambiguity_witness')


def intervals(values):
    values=np.asarray(values);rng=np.random.default_rng(1718)
    means=values[rng.integers(len(values),size=(5000,len(values)))].mean(axis=1)
    return values.mean(),*np.quantile(means,[.025,.975])


def scaling():
    scenarios=[('heldout','h16','One shared event | H = 16k'),('heldout','h24','One shared event | H = 24k'),('long_bridge','h20_bridge3','Three shared events | H = 20k')]
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.1),sharey=True);upper=0
    for ax,(stage,condition,title) in zip(axes,scenarios):
        data=[x for x in read(stage) if x['condition']==condition];ks=sorted({x['k'] for x in data})
        for method in COLORS:
            stats=np.array([intervals([(x['first90'] if x['first90']>=0 else x['budget'])/1000 for x in data if x['k']==k and x['method']==method]) for k in ks])
            upper=max(upper,float(stats[:,2].max()))
            ax.plot(ks,stats[:,0],marker='o',lw=1.8,markersize=4,color=COLORS[method],ls='--' if method=='oracle_skill' else '-',label=LABELS[method])
            ax.fill_between(ks,stats[:,1],stats[:,2],color=COLORS[method],alpha=.12,lw=0)
        ax.set_title(title,fontsize=11,fontweight='bold');ax.set_xticks(ks);ax.set_xlabel('Progress events per branch k');ax.grid(axis='y',alpha=.15)
    axes[0].set_ylim(0,upper*1.08);axes[0].set_ylabel('Training steps to 90% success (thousands)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.14),ncol=3,frameon=False)
    fig.text(.5,-.04,'32 new seeds per point. Count and last-event baselines: 0/32 solved at every point, censored at 100k (omitted from these curves).',ha='center',fontsize=9,color='#596774')
    fig.tight_layout();save(fig,'branching_results')


def causal():
    true=read('heldout');control=read('same_map_control')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    methods=['count_replay','last_event_replay','history_replay','hand_replay','immediate_skill']
    labels=['Count','Last event','Full history','Hand monitor','Immediate skills']
    for data,offset,color,label in [(true,-.18,'#b57488','Different next goals'),(control,.18,'#479b91','Same next goals')]:
        means=[np.mean([r['final_success'] for r in data if r['condition']=='h24' and r['k']==12 and r['method']==m]) for m in methods]
        axes[0].bar(np.arange(len(methods))+offset,means,width=.34,label=label,color=color)
    axes[0].set_xticks(range(len(methods)),labels,rotation=20,ha='right');axes[0].set_ylim(0,1.15);axes[0].set_ylabel('Final balanced success probability');axes[0].set_title('Same map: remove only task-state ambiguity',fontsize=11,fontweight='bold');axes[0].legend(frameon=False,fontsize=9)
    matched=read('matched_epsilon')
    for i,method in enumerate(['delayed_skill','immediate_skill']):
        rows=[r for r in matched if r['condition']=='h24' and r['k']==12 and r['method']==method]
        prefix=[];frontier=[]
        for r in rows:
            j=int(np.argmax(r['first_branch_success']));assert min(r['first_branch_success'])>=0
            prefix.append(r['prefix_at_branch_success'][j]/1000);frontier.append(r['frontier_at_branch_success'][j]/1000)
        p,f=np.mean(prefix),np.mean(frontier)
        axes[1].barh(i,p,color='#5a7d98',label='Known prefix' if i==0 else None)
        axes[1].barh(i,f,left=p,color='#83c1b6',label='Unknown frontier' if i==0 else None)
        axes[1].text(p+f+.3,i,f'{p+f:.1f}k',va='center',fontsize=10)
    axes[1].set_yticks([0,1],['Delayed reuse','Immediate reuse']);axes[1].invert_yaxis();axes[1].set_xlim(0,33)
    axes[1].set_xlabel('Steps until both branches have succeeded (thousands)');axes[1].set_title('Timing control: same epsilon, same 6 updates',fontsize=11,fontweight='bold');axes[1].legend(frameon=False,fontsize=9,loc='lower right')
    fig.text(.5,-.02,'k = 12, H = 24k, 32 paired seeds. Skill banks share code and update rules; online trajectories are allowed to differ.',ha='center',fontsize=9,color='#596774')
    fig.tight_layout(w_pad=2);save(fig,'causal_controls')


if __name__=='__main__':witness();scaling();causal()
