import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {applyPresentationChartFont,finalizePresentation} from '/Users/yuhang/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations/container_tools/artifact_tool_utils.mjs';
const ROOT='/Users/yuhang/Downloads/why TL';
const WORK=ROOT+'/AUTO/research/2026-09-15-project-slides';
const SKILL='/Users/yuhang/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const FONT='Arial Unicode MS';
const C={ink:'#202B33',muted:'#687780',line:'#D9E0E4',blue:'#2176AD',light:'#EAF3F9',orange:'#C47A35',gray:'#94A1AA',red:'#AE514B',green:'#39806D',white:'#FFFFFF'};
const p=Presentation.create({slideSize:{width:1280,height:720}});
let n=0;
function text(s,t,x,y,w,h=45,size=26,color=C.ink,bold=false,align='left'){
 const z=s.shapes.add({geometry:'textbox',name:'text-'+(++n),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 z.text=t; z.text.style={typeface:FONT,fontSize:size,bold,color,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:0,right:0,top:0,bottom:0}};return z;
}
function rect(s,x,y,w,h,fill=C.white,stroke=C.line){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:1.5}});}
function node(s,t,x,y,w=96,h=68,color=C.blue,fill=C.white){let z=rect(s,x,y,w,h,fill,color);z.text=t;z.text.style={typeface:FONT,fontSize:28,color,alignment:'center',verticalAlignment:'middle',autoFit:'none',insets:{left:6,right:6,top:4,bottom:4}};return z;}
function conn(s,a,b,color=C.gray,from='right',to='left',dash=false){return s.shapes.connect(a,b,{kind:'straight',fromSide:from,toSide:to,line:{fill:color,width:2,style:dash?'dashed':'solid'},tail:{type:'triangle',width:'sm',length:'sm'}});}
function line(s,x,y,w,h=0,color=C.line){return s.shapes.add({geometry:'line',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:color,width:1.5}});}
function slide(title,note,foot=''){
 const s=p.slides.add();s.background.fill=C.white;text(s,title,64,38,1140,60,36,C.ink,true);
 if(foot)text(s,foot,64,661,1152,28,14,C.muted);
 text(s,String(p.slides.items.length).padStart(2,'0'),1190,661,35,28,16,C.muted,false,'right');
 s.speakerNotes.textFrame.setText(note);return s;
}
function chart(s,type,cats,series,x,y,w,h,opts={}){
 series=series.map(v=>({...v,valuesFormatCode:v.valuesFormatCode??(Math.max(...v.values)>1000?'#,##0':'0.00')}));
 const ch=s.charts.add(type,{position:{left:x,top:y,width:w,height:h},categories:cats,series,
  hasLegend:series.length>1,legend:{position:'top',overlay:false,textStyle:{typeface:FONT,fontSize:20,fill:C.ink}},
  chartFill:C.white,chartLine:{fill:'none',width:0},plotAreaFill:C.white,plotAreaLine:{fill:'none',width:0},
  barOptions:{direction:'column',grouping:'clustered',gapWidth:100},
  xAxis:{textStyle:{typeface:FONT,fontSize:20,fill:C.ink},line:{fill:C.line,width:1},majorGridlines:null},
  yAxis:{min:0,textStyle:{typeface:FONT,fontSize:18,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:'#E9EEF1',width:1}},
  dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:FONT,fontSize:22,fill:C.ink}},...opts});
 applyPresentationChartFont(ch,{fontFamily:FONT});return ch;
}
function notes(file,body){return body+'\n\nSource: '+ROOT+'/'+file;}
// ============ 主线 01：研究问题 ============
{
 const s=slide('Online Discovery in Unknown Tasks',notes('research/2026-09-15-memory-compression-audit/PROJECT_STATUS.md','The problem is discovering a hidden task order or logic, not goal discovery with no task signal at all. Events are identifiable and the early experiments provide reliable step-by-step advance feedback. The grid drawing is a mechanism sketch only and does not correspond to any specific experimental map.'),'The agent can identify events and is told whether each step advanced the task, but does not know the correct order');
 text(s,'Events visible, progress feedback reliable',80,125,485,40,24);
 const ox=90,oy=199,cs=58;
 for(let r=0;r<6;r++)for(let c=0;c<6;c++)rect(s,ox+c*cs,oy+r*cs,cs,cs,'#FFFFFF','#E2E7EA');
 for(const [t,c,r,col] of [['agent',0,5,C.ink],['A',1,1,C.blue],['B',4,0,C.blue],['C',4,4,C.blue],['D',2,3,C.blue]]) text(s,t,ox+c*cs,oy+r*cs,cs,cs,t==='agent'?15:30,col,true,'center');
 text(s,'A   ?   B   ?   C   ?   D',76,578,390,42,26,C.muted,false,'center');
 const a=node(s,'Interaction feedback',605,205,230,76),b=node(s,'Partial task structure',930,205,230,76),c=node(s,'Keep exploring',605,435,230,76),d=node(s,'Invoke learned skill',930,435,230,76);
 conn(s,a,b);conn(s,b,d,C.blue,'bottom','top');conn(s,d,c,C.blue,'left','right');conn(s,c,a,C.blue,'top','bottom');
 text(s,'Unknown part',632,520,190,36,21,C.muted,false,'center');text(s,'Known prefix',955,520,185,36,21,C.muted,false,'center');
}
// ============ 主线 02：第一个闭环结果 ============
{
 const s=slide('Discover and Use as You Go',notes('progressive_task_discovery/REPORT.md','Stage 6: length 16, action noise, a second batch of 32 fresh seeds. Left chart is the mean environment steps to first reach a 90% success rate. Right chart is the mean steps accumulated inside the already-confirmed goal prefix before the first full success; the two observation windows differ, so they must not be subtracted or read as an exact decomposition of total cost. All three methods do 9 Bellman updates per step; epsilon is 0.005 for immediate and delayed and 0.05 for plain replay (chosen on a dev set). No claim is made of beating ISA or other existing methods. Almost all of the saving comes from no longer re-walking the known prefix, not from exploration becoming smarter.'),'Length 16, 32 seeds; immediate vs delayed is a matched timing contrast; plain replay uses a different exploration rate and is a reference baseline only');
 text(s,'Steps to first 90%',80,137,620,38,27);text(s,'Before first full success: known prefix',816,137,400,38,25);
 chart(s,'bar',['Plain replay','Delayed use','Immediate use'],[{name:'Mean first90',values:[18594,37938,6906],fill:C.blue,points:[{idx:0,fill:C.gray},{idx:1,fill:C.orange},{idx:2,fill:C.blue}]}],65,198,705,380,{yAxis:{min:0,max:45000,majorUnit:10000,numberFormatCode:'#,##0',textStyle:{fontSize:18,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:27,fill:C.ink}}});
 chart(s,'bar',['Delayed','Immediate'],[{name:'Steps spent in known prefix',values:[33194,2566],fill:C.blue,points:[{idx:0,fill:C.orange}]}],812,198,404,380,{yAxis:{min:0,max:40000,majorUnit:10000,numberFormatCode:'#,##0',textStyle:{fontSize:16,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:25,fill:C.ink}}});
 text(s,'Almost all of the saving is the cost of re-walking the known prefix, not smarter exploration',64,600,1150,40,25,C.ink,true,'center');
}
// ============ 主线 03：线性够用，分支不够 ============
{
 const s=slide('Linear, Branching',notes('progressive_task_discovery/stage7_branching/REPORT.md','Left: in a linear sequence the progress count already identifies the current stage, so a stage-to-next-goal list suffices; the first positive result therefore cannot prove the value of general automaton learning. Right: A-then-E and B-then-E arrive at the same physical location with progress 2 and the same last event E, yet require different continuations; the two E boxes denote task states under different histories, not two physical locations. A prefix tree and a sufficient-history dictionary are identical action for action under matched inputs and random numbers, so the value does not come from the graph form. The coffee example is an isomorphic intuition aid, not a task used in any experiment: A and B are two different first steps (boil water / take cold water) and E is the one action both share (pour over grounds).'),'Hot brew / cold brew: the same pour over grounds, different history, different continuation');
 line(s,640,128,0,468,C.line);
 text(s,'Linear task: progress is enough',72,130,520,42,27,C.ink,true);
 text(s,'progress',110,196,180,38,23,C.muted);text(s,'goal',330,196,140,38,23,C.muted);line(s,104,238,380);
 [['0','A'],['1','B'],['2','C'],['3','?']].forEach(([a,b],i)=>{const y=256+i*58;text(s,a,110,y,180,44,27,C.ink);text(s,b,330,y,140,44,27,i===3?C.gray:C.blue);});
 text(s,'Branching task: progress is not enough',676,130,520,42,27,C.ink,true);
 const a=node(s,'A',690,206,78,56),ea=node(s,'E',900,206,78,56),d=node(s,'D',1110,206,78,56);
 const b=node(s,'B',690,376,78,56,C.orange),eb=node(s,'E',900,376,78,56,C.orange),c=node(s,'C',1110,376,78,56,C.orange);
 conn(s,a,ea,C.blue);conn(s,ea,d,C.blue);conn(s,b,eb,C.orange);conn(s,eb,c,C.orange);
}
// ============ 主线 04：State Abstraction ============
{
 const s=slide('State Abstraction',notes('AUTO/docs/MEMORY_COMPRESSION_PILOT.md','Parts-task rule: if A comes before B choose X, otherwise choose Y; the positions of the other parts are irrelevant. The full order is supplied by the environment and the agent makes a single binary choice at the end. The figure states equivalence of that terminal decision; it does not claim every intermediate prefix is equivalent in environment state. The correct automaton and an equivalent hand-written program use the same state partition. What has to be learned is not remembering the procedure but which histories may share future experience.'),'Different history does not mean different task state');
 text(s,'Full history',95,136,310,45,27);text(s,'Final task state',532,136,320,45,27);text(s,'Shared action values',954,136,280,45,27);
 const u=node(s,'A before B',540,264,230,80),v=node(s,'B before A',540,478,230,80,C.orange);
 ['ABC','ACB','CAB'].forEach((t,i)=>{const z=node(s,t,112,211+i*73,126,50);conn(s,z,u,C.blue);});
 ['BAC','BCA','CBA'].forEach((t,i)=>{const z=node(s,t,112,425+i*73,126,50,C.orange);conn(s,z,v,C.orange);});
 const x=node(s,'Q(state, X/Y)',970,264,240,80),y=node(s,'Q(state, X/Y)',970,478,240,80,C.orange);conn(s,u,x,C.blue);conn(s,v,y,C.orange);
}
// ============ 主线 05：Correct Compression的价值 ============
{
 const s=slide('Correct Compression',notes('AUTO/results/runs/memory_compression_pilot_5k.md','400 paired seeds per size, 5000 episodes per method. T95 is the episode at which accuracy first reaches and then holds at least 95% at later checkpoints; the main comparisons in this batch have no censoring, so RMST equals mean T95. The compression equivalence was supplied by the designer. Full history and correct compression use the same Q-learning. The automaton and the equivalent hand-written program agree seed by seed and checkpoint by checkpoint. Ratios come from unrounded means. The magnitude has a closed-form account: Q for the wrong action stays 0 and the right action becomes 1 after its first positive reward, after which that state is always answered correctly; uncovered states tie at random and are right half the time, so accuracy = (H+c)/2H and reaching 95% needs c >= 0.9H. With i covered, the chance of covering one more is (H-i)/(2H) and the expected wait is 2H/(H-i); summing gives sum(2H/(H-i), i=0..ceil(0.9H)-1). H = 6/24/120 gives 29.40/109.25/543.76 episodes and H = 2 after compression gives 6.0, a theoretical factor of 90.63x against an observed 89.75x. The large factor comes from how many table entries must be filled and needs no extra mechanism.'),'Correct equivalence given; 400 paired seeds per size (not 400 tasks: the candidate family holds only 6 / 12 / 20 rules)');
 chart(s,'bar',['3 parts','4 parts','5 parts'],[{name:'Full history',values:[28.28,109.15,547.23],fill:C.gray},{name:'Correct Compression',values:[6.05,5.89,6.10],fill:C.blue}],75,149,1120,415,{yAxis:{min:0,max:620,majorUnit:200,title:'Mean T95 / episodes',textStyle:{fontSize:18,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:23,fill:C.ink}}});
 [['4.67×',285],['18.55×',638],['89.75×',991]].forEach(([t,x])=>text(s,t,x-100,568,200,46,35,C.blue,true,'center'));
 text(s,'The number of full histories grows as n!; correct compression always has just two classes',80,616,1120,34,20,C.muted,false,'center');
}
// ============ 主线 06：在线判断能不能合并很难 ============
{
 const s=slide('Deciding Online Which Histories Can Merge',notes('progressive_task_discovery/stage8_compression/REPORT_STEP35.md','Step 35: 8 task families at n=3, 792 queries. The baseline trajectory is fixed and each query is replayed using only the evidence available at that moment; these merges were never wired into a controller and there is no corresponding learning curve. k_min is the smallest machine state count the evidence at that moment allows, and +1 relaxes that bound. Savings count only the redundant part (134/792); a further 102 reuses landed on necessary queries, which is replacement rather than saving. The acceptance criteria were written before the run: query saving >= 20% and wrong sharing near zero. All three failed. Assigning all 53 solver timeouts to Occam raises the saving ceiling to 23.6%, still short. Online k_min is 8-12 while the true state count is 19-27, so with little evidence the minimum machine over-merges heavily; that is the whole reason the offline 97.6% precision does not reproduce online. Verdict accuracy of 80.1% / 77.3% is in the appendix: it exceeds attribution accuracy because many wrong attributions happen to give the same verdict.'),'Replay audit on a fixed trajectory; these merges were never wired into the controller');
 const a=node(s,'History A',86,250,150,66,C.gray),b=node(s,'History B',86,430,150,66,C.gray),q=node(s,'Same state',330,340,160,72,C.red);conn(s,a,q,C.red);conn(s,b,q,C.red);
 text(s,'Unseen suffixes may differ, so merging misapplies the experience',86,560,500,40,22,C.red);
 line(s,620,128,0,450,C.line);
 text(s,'The current method shows a clear trade-off',668,136,540,45,28,C.ink);
 chart(s,'bar',['k_min  more aggressive','k_min+1  more conservative'],[
  {name:'State-attribution accuracy',values:[.407,.636],valuesFormatCode:'0.0%',fill:C.blue},
  {name:'Queries saved',values:[.169,.018],valuesFormatCode:'0.0%',fill:C.orange}],650,196,566,368,
  {yAxis:{min:0,max:.75,majorUnit:.25,numberFormatCode:'0%',textStyle:{fontSize:17,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},xAxis:{textStyle:{fontSize:19,fill:C.ink},line:{fill:C.line,width:1},majorGridlines:null},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:21,fill:C.ink}}});
}
// ============ 主线 07：提前压缩为什么收益小（Structure discovery线收尾）============
{
 const s=slide('Why Recent Early Compression Did Not Reproduce the Early Gain',notes('research/2026-09-15-memory-compression-audit/REVIEW.md','The latest parts pilot uses 400 seeds per size. During training and before the rule is uniquely identified, the immediate method achieves on average 0.1225, 0.4525 and 0.635 more successes than the delayed method. These are not the greedy-evaluation cumulative expectations of 0.137/0.533/0.696. Accuracy is below 95% before identification, and after identification the two arms are identical in this implementation, so T95 has no discriminating power; that necessity holds only for this rule family, deterministic feedback, and these update and threshold settings. In the active task, invoking skills early changes which suffixes are reached; in the passive task each episode presents an independently drawn permutation and the action does not affect the next piece of evidence. This does not establish that every early-use benefit must rest on active evidence acquisition, and it does not overturn the result on page 2.'),'The two kinds of immediate use differ: in the early experiment behaviour changes the next evidence, in the parts experiment it does not');
 text(s,'Early navigation task: behaviour changes the evidence',100,142,480,38,17,C.ink,true);text(s,'Latest parts experiment: evidence independent of behaviour',700,142,500,38,17,C.ink,true);
 let a=node(s,'Partial procedure',89,220,172,62),b=node(s,'Execute prefix',379,220,172,62),c=node(s,'Suffix evidence',238,355,172,62);conn(s,a,b,C.blue);conn(s,b,c,C.blue,'bottom','right');conn(s,c,a,C.blue,'left','bottom');
 let d=node(s,'Permutation',700,220,208,62,C.gray),e=node(s,'Choose X / Y',993,220,188,62,C.blue);conn(s,d,e);
 text(s,'The next episode is still sampled independently by the environment',660,330,560,40,16,C.muted,false,'center');
 text(s,'Extra successes before identification',700,420,470,44,25);
 chart(s,'bar',['3','4','5'],[{name:'Immediate minus delayed',values:[.123,.453,.635],valuesFormatCode:'0.000',fill:C.blue}],684,462,528,166,{yAxis:{min:0,max:.85,majorUnit:.4,numberFormatCode:'0.0',textStyle:{fontSize:15,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},xAxis:{textStyle:{fontSize:19,fill:C.ink},majorGridlines:null},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:22,fill:C.ink}}});
 text(s,'A real closed loop forms only when',80,516,560,38,22,C.ink);
 text(s,'behaviour changes the later evidence',80,552,560,38,22,C.ink);
}
// ============ 主线 08：换章节 —— 结构已有，怎么帮 RL ============
{
 const s=slide('After the Structure',notes('progressive_task_discovery/stage8_compression/REPORT_STEP36.md','Spoken: once you have the structure a second question remains, namely whether it only tells you the current goal or can also let different stages share control experience. This group all has the task rules given and does not involve learning while building. Left 2x2: Y00 has neither mechanism, Y01 uses a learned meta-controller over shared skills (reuse without routing), Y10 routes by the true rules but keeps skills separate per task state (routing without reuse), and Y11 has both. 24 families x 30 seeds x 5 arms, budget 200000 steps. The original AUC reading that routing overwhelmingly beats reuse was withdrawn in Step 38 because of ceiling effects; this page ranks no mechanism, and full AUC and first90 are in the appendix. Right is the Step 38.5 dose sweep: expensive navigation, task rules given, training budget 1000000 environment steps, attainment rate being the fraction of runs reaching 90% success within budget. The per-seed raw data of that sweep was later confirmed lost, so intervals cannot be recomputed. The Step 40 grid shows that step cost changes the QRM ordering and shrinks the risk interaction, so this must not be read as a protocol-independent effect size. The bottom line is taken from REPORT_STEP38_5 lines 125-126.'),'Task rules given; per-seed records of the hazard sweep are lost, so intervals cannot be recomputed');
 text(s,'Normal environment: all at the ceiling',72,130,530,42,26,C.ink,true);
 text(s,'No reuse',232,214,150,36,22,C.muted,false,'center');text(s,'Reuse',400,214,150,36,22,C.muted,false,'center');
 text(s,'No routing',72,290,140,36,22,C.muted);text(s,'Routing',72,420,140,36,22,C.muted);
 for(const [id,x,y,hl] of [['Y00',232,262,0],['Y01',400,262,0],['Y10',232,392,0],['Y11',400,392,1]]){
  rect(s,x,y,150,112,hl?C.light:C.white,C.line);text(s,id,x,y+34,150,44,29,hl?C.blue:C.ink,true,'center');}
 text(s,'The four arms differ little; this cannot rank the mechanisms',72,530,530,36,17,C.muted);
 line(s,610,128,0,450,C.line);
 text(s,'Hazardous / irreversible environment: the gap widens',652,130,560,42,26,C.ink,true);
 chart(s,'line',['0','1','2','3','4'],[
  {name:'Y10  per-stage skills',values:[1,.32,.16,.11,.10],valuesFormatCode:'0%',line:{fill:C.orange,width:3},marker:{symbol:'circle',size:7}},
  {name:'Y11  shared skills',values:[1,1,.99,.95,.91],valuesFormatCode:'0%',line:{fill:C.blue,width:3},marker:{symbol:'circle',size:7}}],640,186,580,340,
  {lineOptions:{smooth:false},xAxis:{title:'Hazard cells',textStyle:{fontSize:20,fill:C.ink},majorGridlines:null,line:{fill:C.line,width:1}},yAxis:{min:0,max:1.16,majorUnit:.5,numberFormatCode:'0%',textStyle:{fontSize:17,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:18,fill:C.ink}}});
 text(s,'Skills without routing 0.882   >>   routing without skills 0.106',640,538,580,36,23,C.ink,true,'center');
 text(s,'Under this protocol, reuse-related differences are clearly amplified',640,576,580,34,21,C.orange,false,'center');
}
// ============ 主线 09：Three-Layer Research Map ============
{
 const s=slide('Three-Layer Research Map',notes('research/2026-09-15-memory-compression-audit/PROJECT_STATUS.md','These three layers are not one integrated general algorithm. Structure discovery has a positive result for a restricted unknown-order loop; state abstraction has a clear gain from known correct compression while reliable online merging is unsolved; structure use shows conditional gains for routing and sharing, but general mechanism attribution, protocol and task scope remain open. No general method beating ISA and others has been established, and no claim is made that automaton syntax beats an equivalent hand-written implementation.'));
 text(s,'Verified',820,142,200,42,25,C.blue);text(s,'Open',1050,142,200,42,25,C.orange);
 const labels=['Structure discovery','State Abstraction','Structure use'];
 const ok=['Restricted online loop','Gain from correct compression','Conditional control gain'];const open=['General rule discovery','Reliable online merging','Mechanism attribution and scope'];
 for(let i=0;i<3;i++){
 let y=219+i*143; text(s,labels[i],77,y+18,200,50,31,C.ink,true);
 text(s,ok[i],807,y+24,229,43,24,C.blue);text(s,open[i],1045,y+24,231,43,24,C.orange);
 if(i===0){let a=node(s,'A',327,y+13,70,62),b=node(s,'B',475,y+13,70,62),c=node(s,'?',623,y+13,70,62,C.gray);conn(s,a,b);conn(s,b,c,C.gray,'right','left',true);}
 if(i===1){let a=node(s,'ABC',327,y-4,100,49,C.gray),b=node(s,'ACB',327,y+63,100,49,C.gray),c=node(s,'q',581,y+28,110,58,C.blue);conn(s,a,c);conn(s,b,c);}
 if(i===2){let a=node(s,'q',327,y+13,70,62),b=node(s,'goal',475,y+13,88,62),c=node(s,'skill',631,y+13,102,62);conn(s,a,b);conn(s,b,c);}
 if(i<2)line(s,76,y+123,1150);
 }
}
// ============ 10：Backup 分隔 ============
{
 const s=p.slides.add();s.background.fill=C.white;
 text(s,'Backup',64,300,600,70,44,C.ink,true);
 text(s,'10',1190,661,35,28,16,C.muted,false,'right');
 s.speakerNotes.textFrame.setText('The main line ends here. The six pages that follow cover the feedback budget, the equivalence of prefix tree and history dictionary, the closed-form account of the compression factor, the full definitions for the merge audit, the full five-arm numbers, and the complete hazard dose curve.');
}
// ============ B1：可靠任务反馈的预算 ============
{
 const s=slide('Appendix · Budget for Reliable Task Feedback',notes('AUTO/discovery/src/run_verify.py','Raw archive AUTO/discovery/results/verify.jsonl.gz: at budgets 40/80/160/320 the runs with first90 < 200000 are 1/61/131/144 out of 144. At budget 160 only 127/144 still hold at or above 90% at the end; this chart measures ever having attained, not final success rate. The controller reads local evidence by history node and merging is disabled. One must not infer from the final tree node count that a near-complete structure is required before it can be used: the nodes field records the size of the history prefix tree rather than the compressed state count, and it is written at the end of the run; the real compressed state count, classes, was never archived.'),'24 task families x 6 seeds; reached 90% at some point within 200,000 environment steps; merging disabled');
 chart(s,'bar',['40','80','160','320'],[{name:'Fraction of runs attaining',values:[0.006944444444,0.423611111111,0.909722222222,1],valuesFormatCode:'0%',fill:C.blue}],105,151,1080,400,{yAxis:{min:0,max:1.12,majorUnit:.25,numberFormatCode:'0%',textStyle:{fontSize:19,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},xAxis:{title:'Query budget',textStyle:{fontSize:25,fill:C.ink},line:{fill:C.line,width:1},majorGridlines:null},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:30,fill:C.ink}}});
 ['1 / 144','61 / 144','131 / 144','144 / 144'].forEach((t,i)=>text(s,t,209+i*245,560,190,40,23,C.muted,false,'center'));
 text(s,'Shows only that more feedback makes learning easier; it does not imply the structure must be near-complete to be usable',80,608,1120,36,22,C.orange,false,'center');
}
// ============ B2：前缀树与History字典 ============
{
 const s=slide('Appendix · Two Representations of the Same History',notes('progressive_task_discovery/stage7_branching/REPORT.md','A prefix tree and a dictionary over full successful event histories are identical action for action under matched inputs and random numbers. The dictionary holds sufficient history and does not store only progress or the last event. This compares representations and does not involve merging different histories. The value of the automaton therefore does not come from drawing a graph but from whether equivalent histories can later be merged.'),'Same information, two notations, identical action for action, so the value does not lie in the graph form');
 text(s,'Prefix tree',150,145,330,45,29);text(s,'History dictionary',780,145,400,45,29);
 const r=node(s,'∅',102,353,62,62,C.gray),a=node(s,'A',247,262,65,62),b=node(s,'B',247,457,65,62,C.orange),ea=node(s,'E',402,262,65,62),eb=node(s,'E',402,457,65,62,C.orange),d=node(s,'D',548,262,65,62),c=node(s,'C',548,457,65,62,C.orange);
 [[r,a],[a,ea],[ea,d]].forEach(([x,y])=>conn(s,x,y,C.blue));[[r,b],[b,eb],[eb,c]].forEach(([x,y])=>conn(s,x,y,C.orange));
 text(s,'≡',654,343,90,85,65,C.muted,false,'center');
 text(s,'History',786,247,170,42,24,C.muted);text(s,'Next goal',1040,247,190,42,24,C.muted);line(s,775,299,430);
 text(s,'(A, E)',800,327,180,55,32,C.blue);text(s,'D',1088,327,80,55,32,C.blue);
 text(s,'(B, E)',800,445,180,55,32,C.orange);text(s,'C',1088,445,80,55,32,C.orange);
}
// ============ B3：压缩倍数的闭式解释 ============
{
 const s=slide('Appendix · Closed-Form Account of the Compression Factor',notes('research/2026-09-15-memory-compression-audit/REVIEW.md','Within these clean states Q for the wrong action stays 0 and the right action becomes 1 once it receives a single positive reward, after which greedy evaluation answers that state correctly; states with no positive reward remain tied at random. If the number of full histories is H then accuracy = (H+c)/2H and reaching 95% needs c >= ceil(0.9H). Under iid uniform permutations and the present initialisation, with i covered the chance of covering one more is (H-i)/(2H) and the expected wait is 2H/(H-i); summing gives sum(2H/(H-i), i=0..ceil(0.9H)-1). H = 6/24/120 gives 29.40/109.25/543.76 episodes and H = 2 after compression gives 6.0. Theoretical factor 90.63x against an observed 89.75x. Formal T95 is recorded at checkpoints and may land slightly later than the exact per-episode moment. This shows the large factor is explained by a clear repeated table-filling mechanism and needs no extra assumption.'),'The large factor comes from how many table entries must be filled; no extra mechanism is needed');
 text(s,'Why this magnitude',72,132,540,44,27,C.ink,true);
 const L=[['Wrong action: Q stays 0',0],['Right action: Q = 1 after its first positive reward,',0],
          ['and that state is answered correctly ever after',1],
          ['Uncovered states tie at random, right half the time',0],
          ['-> accuracy = (H + c) / 2H; 95% needs c >= 0.9H',1],
          ['With i covered, P(cover one more) = (H - i) / 2H',0],
          ['-> expected wait 2H / (H - i)',1],
          ['T95 = Σ 2H/(H - i), i = 0 … ceil(0.9H) - 1',0],
          ['H = n! = 6 / 24 / 120; compressed H = 2 -> 6.0 episodes',1]];
 L.forEach((t,i)=>text(s,t[0],72,192+i*42,560,38,19,t[1]?C.blue:C.ink));
 chart(s,'bar',['3 parts','4 parts','5 parts'],[{name:'Closed form',values:[29.40,109.25,543.76],fill:C.gray,valuesFormatCode:'0.0'},{name:'Observed',values:[28.28,109.15,547.23],fill:C.blue,valuesFormatCode:'0.0'}],650,180,566,380,{barOptions:{direction:'column',grouping:'clustered',gapWidth:200},yAxis:{min:0,max:700,majorUnit:200,title:'Mean T95 / episodes',textStyle:{fontSize:16,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:11,fill:C.ink}}});
 text(s,'Theoretical factor 90.63x    observed 89.75x',650,578,566,40,25,C.blue,true,'center');
}
// ============ B4：合并审计完整口径 ============
{
 const s=slide('Appendix · Full Definitions for the Merge Audit',notes('progressive_task_discovery/stage8_compression/REPORT_STEP35.md','Step 35: 8 task families at n=3, 792 queries, of which 313 (39.5%) were redundant. Under k_min, 236 merges were forced: 47 wrong verdicts give a verdict accuracy of 80.1%, 140 wrong attributions give an attribution accuracy of 40.7%, and the saving is 134/792 = 16.9%. Under k_min+1 there were 22 reuses with 5 wrong verdicts and 8 wrong attributions, saving 14/792 = 1.8%. Verdict accuracy exceeds attribution accuracy because many wrong attributions happen to give the same verdict, and that luck does not survive the next event. A further 102 reuses landed on necessary queries, which is replacement rather than saving and is not counted in the 16.9%.'),'Verdict correct = the borrowed answer matches the true verdict; attribution correct = the merged partner really is the same task state');
 ['Verdict accuracy','State-attribution accuracy','Queries saved'].forEach((t,i)=>text(s,t,67+i*402,145,363,45,27,C.ink,false,'center'));
 [[.801,.773],[.407,.636],[.169,.018]].forEach((v,i)=>chart(s,'bar',['k_min','k_min+1'],[{name:'Fraction',values:v,valuesFormatCode:'0.0%',fill:C.blue,points:[{idx:1,fill:C.orange}]}],62+i*402,208,353,360,{yAxis:{min:0,max:1,majorUnit:.25,numberFormatCode:'0%',textStyle:{fontSize:17,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:24,fill:C.ink}}}));
 text(s,'Of 236 forced merges, 140 proved histories of different true states to be the same; online k_min is 8-12 while the true state count is 19-27',80,586,1120,36,22,C.ink,false,'center');
 text(s,'Pre-registered criteria 3 / 3 not met: saving 16.9% < 20%; 47 wrong verdicts; 140 wrong attributions; sensitivity ceiling 23.6% still short',80,624,1120,34,21,C.red,false,'center');
}
// ============ B5：五臂完整数值 ============
{
 const s=slide('Appendix · Full Five-Arm Numbers for Structure Use',notes('progressive_task_discovery/stage8_compression/REPORT_STEP36.md','24 families x 30 seeds x 5 arms, budget 200000 steps. first90 is the median and AUC is the area under the success-rate curve. The original AUC reading that routing overwhelmingly beats reuse was withdrawn in Step 38 because of ceiling effects. Censoring: in the raw archive results/factorial.json there are 40 runs with first90 = -1, all in Y00 (30/720) and Y01 (10/720), with none in Y10, Y11 or QRM. The medians on this page are over attainers only; if -1 is recorded as the budget of 200000 the Y00 median becomes 41000 rather than 38500, so dropping them flatters Y00 and Y01 and makes the comparison conservative. The report once gave sample-complexity main effects of 6.39x for routing and 5.34x for reuse; these could not be reproduced from the archive this time (recording -1 as the budget gives 9.03x and 4.50x), so that reading is not shown.'),'AUC sits near the ceiling and cannot rank the mechanisms');
 text(s,'No reuse',266,144,185,40,24,C.muted,false,'center');text(s,'Reuse',463,144,185,40,24,C.muted,false,'center');
 text(s,'No routing',61,260,140,40,24,C.muted);text(s,'Routing',61,463,140,40,24,C.muted);
 const cells=[['Y00','Flat history','.821',222,210],['Y01','Learned meta-controller','.883',438,210],['Y10','Per-stage skills','.988',222,411],['Y11','Shared skills','.997',438,411]];
 for(const [id,sub,auc,x,y] of cells){rect(s,x,y,196,171,id==='Y11'?C.light:C.white,C.line);text(s,id,x+18,y+13,160,46,29,id==='Y11'?C.blue:C.ink,true);text(s,sub,x+18,y+62,172,40,19,C.muted);text(s,'AUC '+auc,x+18,y+113,166,35,22,C.ink);}
 text(s,'Median first90 / environment steps',754,140,425,42,25);
 chart(s,'bar',['Y00','Y01','Y10','Y11','QRM'],[{name:'first90',values:[38500,4000,3750,1000,7000],fill:C.gray,points:[{idx:1,fill:C.orange},{idx:2,fill:'#72A0BA'},{idx:3,fill:C.blue}]}],694,220,525,350,{yAxis:{min:0,max:45000,majorUnit:15000,numberFormatCode:'#,##0',textStyle:{fontSize:16,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:20,fill:C.ink}}});
 text(s,'QRM  AUC .972',812,588,340,35,23,C.muted,false,'center');
 text(s,'Censoring: 30/720 runs of Y00 and 10/720 of Y01 never attained; medians are over attainers only',80,626,1120,32,19,C.orange,false,'center');
}
// ============ B6：危险格完整剂量曲线 ============
{
 const s=slide('Appendix · Hazard-Cell Dose Sweep',notes('progressive_task_discovery/stage8_compression/REPORT_STEP38_5.md','Step 38.5 dose sweep, expensive navigation, own derived simulator, task rules given, training budget 1000000 environment steps. Attainment rate is the fraction of runs reaching 90% success within budget. The per-seed raw data of this sweep was later confirmed lost, so confidence intervals cannot be recomputed. The Step 40 grid archives every run: step cost changes the QRM ordering and shrinks the risk interaction effect, so this page must not be taken as a protocol-independent effect size. Whatever graph-theoretic paths exist, hazard cells may force successful trajectories to detour, and no claim is made here that geometric difficulty is unchanged. The bottom line is taken from REPORT_STEP38_5 lines 125-126.'),'Rules given; fixed step-cost protocol; 1,000,000-step budget; per-seed records lost, intervals cannot be recomputed');
 chart(s,'line',['0','1','2','3','4'],[
 {name:'Y10  per-stage skills',values:[1,.32,.16,.11,.10],valuesFormatCode:'0%',line:{fill:C.orange,width:3},marker:{symbol:'circle',size:8}},
 {name:'Y11  shared skills',values:[1,1,.99,.95,.91],valuesFormatCode:'0%',line:{fill:C.blue,width:3},marker:{symbol:'circle',size:8}}],91,144,1105,424,{lineOptions:{smooth:false},xAxis:{title:'Hazard cells',textStyle:{fontSize:24,fill:C.ink},majorGridlines:null,line:{fill:C.line,width:1}},yAxis:{min:0,max:1.15,majorUnit:.25,numberFormatCode:'0%',textStyle:{fontSize:20,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:24,fill:C.ink}}});
 line(s,150,584,980);
 text(s,'Skills without routing 0.882   >>   routing without skills 0.106',80,596,1120,38,25,C.ink,true,'center');
 text(s,'Irreversibility amplifies reuse, not routing (with no hazard cells both arms sit at 100%)',80,631,1120,32,21,C.orange,false,'center');
}
await fs.mkdir(WORK+'/.build/previews_en',{recursive:true});
await (await PresentationFile.exportPptx(p)).save(WORK+'/.build/candidate_en.pptx');
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];const png=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(WORK+'/.build/previews_en/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
 console.log('rendered',i+1);
}
const FINAL=WORK+'/output/Task_Structure_Research_EN.pptx';
const result=await finalizePresentation({workspaceDir:WORK,candidatePath:WORK+'/.build/candidate_en.pptx',finalPath:FINAL,
 pythonExecutable:'/Users/yuhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:16,requiredNativeChartOwnerSlides:[2,5,6,7,8,11,13,14,15,16],requiredNativeTableOwnerSlides:[],
 materializeLiteralChartWorkbooks:true,nativeChartTargetApplication:'portable',fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
 receiptPath:WORK+'/.build/validation_en3.json'});
console.log('OK', result?.finalPath ?? FINAL);
