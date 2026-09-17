import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {applyPresentationChartFont,finalizePresentation} from '/Users/yuhang/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations/container_tools/artifact_tool_utils.mjs';
const ROOT='/Users/yuhang/Downloads/why TL';
const WORK=ROOT+'/research/2026-09-15-project-slides';
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
 if(foot)text(s,foot,64,661,1110,28,17,C.muted);
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
function notes(file,body){return body+'\n\n资料来源：'+ROOT+'/'+file;}
// ============ 主线 01：研究问题 ============
{
 const s=slide('未知任务中的在线发现',notes('research/2026-09-15-memory-compression-audit/PROJECT_STATUS.md','问题是隐藏任务顺序/逻辑的发现，不是完全没有任务信号的目标发现。事件可识别，早期实验提供可靠的逐步推进反馈。格子图仅为机制示意，不对应具体实验地图。'),'智能体能识别事件，也能得到「这一步推进了吗」的反馈；但不知道正确顺序');
 text(s,'事件可见，进度反馈可靠',80,125,485,40,24);
 const ox=90,oy=199,cs=58;
 for(let r=0;r<6;r++)for(let c=0;c<6;c++)rect(s,ox+c*cs,oy+r*cs,cs,cs,'#FFFFFF','#E2E7EA');
 for(const [t,c,r,col] of [['agent',0,5,C.ink],['A',1,1,C.blue],['B',4,0,C.blue],['C',4,4,C.blue],['D',2,3,C.blue]]) text(s,t,ox+c*cs,oy+r*cs,cs,cs,t==='agent'?15:30,col,true,'center');
 text(s,'A   ?   B   ?   C   ?   D',76,578,390,42,26,C.muted,false,'center');
 const a=node(s,'交互反馈',605,205,230,76),b=node(s,'局部任务结构',930,205,230,76),c=node(s,'继续探索',605,435,230,76),d=node(s,'调用已学技能',930,435,230,76);
 conn(s,a,b);conn(s,b,d,C.blue,'bottom','top');conn(s,d,c,C.blue,'left','right');conn(s,c,a,C.blue,'top','bottom');
 text(s,'未知部分',632,520,190,36,21,C.muted,false,'center');text(s,'已知前缀',955,520,185,36,21,C.muted,false,'center');
}
// ============ 主线 02：第一个闭环结果 ============
{
 const s=slide('边发现边使用：第一个闭环结果',notes('progressive_task_discovery/REPORT.md','Stage 6：长度16，动作噪声，第二批32个新种子。左图为首次成功率达到90%的平均环境步。右图为第一次完整成功之前，在已知目标前缀中累计执行的平均步数，两者观察区间不同，不应当相减或当成总成本的精确分解。三方法每步9次Bellman更新；立即/延迟epsilon均为0.005，普通replay为0.05（开发集选择）。未声称超过ISA等既有方法。节省几乎全部来自不再反复执行已知前缀，而不是探索本身变聪明。'),'长度 16，32 个种子；立即与延迟是同配置的时机对照，普通 replay 探索率不同，仅作参考基线');
 text(s,'首次达到 90%',80,137,620,38,27);text(s,'首次完整成功前：已知前缀',816,137,400,38,25);
 chart(s,'bar',['普通 replay','延迟使用','立即使用'],[{name:'平均 first90',values:[18594,37938,6906],fill:C.blue,points:[{idx:0,fill:C.gray},{idx:1,fill:C.orange},{idx:2,fill:C.blue}]}],65,198,705,380,{yAxis:{min:0,max:45000,majorUnit:10000,numberFormatCode:'#,##0',textStyle:{fontSize:18,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:27,fill:C.ink}}});
 chart(s,'bar',['延迟','立即'],[{name:'已知前缀执行步',values:[33194,2566],fill:C.blue,points:[{idx:0,fill:C.orange}]}],812,198,404,380,{yAxis:{min:0,max:40000,majorUnit:10000,numberFormatCode:'#,##0',textStyle:{fontSize:16,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:25,fill:C.ink}}});
 text(s,'省下来的几乎全是「重走已知前缀」的路费，不是探索本身变聪明',64,600,1150,40,25,C.ink,true,'center');
}
// ============ 主线 03：线性够用，分支不够 ============
{
 const s=slide('线性任务够用，分支任务不够',notes('progressive_task_discovery/stage7_branching/REPORT.md','左：线性序列中进度次数已能表示当前阶段，一个「阶段到下一目标」的列表就足够，所以第一个正结果不能证明一般自动机学习的价值。右：A→E与B→E汇入同一物理位置，进度均为2、末事件均为E，但后继要求不同；两个E表示不同历史下的任务状态，不是两个物理位置。底行：前缀树与充分历史dictionary在匹配输入与随机数时逐动作一致，所以价值不来自图这种形式。咖啡只是同构直觉说明，不是实验中的任务。'),'热泡 / 冷萃：同一个「冲粉」，前史不同 → 后续不同');
 line(s,640,128,0,468,C.line);
 text(s,'线性任务：进度就够',72,130,520,42,27,C.ink,true);
 text(s,'progress',110,196,180,38,23,C.muted);text(s,'goal',330,196,140,38,23,C.muted);line(s,104,238,380);
 [['0','A'],['1','B'],['2','C'],['3','?']].forEach(([a,b],i)=>{const y=256+i*58;text(s,a,110,y,180,44,27,C.ink);text(s,b,330,y,140,44,27,i===3?C.gray:C.blue);});
 text(s,'进度唯一标记阶段 —— 不需要真正的结构学习',72,510,540,40,23,C.muted);
 text(s,'分支任务：进度不够',676,130,520,42,27,C.ink,true);
 const a=node(s,'A',690,206,78,56),ea=node(s,'E',900,206,78,56),d=node(s,'D',1110,206,78,56);
 const b=node(s,'B',690,376,78,56,C.orange),eb=node(s,'E',900,376,78,56,C.orange),c=node(s,'C',1110,376,78,56,C.orange);
 conn(s,a,ea,C.blue);conn(s,ea,d,C.blue);conn(s,b,eb,C.orange);conn(s,eb,c,C.orange);
 text(s,'进度都 = 2，位置都是 E，下一步却不同',676,296,520,40,24,C.ink,false,'center');
 text(s,'Prefix tree  ≡  充分历史字典 —— 价值不在画成一张图',676,510,520,40,23,C.muted,false,'center');
}
// ============ 主线 04：状态抽象 ============
{
 const s=slide('于是问题变成状态抽象',notes('AUTO/docs/MEMORY_COMPRESSION_PILOT.md','零件任务规则：A先于B则选X，否则选Y，其他零件位置无关。完整顺序由环境提供；末端只做一次二选一。此图表示末端决策等价，不声称每个中间前缀在环境状态上都等价。正确自动机与等价手写程序使用同一状态划分。真正要学的不是「记住流程」，而是「哪些历史可以共享未来经验」。'),'历史不同 ≠ 任务状态不同');
 text(s,'完整历史',95,136,310,45,27);text(s,'最终任务状态',532,136,320,45,27);text(s,'共享动作价值',954,136,280,45,27);
 const u=node(s,'A 先于 B',540,264,230,80),v=node(s,'B 先于 A',540,478,230,80,C.orange);
 ['ABC','ACB','CAB'].forEach((t,i)=>{const z=node(s,t,112,211+i*73,126,50);conn(s,z,u,C.blue);});
 ['BAC','BCA','CBA'].forEach((t,i)=>{const z=node(s,t,112,425+i*73,126,50,C.orange);conn(s,z,v,C.orange);});
 const x=node(s,'Q(状态, X/Y)',970,264,240,80),y=node(s,'Q(状态, X/Y)',970,478,240,80,C.orange);conn(s,u,x,C.blue);conn(s,v,y,C.orange);
}
// ============ 主线 05：正确压缩的价值 ============
{
 const s=slide('正确压缩确实值钱',notes('AUTO/results/runs/memory_compression_pilot_5k.md','每个规模400个配对种子，每方法5000幕。T95为首次达到并在后续记录点保持至少95%准确率的幕数，本批主要对照无删失，因此RMST等于平均T95。压缩等价关系由设计者提供。完整历史与正确压缩使用同样Q-learning。自动机与等价手写程序逐种子逐评估点一致。比值来自未取整均值。该量级有闭式解释：错误动作Q恒为0，正确动作首次获正奖励后Q=1，此后该状态永远答对；未覆盖状态随机并列答对一半，故准确率=(H+c)/2H，要达95%需覆盖c>=0.9H。已覆盖i格时新覆盖一格的概率为(H-i)/(2H)，期望等待2H/(H-i)，累加得 sum(2H/(H-i), i=0..ceil(0.9H)-1)。H=6/24/120给出29.40/109.25/543.76幕，压缩后H=2给出6.0幕，理论倍数90.63x，实测89.75x。大倍数来自填表次数，不需要额外机制假设。'),'正确等价关系已知；每规模 400 个配对种子（不是 400 个任务：候选族只有 6 / 12 / 20 条规则）');
 chart(s,'bar',['3 个零件','4 个零件','5 个零件'],[{name:'完整历史',values:[28.28,109.15,547.23],fill:C.gray},{name:'正确压缩',values:[6.05,5.89,6.10],fill:C.blue}],75,149,1120,415,{yAxis:{min:0,max:620,majorUnit:200,title:'平均 T95 / 幕',textStyle:{fontSize:18,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:23,fill:C.ink}}});
 [['4.67×',285],['18.55×',638],['89.75×',991]].forEach(([t,x])=>text(s,t,x-100,568,200,46,35,C.blue,true,'center'));
 text(s,'完整历史的条数随 n! 增长；正确压缩始终只有两类',80,616,1120,34,20,C.muted,false,'center');
}
// ============ 主线 06：在线判断能不能合并很难 ============
{
 const s=slide('真正的难点：在线判断哪些历史能合并',notes('progressive_task_discovery/stage8_compression/REPORT_STEP35.md','Step35：n=3的8个任务族，792次查询。固定基线轨迹，逐查询只使用当时证据进行回放审计；未将这些合并接入控制器，也没有相应学习曲线。k_min是当时证据所容许的最小机器状态数，+1放松该上界。省下只计冗余部分（134/792），另有102次复用落在必要查询上，那不是节省而是替换。预注册判据在跑之前写定：查询节省>=20%且错误共享约等于0，三项全不达标；把53次求解超时全判给Occam，节省上界23.6%仍不达标。在线k_min是8-12而真状态数是19-27，证据少时最小机器严重过度合并，这是离线97.6%精确率无法在线复现的全部原因。裁决正确率80.1%/77.3%见附录：它高于归属正确率，是因为大量错误归属碰巧给出相同裁决。'),'固定轨迹回放审计，未把这些合并接进控制器');
 text(s,'过早合并的代价',86,136,420,45,28,C.red);
 const a=node(s,'历史 A',86,250,150,66,C.gray),b=node(s,'历史 B',86,430,150,66,C.gray),q=node(s,'同一状态',330,340,160,72,C.red);conn(s,a,q,C.red);conn(s,b,q,C.red);
 text(s,'过滤？',408,246,170,45,26,C.red);text(s,'冷藏？',408,470,170,45,26,C.red);
 text(s,'未见后缀可能不同 —— 合并就会错用经验',86,560,500,40,22,C.red);
 line(s,620,128,0,450,C.line);
 text(s,'当前方法出现明显 trade-off',668,136,540,45,28,C.ink);
 chart(s,'bar',['k_min　判断更激进','k_min+1　判断更保守'],[
  {name:'状态归属正确率',values:[.407,.636],valuesFormatCode:'0.0%',fill:C.blue},
  {name:'可省查询比例',values:[.169,.018],valuesFormatCode:'0.0%',fill:C.orange}],650,196,566,368,
  {yAxis:{min:0,max:.75,majorUnit:.25,numberFormatCode:'0%',textStyle:{fontSize:17,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},xAxis:{textStyle:{fontSize:19,fill:C.ink},line:{fill:C.line,width:1},majorGridlines:null},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:21,fill:C.ink}}});
 text(s,'预注册判据 3 / 3 不达标',650,578,566,40,24,C.red,true,'center');
}
// ============ 主线 07：提前压缩为什么收益小（结构发现线收尾）============
{
 const s=slide('为什么最近的「提前压缩」没复现早期大收益',notes('research/2026-09-15-memory-compression-audit/REVIEW.md','最新零件pilot每规模400种子。训练期间、规则唯一识别前立即法相比延迟法实际额外成功均值为0.1225、0.4525、0.635。不是贪心评估累计期望0.137/0.533/0.696。规则识别前准确率不足95%；识别后两臂在该实现中相同，因此T95没有分辨能力，该必然性只针对这个规则族、确定反馈、更新与阈值设置。主动任务早期调用技能改变到达后缀的行为，被动任务每幕独立给出排列，动作不影响下一份证据。不能据此断言所有边学边用收益都必须依赖主动取证，也不能用它否定第2页的结果。'),'两种「立即使用」不是一回事：早期实验里行为会改变下一份证据，零件实验里不会');
 text(s,'早期导航任务：行为改变证据',100,140,470,40,25,C.ink,true);text(s,'最新零件实验：证据与行为无关',700,140,470,40,25,C.ink,true);
 let a=node(s,'局部流程',89,220,172,62),b=node(s,'执行前缀',379,220,172,62),c=node(s,'后缀证据',238,355,172,62);conn(s,a,b,C.blue);conn(s,b,c,C.blue,'bottom','right');conn(s,c,a,C.blue,'left','bottom');
 let d=node(s,'环境给出排列',700,220,208,62,C.gray),e=node(s,'选择 X / Y',993,220,188,62,C.blue);conn(s,d,e);
 text(s,'下一幕仍由环境独立抽样',738,325,430,55,23,C.muted,false,'center');
 text(s,'识别前实际额外成功 / 次',700,420,470,44,25);
 chart(s,'bar',['3 个','4 个','5 个'],[{name:'立即减延迟',values:[.123,.453,.635],valuesFormatCode:'0.000',fill:C.blue}],684,462,528,166,{yAxis:{min:0,max:.85,majorUnit:.4,numberFormatCode:'0.0',textStyle:{fontSize:15,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},xAxis:{textStyle:{fontSize:19,fill:C.ink},majorGridlines:null},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:22,fill:C.ink}}});
 text(s,'当行为会改变后续证据时，提前使用才能形成真正的闭环',80,528,560,120,24,C.ink);
}
// ============ 主线 08：换章节 —— 结构已有，怎么帮 RL ============
{
 const s=slide('结构拿到以后，还有第二个独立问题',notes('progressive_task_discovery/stage8_compression/REPORT_STEP36.md','口播：有了结构以后还有第二个问题——它只是告诉我当前目标，还是还能让不同阶段共享控制经验？这一组全部是「任务规则已给定」，不涉及边学边建。左侧2x2：Y00两种机制都没有，Y01用学习型元控制器调用共享技能（有复用无路由），Y10真规则路由但技能按任务状态独立（有路由无复用），Y11两者都有。24族×30种子×5臂，预算200000步。原报告关于路由压倒性强于复用的AUC解释已在Step38撤回，因天花板影响；本页不作机制排序，完整AUC与first90见附录。右侧为Step38.5剂量扫描：昂贵导航，任务规则已给定，训练预算1000000环境步，达标率为预算内达到90%成功率的运行比例；该扫描逐种子原始数据后来确认遗失，不能重算区间。Step40新网格显示步费会改变QRM排序并使风险交互缩小，不能视为协议不变的效应大小。底行取自REPORT_STEP38_5第125-126行。'),'任务规则已给定；危险格扫描的逐种子记录已缺失，不能重算区间');
 text(s,'普通环境：都贴着天花板',72,130,530,42,26,C.ink,true);
 text(s,'无复用',232,214,150,36,22,C.muted,false,'center');text(s,'有复用',400,214,150,36,22,C.muted,false,'center');
 text(s,'无路由',72,290,140,36,22,C.muted);text(s,'有路由',72,420,140,36,22,C.muted);
 for(const [id,x,y,hl] of [['Y00',232,262,0],['Y01',400,262,0],['Y10',232,392,0],['Y11',400,392,1]]){
  rect(s,x,y,150,112,hl?C.light:C.white,C.line);text(s,id,x,y+34,150,44,29,hl?C.blue:C.ink,true,'center');}
 text(s,'四臂差距很小，不能据此排机制强弱',72,528,530,38,22,C.muted);
 line(s,610,128,0,450,C.line);
 text(s,'危险 / 不可逆环境：差异被放大',652,130,560,42,26,C.ink,true);
 chart(s,'line',['0','1','2','3','4'],[
  {name:'Y10  分阶段技能',values:[1,.32,.16,.11,.10],valuesFormatCode:'0%',line:{fill:C.orange,width:3},marker:{symbol:'circle',size:7}},
  {name:'Y11  共享技能',values:[1,1,.99,.95,.91],valuesFormatCode:'0%',line:{fill:C.blue,width:3},marker:{symbol:'circle',size:7}}],640,186,580,340,
  {lineOptions:{smooth:false},xAxis:{title:'危险格 / 个',textStyle:{fontSize:20,fill:C.ink},majorGridlines:null,line:{fill:C.line,width:1}},yAxis:{min:0,max:1.16,majorUnit:.5,numberFormatCode:'0%',textStyle:{fontSize:17,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:18,fill:C.ink}}});
 text(s,'有技能无路由 0.882　≫　有路由无技能 0.106',640,538,580,36,23,C.ink,true,'center');
 text(s,'该协议下，复用相关的差异被明显放大',640,576,580,34,21,C.orange,false,'center');
}
// ============ 主线 09：三层研究地图 ============
{
 const s=slide('三层研究地图',notes('research/2026-09-15-memory-compression-audit/PROJECT_STATUS.md','这三层不是全都已集成的一套通用算法。结构发现的受限未知顺序闭环有正结果；状态抽象的已知正确压缩有明确收益，可靠在线合并尚未解决；结构利用的路由/共享实现具有条件性收益，但一般机制归因、协议与任务边界未定。未建立超过ISA等的通用新方法，不声称自动机语法优于等价手写实现。'));
 text(s,'已验证',820,142,200,42,25,C.blue);text(s,'尚未解决',1050,142,200,42,25,C.orange);
 const labels=['结构发现','状态抽象','结构利用'];
 const ok=['受限在线闭环','正确压缩收益','条件性控制收益'];const open=['一般规则发现','可靠在线合并','机制归因与边界'];
 for(let i=0;i<3;i++){
 let y=219+i*143; text(s,labels[i],77,y+18,200,50,31,C.ink,true);
 text(s,ok[i],807,y+24,229,43,24,C.blue);text(s,open[i],1045,y+24,231,43,24,C.orange);
 if(i===0){let a=node(s,'A',327,y+13,70,62),b=node(s,'B',475,y+13,70,62),c=node(s,'?',623,y+13,70,62,C.gray);conn(s,a,b);conn(s,b,c,C.gray,'right','left',true);}
 if(i===1){let a=node(s,'ABC',327,y-4,100,49,C.gray),b=node(s,'ACB',327,y+63,100,49,C.gray),c=node(s,'q',581,y+28,110,58,C.blue);conn(s,a,c);conn(s,b,c);}
 if(i===2){let a=node(s,'q',327,y+13,70,62),b=node(s,'goal',475,y+13,88,62),c=node(s,'技能',631,y+13,102,62);conn(s,a,b);conn(s,b,c);}
 if(i<2)line(s,76,y+123,1150);
 }
}
// ============ 10：Backup 分隔 ============
{
 const s=p.slides.add();s.background.fill=C.white;
 text(s,'Backup',64,300,600,70,44,C.ink,true);
 text(s,'以下六页不在主线里，用于回答具体提问',64,384,760,44,25,C.muted);
 text(s,'10',1190,661,35,28,16,C.muted,false,'right');
 s.speakerNotes.textFrame.setText('主线到此结束。后面六页分别对应：反馈预算、前缀树与历史字典的等价、压缩倍数的闭式解释、合并审计的完整口径、五臂完整数值、危险格完整剂量曲线。');
}
// ============ B1：可靠任务反馈的预算 ============
{
 const s=slide('附录 · 可靠任务反馈的预算',notes('AUTO/discovery/src/run_verify.py','原始归档AUTO/discovery/results/verify.jsonl.gz：预算40/80/160/320，first90<200000的运行分别1/61/131/144，共144。160预算时最终仍>=90%的只有127/144；本图量的是曾经达标，不是最终成功比例。控制器按历史节点读取局部证据，未启用合并。不能由最终树节点数推断必须完整结构才能使用：nodes字段记录的是历史前缀树大小而非压缩状态数，且在运行结束时输出；真正的压缩状态数classes未被归档。'),'24 个任务族 × 6 个种子；200,000 环境步内曾达到 90%；未启用合并');
 chart(s,'bar',['40','80','160','320'],[{name:'达标运行比例',values:[0.006944444444,0.423611111111,0.909722222222,1],valuesFormatCode:'0%',fill:C.blue}],105,151,1080,400,{yAxis:{min:0,max:1.12,majorUnit:.25,numberFormatCode:'0%',textStyle:{fontSize:19,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},xAxis:{title:'提问预算 / 次',textStyle:{fontSize:25,fill:C.ink},line:{fill:C.line,width:1},majorGridlines:null},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:30,fill:C.ink}}});
 ['1 / 144','61 / 144','131 / 144','144 / 144'].forEach((t,i)=>text(s,t,209+i*245,560,190,40,23,C.muted,false,'center'));
 text(s,'只说明反馈越多越容易学会；不能推出「结构必须接近完整才能使用」',80,608,1120,36,22,C.orange,false,'center');
}
// ============ B2：前缀树与历史字典 ============
{
 const s=slide('附录 · 同一份历史的两种表示',notes('progressive_task_discovery/stage7_branching/REPORT.md','前缀树与完整成功事件历史dictionary在匹配输入与随机数时逐动作一致。字典包含足够历史，并非只存进度或最后事件。这里比较表示形式，不涉及合并不同历史。因此自动机的价值不来自「画成图」这件事本身，而来自后面能不能合并等价历史。'),'同样的信息量，两种写法逐动作一致 —— 所以价值不在图这种形式');
 text(s,'Prefix tree',150,145,330,45,29);text(s,'History dictionary',780,145,400,45,29);
 const r=node(s,'∅',102,353,62,62,C.gray),a=node(s,'A',247,262,65,62),b=node(s,'B',247,457,65,62,C.orange),ea=node(s,'E',402,262,65,62),eb=node(s,'E',402,457,65,62,C.orange),d=node(s,'D',548,262,65,62),c=node(s,'C',548,457,65,62,C.orange);
 [[r,a],[a,ea],[ea,d]].forEach(([x,y])=>conn(s,x,y,C.blue));[[r,b],[b,eb],[eb,c]].forEach(([x,y])=>conn(s,x,y,C.orange));
 text(s,'≡',654,343,90,85,65,C.muted,false,'center');
 text(s,'历史',786,247,170,42,24,C.muted);text(s,'下一目标',1040,247,190,42,24,C.muted);line(s,775,299,430);
 text(s,'(A, E)',800,327,180,55,32,C.blue);text(s,'D',1088,327,80,55,32,C.blue);
 text(s,'(B, E)',800,445,180,55,32,C.orange);text(s,'C',1088,445,80,55,32,C.orange);
 text(s,'逐动作一致',503,588,275,45,27,C.muted,false,'center');
}
// ============ B3：压缩倍数的闭式解释 ============
{
 const s=slide('附录 · 压缩倍数的闭式解释',notes('research/2026-09-15-memory-compression-audit/REVIEW.md','这些纯净状态内错误动作Q恒为0，正确动作一旦得到一次正奖励便变成1，此后贪心评估就答对该状态；未获正奖励的状态仍随机并列。若完整历史数为H，准确率=(H+c)/2H，要达95%需覆盖c>=ceil(0.9H)。在iid均匀排列与当前初始化下，已覆盖i格时新覆盖一格的概率为(H-i)/(2H)，期望等待2H/(H-i)，累加得 sum(2H/(H-i), i=0..ceil(0.9H)-1)。H=6/24/120时为29.40/109.25/543.76幕；压缩后H=2给出6.0幕。理论倍数90.63x，实测89.75x。正式T95按检查点记录，可能比逐幕精确时刻稍晚。这说明大倍数可以由明确的重复建表机制解释，不需要额外假设。'),'大倍数来自要填多少张表，不需要额外的机制假设');
 text(s,'为什么是这个量级',72,132,540,44,27,C.ink,true);
 const L=['错误动作 Q 恒为 0；正确动作首次拿到正奖励后 Q = 1，','此后这个状态永远答对',
          '未覆盖的状态随机并列，答对一半','→ 准确率 = (H + c) / 2H，要达 95% 需覆盖 c ≥ 0.9H',
          '已覆盖 i 格时新覆盖一格的概率 = (H − i) / 2H','→ 期望等待 2H / (H − i)',
          'T95 = Σ 2H/(H − i)，i = 0 … ⌈0.9H⌉ − 1','H = n! = 6 / 24 / 120；压缩后 H = 2 → 6.0 幕'];
 L.forEach((t,i)=>text(s,t,72,196+i*46,560,40,i%2?20:21,i%2?C.blue:C.ink));
 chart(s,'bar',['3 个零件','4 个零件','5 个零件'],[{name:'闭式理论',values:[29.40,109.25,543.76],fill:C.gray},{name:'实测',values:[28.28,109.15,547.23],fill:C.blue}],650,180,566,380,{barOptions:{direction:'column',grouping:'clustered',gapWidth:160},yAxis:{min:0,max:680,majorUnit:200,title:'平均 T95 / 幕',textStyle:{fontSize:16,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:15,fill:C.ink}}});
 text(s,'理论倍数 90.63×　　实测 89.75×',650,578,566,40,25,C.blue,true,'center');
}
// ============ B4：合并审计完整口径 ============
{
 const s=slide('附录 · 合并审计的完整口径',notes('progressive_task_discovery/stage8_compression/REPORT_STEP35.md','Step35：n=3的8个任务族，792次查询，其中冗余313次（39.5%）。k_min下被强制的合并236次：裁决错47次故裁决正确率80.1%，归属错140次故归属正确率40.7%，省下134/792=16.9%。k_min+1下复用22次：裁决错5、归属错8，省下14/792=1.8%。裁决正确率高于归属正确率，是因为大量错误归属碰巧给出相同裁决，这种运气撑不过下一个事件。另有102次复用落在必要查询上，是替换不是节省，未计入16.9%。'),'裁决正确 = 借来的答案与真裁决一致；归属正确 = 被合并的伙伴确实是同一任务状态');
 ['裁决正确率','状态归属正确率','可省查询比例'].forEach((t,i)=>text(s,t,67+i*402,145,363,45,27,C.ink,false,'center'));
 [[.801,.773],[.407,.636],[.169,.018]].forEach((v,i)=>chart(s,'bar',['k_min','k_min+1'],[{name:'比例',values:v,valuesFormatCode:'0.0%',fill:C.blue,points:[{idx:1,fill:C.orange}]}],62+i*402,208,353,360,{yAxis:{min:0,max:1,majorUnit:.25,numberFormatCode:'0%',textStyle:{fontSize:17,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:24,fill:C.ink}}}));
 text(s,'236 次强制合并中 140 次把真状态不同的历史证成同态；在线 k_min 为 8–12，真状态数 19–27',80,586,1120,36,22,C.ink,false,'center');
 text(s,'预注册判据 3 / 3 不达标：节省 16.9% < 20%；错误裁决 47；错误归属 140；敏感性上界 23.6% 仍不达标',80,624,1120,34,21,C.red,false,'center');
}
// ============ B5：五臂完整数值 ============
{
 const s=slide('附录 · 结构利用五臂完整数值',notes('progressive_task_discovery/stage8_compression/REPORT_STEP36.md','24族×30种子×5臂，预算200000步。first90是中位数，AUC为成功率曲线面积。原报告关于路由压倒性强于复用的AUC解释已在Step38撤回，因天花板影响。删失：原始归档results/factorial.json中first90=-1共40次，全部落在Y00（30/720）与Y01（10/720），Y10/Y11/QRM为0；页面中位数按达标者计，Y00若把-1记作预算200000则中位为41000而非38500，该丢弃使Y00/Y01显得偏好，因此对比是保守的。报告曾给出样本复杂度主效应路由6.39x、复用5.34x，本次未能从归档复现（把-1记作预算得9.03x/4.50x），故不列该口径。'),'AUC 接近天花板，不能据此排序机制强弱');
 text(s,'无复用',266,144,185,40,24,C.muted,false,'center');text(s,'有复用',463,144,185,40,24,C.muted,false,'center');
 text(s,'无路由',61,260,140,40,24,C.muted);text(s,'有路由',61,463,140,40,24,C.muted);
 const cells=[['Y00','平坦历史','.821',222,210],['Y01','学习型元控制','.883',438,210],['Y10','分阶段技能','.988',222,411],['Y11','共享技能','.997',438,411]];
 for(const [id,sub,auc,x,y] of cells){rect(s,x,y,196,171,id==='Y11'?C.light:C.white,C.line);text(s,id,x+18,y+13,160,46,29,id==='Y11'?C.blue:C.ink,true);text(s,sub,x+18,y+62,172,40,19,C.muted);text(s,'AUC '+auc,x+18,y+113,166,35,22,C.ink);}
 text(s,'first90 中位 / 环境步',754,140,425,42,25);
 chart(s,'bar',['Y00','Y01','Y10','Y11','QRM'],[{name:'first90',values:[38500,4000,3750,1000,7000],fill:C.gray,points:[{idx:1,fill:C.orange},{idx:2,fill:'#72A0BA'},{idx:3,fill:C.blue}]}],694,220,525,350,{yAxis:{min:0,max:45000,majorUnit:15000,numberFormatCode:'#,##0',textStyle:{fontSize:16,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:20,fill:C.ink}}});
 text(s,'QRM  AUC .972',812,588,340,35,23,C.muted,false,'center');
 text(s,'删失：Y00 有 30/720、Y01 有 10/720 次从未达标，中位数按达标者计',80,626,1120,32,19,C.orange,false,'center');
}
// ============ B6：危险格完整剂量曲线 ============
{
 const s=slide('附录 · 危险格剂量扫描',notes('progressive_task_discovery/stage8_compression/REPORT_STEP38_5.md','Step38.5剂量扫描，昂贵导航，自有导出模拟器，任务规则已给定，训练预算1000000环境步。达标率为预算内达到90%成功率的运行比例。此扫描的逐种子原始数据后来确认遗失，因此不能重算置信区间。Step40新网格逐运行归档：步费改变QRM排序，并使风险交互效应缩小；不能将此页视为协议不变的效应大小。无论存在何种图论路径，危险格可能迫使成功轨迹绕行，这里不声明几何难度完全不变。底行取自REPORT_STEP38_5第125-126行。'),'已知规则；固定步费协议；100 万步预算；逐种子记录已缺失，不能重算区间');
 chart(s,'line',['0','1','2','3','4'],[
 {name:'Y10  分阶段技能',values:[1,.32,.16,.11,.10],valuesFormatCode:'0%',line:{fill:C.orange,width:3},marker:{symbol:'circle',size:8}},
 {name:'Y11  共享技能',values:[1,1,.99,.95,.91],valuesFormatCode:'0%',line:{fill:C.blue,width:3},marker:{symbol:'circle',size:8}}],91,144,1105,424,{lineOptions:{smooth:false},xAxis:{title:'危险格 / 个',textStyle:{fontSize:24,fill:C.ink},majorGridlines:null,line:{fill:C.line,width:1}},yAxis:{min:0,max:1.15,majorUnit:.25,numberFormatCode:'0%',textStyle:{fontSize:20,fill:C.muted},majorGridlines:{fill:'#E9EEF1',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:24,fill:C.ink}}});
 line(s,150,584,980);
 text(s,'有技能无路由 0.882　≫　有路由无技能 0.106',80,596,1120,38,25,C.ink,true,'center');
 text(s,'不可逆性放大的是复用，不是路由（无危险格时两臂同为 100%）',80,631,1120,32,21,C.orange,false,'center');
}
await fs.mkdir(WORK+'/.build/previews_v3',{recursive:true});
await (await PresentationFile.exportPptx(p)).save(WORK+'/.build/candidate_v3.pptx');
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];const png=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(WORK+'/.build/previews_v3/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
 console.log('rendered',i+1);
}
const FINAL=WORK+'/output/任务结构研究_主线9页_v3.pptx';
const result=await finalizePresentation({workspaceDir:WORK,candidatePath:WORK+'/.build/candidate_v3.pptx',finalPath:FINAL,
 pythonExecutable:'/Users/yuhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:16,requiredNativeChartOwnerSlides:[2,5,6,7,8,11,13,14,15,16],requiredNativeTableOwnerSlides:[],
 materializeLiteralChartWorkbooks:true,nativeChartTargetApplication:'portable',fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
 receiptPath:WORK+'/.build/validation_final2.json'});
console.log('OK', result?.finalPath ?? FINAL);
