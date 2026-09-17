#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int W=13,N=169,M=5,SLOTS=128;
const std::array<int,M> CELLS={18,20,78,90,162};
const std::array<int,2> START={17,21};
constexpr int J=84;
bool walkable(int s){int x=s%W,y=s/W;return (y==1&&x>=4&&x<=8)||(x==6&&y>=1&&y<=12)||(y==6);}
struct Move{int s,event;};
Move physical(int s,int a){
    int x=s%W,y=s/W;static const int dx[]={0,1,0,-1},dy[]={-1,0,1,0};
    int xx=x+dx[a],yy=y+dy[a],z=s;
    if(xx>=0&&xx<W&&yy>=0&&yy<W&&walkable(yy*W+xx))z=yy*W+xx;
    int e=-1;if(z!=s)for(int g=0;g<M;g++)if(z==CELLS[g])e=g;
    return {z,e};
}
std::array<std::array<Move,4>,N> mv;
uint64_t mix(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
void hashadd(uint64_t&h,uint64_t x){h^=mix(x);h*=1099511628211ULL;}
double uni(uint64_t x){return (x>>11)*0x1.0p-53;}
struct Config{std::string method="count_replay";int k=8,seed=0,budget=80000,every=1000,hfactor=20,bridge=0,branching=1;double epsilon=.05;};
struct Context{uint64_t key=0;int depth=0,cue=-1;void advance(int e){key=key*6+e+1;if(depth==0)cue=e;depth++;}};
struct Observation{int s,event;bool progress,done,success;};
struct Environment{
    std::array<std::vector<int>,2> seq;
    int s,depth=0,branch=-1,t=0,H,k,decision,branching;
    explicit Environment(const Config&c):H(c.hfactor*c.k),k(c.k),decision(c.bridge+1),branching(c.branching){
        std::mt19937_64 rng(mix(c.seed+111));std::vector<int> g={2,3,4};
        for(int i=2;i>0;i--)std::swap(g[i],g[rng()%(i+1)]);
        seq[0]={0};seq[1]={1};
        for(int i=0;i<c.bridge;i++){int event=g[(i+c.bridge)%2];if(i==c.bridge-1)event=g[2];seq[0].push_back(event);seq[1].push_back(event);}
        seq[0].push_back(g[0]);seq[1].push_back(g[1]);
        if(k>decision+1){seq[0].push_back(g[2]);seq[1].push_back(g[2]);}
        for(int d=decision+2;d<k;d++){int event;do{event=2+rng()%3;}while(event==seq[0].back());seq[0].push_back(event);seq[1].push_back(event);}
        if(!branching)for(int d=1;d<k;d++)seq[1][d]=seq[0][d];
        reset(0);
    }
    void reset(int context){s=START[context];depth=0;branch=-1;t=0;}
    int true_state()const{return !branching?depth:(depth==0?0:(depth<=decision?2*depth-1+branch:depth+decision));}
    bool valid(int g)const{return depth==0?(g==0||g==1):(g==seq[branch][depth]);}
    Observation step(int action){
        auto m=mv[s][action];s=m.s;t++;bool p=false;
        if(depth==0&&m.event>=0&&m.event<2){branch=m.event;p=true;}
        else if(depth>0&&m.event==seq[branch][depth])p=true;
        if(p)depth++;bool win=depth==k;
        return{s,m.event,p,win||t>=H,win};
    }
};
struct Node{
    uint64_t key;std::array<int,M> edges;std::array<int,M> discovered;
    explicit Node(uint64_t key=0):key(key){edges.fill(-1);discovered.fill(-1);}
};
struct Sample{int f,s,a,nf,ns;bool p,win;};
struct Agent{
    Config c;Context ctx;std::map<uint64_t,int> ids;std::vector<Node> nodes;
    std::map<int,std::set<int>> binding_edges;
    std::vector<double> Q,G;std::vector<Sample> buffer;std::array<std::vector<int>,2> oracle_seq;
    std::array<bool,2> solved={false,false};long long updates=0;
    bool skills,hand,count,oracle,delay,last_event;
    explicit Agent(const Config&c,const Environment&env):c(c),Q(SLOTS*N*4,0),G(M*N*4,0),
      skills(c.method.find("skill")!=std::string::npos),hand(c.method=="hand_replay"||c.method=="hand_skill"||c.method=="oracle_skill"),
      count(c.method=="count_replay"),oracle(c.method=="oracle_skill"),delay(c.method=="delayed_skill"),last_event(c.method=="last_event_replay"){
        ensure(0);if(oracle)oracle_seq=env.seq;
    }
    int ensure(uint64_t key){auto it=ids.find(key);if(it!=ids.end())return it->second;int id=nodes.size();assert(id<SLOTS);ids[key]=id;nodes.emplace_back(key);return id;}
    int node(const Context&x)const{auto it=ids.find(x.key);return it==ids.end()?-1:it->second;}
    int feature(const Context&x)const{if(count||(hand&&!c.branching))return x.depth;if(last_event)return x.depth==0?0:(x.depth-1)*M+int(x.key%6);if(hand)return x.depth==0?0:(x.depth<=c.bridge+1?2*x.depth-1+x.cue:x.depth+c.bridge+1);return node(x);}
    double qval(int f,int s,int a)const{return f<0?0:Q[(f*N+s)*4+a];}
    double best(int f,int s)const{double b=qval(f,s,0);for(int a=1;a<4;a++)b=std::max(b,qval(f,s,a));return b;}
    int predicted_goal(const Context&x)const{
        if(oracle)return x.depth==0?0:oracle_seq[x.cue][x.depth];
        if(c.method=="hand_skill"){auto it=binding_edges.find(feature(x));return it==binding_edges.end()||it->second.empty()?-1:*it->second.begin();}
        int n=node(x);if(n<0)return -1;for(int g=0;g<M;g++)if(nodes[n].edges[g]>=0)return g;return -1;
    }
    int active_goal(const Context&x)const{return skills&&(!delay||(solved[0]&&solved[1]))?predicted_goal(x):-1;}
    double value(const Context&x,int s,int a)const{int g=active_goal(x);return g>=0?G[(g*N+s)*4+a]:qval(feature(x),s,a);}
    int greedy(const Context&x,int s)const{int a=0;for(int b=1;b<4;b++)if(value(x,s,b)>value(x,s,a))a=b;return a;}
    int act(int s,double er,uint64_t ra,uint64_t tie)const{
        if(er<c.epsilon)return ra%4;double v=value(ctx,s,greedy(ctx,s));int aa[4],n=0;
        for(int a=0;a<4;a++)if(value(ctx,s,a)==v)aa[n++]=a;return aa[tie%n];
    }
    void update(const Sample&x){double target=double(x.p)-.01+(x.win?0:.98*best(x.nf,x.ns));double&v=Q[(x.f*N+x.s)*4+x.a];v+=.3*(target-v);updates++;}
    void observe(int s,int action,const Observation&o,int step){
        int f=feature(ctx),oldnode=node(ctx);Context next=ctx;
        if(o.progress){binding_edges[f].insert(o.event);next.advance(o.event);int dest=ensure(next.key);if(nodes[oldnode].edges[o.event]<0){nodes[oldnode].edges[o.event]=dest;nodes[oldnode].discovered[o.event]=step;}}
        Sample x{f,s,action,feature(next),o.s,o.progress,o.success};update(x);
        if(skills){for(int g=0;g<M;g++){int base=(g*N+o.s)*4;double b=*std::max_element(G.begin()+base,G.begin()+base+4);bool hit=o.event==g;
            double target=double(hit)-.01+(hit?0:.98*b);double&v=G[(g*N+s)*4+action];v+=.3*(target-v);updates++;}}
        else{if(buffer.size()<10000)buffer.push_back(x);else buffer[(step-1)%10000]=x;
            for(int j=0;j<M;j++)update(buffer[mix(uint64_t(step)*M+j+mix(c.seed+333))%buffer.size()]);}
        if(o.success)solved[next.cue]=true;ctx=o.done?Context{}:next;
    }
};
Context true_context(const Environment&e,int b,int depth){Context x;for(int i=0;i<depth;i++)x.advance(e.seq[b][i]);return x;}
std::vector<int> policy(const Agent&a,const Environment&e){std::vector<int> out;for(int b=0;b<2;b++)for(int d=0;d<e.k;d++){auto x=true_context(e,b,d);for(int s=0;s<N;s++)out.push_back(a.greedy(x,s));}return out;}
struct Eval{std::array<double,2> success{},steps{};double mean()const{return(success[0]+success[1])/2;}};
Eval evaluate(const Agent&a,const Environment&model){
    Eval ev;auto pol=policy(a,model);
    for(int b=0;b<2;b++){Environment e=model;e.reset(b);for(int t=0;t<e.H;t++){int act=pol[(b*e.k+e.depth)*N+e.s];auto o=e.step(act);ev.steps[b]=t+1;if(o.success){ev.success[b]=1;break;}}}
    return ev;
}
double stochastic_count_eval(const Agent&a,const Environment&e,double epsilon){
    auto pol=policy(a,e);double total=0;
    for(int b=0;b<2;b++){
        std::vector<double> v(e.k*N,0),next(e.k*N,0);v[START[b]]=1;
        for(int t=0;t<e.H;t++){std::fill(next.begin(),next.end(),0);
            for(int d=0;d<e.k;d++)for(int s=0;s<N;s++){double mass=v[d*N+s];if(mass==0)continue;
                for(int act=0;act<4;act++){double p=epsilon/4+(act==pol[(b*e.k+d)*N+s]?1-epsilon:0);auto m=mv[s][act];int nd=d+(m.event==e.seq[b][d]);
                    if(nd==e.k)total+=mass*p/2;else next[nd*N+m.s]+=mass*p;}}
            v.swap(next);
        }
    }return total;
}
struct Diagnostics{double state_accuracy=0,junction_accuracy=0,goal_coverage=0,goal_accuracy=0;int feature_states=0,true_states=0;};
Diagnostics diagnostics(const Agent&a,const Environment&e){
    std::map<int,std::map<int,int>> counts;std::map<int,std::map<int,int>> junction;std::set<int> truth;
    int known=0,right=0;
    for(int b=0;b<2;b++)for(int d=0;d<e.k;d++){
        auto x=true_context(e,b,d);int f=a.feature(x),q=!e.branching?d:(d==0?0:(d<=e.decision?2*d-1+b:d+e.decision));truth.insert(q);
        if(f>=0){counts[f][q]++;if(d==e.decision)junction[f][q]++;}
        int g=a.predicted_goal(x);if(g>=0){known++;if(d==0?(g==0||g==1):(g==e.seq[b][d]))right++;}
    }
    int correct=0,jcorrect=0;for(auto&[f,qs]:counts){int best=0;for(auto [q,n]:qs)best=std::max(best,n);correct+=best;}
    for(auto&[f,qs]:junction){int best=0;for(auto [q,n]:qs)best=std::max(best,n);jcorrect+=best;}
    return{double(correct)/(2*e.k),double(jcorrect)/2,double(known)/(2*e.k),double(right)/(2*e.k),int(counts.size()),int(truth.size())};
}
struct Check{int step;Eval e;Diagnostics d;};
template<class V>void arr(const V&v){std::cout<<"[";int i=0;for(auto x:v){if(i++)std::cout<<",";std::cout<<x;}std::cout<<"]";}
void run(const Config&c){
    Environment e(c);Agent a(c,e);std::mt19937_64 rng(mix(c.seed+222));
    int episodes=0;std::array<int,2> firstsuccess={-1,-1},wins={0,0};
    long long prefix=0,frontier=0,bindings=0,wrong=0,skill_attempts=0,skill_success=0,skill_other_progress=0;
    int segment_goal=-1;uint64_t segment_key=0;
    // Per observed task-prefix and requested goal: attempts, target hits,
    // interruption by another progress event, timeouts, wrong-binding steps.
    std::map<std::pair<uint64_t,int>,std::array<long long,5>> skill_stats;
    std::array<long long,2> prefix_at={-1,-1},frontier_at={-1,-1};
    uint64_t trace=1469598103934665603ULL;
    std::vector<Check> checks{{0,evaluate(a,e),diagnostics(a,e)}};
    for(int step=1;step<=c.budget;step++){
        int s=e.s,g=a.active_goal(a.ctx);if(a.predicted_goal(a.ctx)>=0)prefix++;else frontier++;
        if(g>=0){bindings++;if(!e.valid(g)){wrong++;skill_stats[{a.ctx.key,g}][4]++;}if(segment_goal<0||segment_key!=a.ctx.key){skill_attempts++;segment_goal=g;segment_key=a.ctx.key;skill_stats[{segment_key,g}][0]++;}}
        double er=uni(rng());auto ra=rng(),tie=rng();int act=a.act(s,er,ra,tie);
        auto o=e.step(act);hashadd(trace,s+N*(act+4*(int(o.progress)+2*a.ctx.depth)));
        if(segment_goal>=0&&(o.progress||o.done)){auto&stats=skill_stats[{segment_key,segment_goal}];if(o.event==segment_goal&&o.progress){skill_success++;stats[1]++;}else if(o.progress){skill_other_progress++;stats[2]++;}else stats[3]++;segment_goal=-1;}
        int currentcue=o.success?e.branch:-1;
        a.observe(s,act,o,step);
        if(o.done){episodes++;if(o.success){wins[currentcue]++;if(firstsuccess[currentcue]<0){firstsuccess[currentcue]=step;prefix_at[currentcue]=prefix;frontier_at[currentcue]=frontier;}}e.reset(episodes%2);}
        if(step%c.every==0||step==c.budget)checks.push_back({step,evaluate(a,e),diagnostics(a,e)});
    }
    uint64_t qhash=1469598103934665603ULL;for(double v:a.Q){uint64_t bits;std::memcpy(&bits,&v,8);hashadd(qhash,bits);}if(a.skills)for(double v:a.G){uint64_t bits;std::memcpy(&bits,&v,8);hashadd(qhash,bits);}
    int hit=-1,stable=-1;double auc=0;
    for(size_t i=1;i<checks.size();i++){if(checks[i].e.mean()>=.9&&hit<0)hit=checks[i].step;
        if(checks[i].e.mean()>=.9&&checks[i-1].e.mean()>=.9&&stable<0)stable=checks[i-1].step;
        auc+=(checks[i].step-checks[i-1].step)*(checks[i].e.mean()+checks[i-1].e.mean())/2/c.budget;}
    auto last=checks.back();std::cout<<std::setprecision(12);
    std::cout<<"{\"method\":\""<<c.method<<"\",\"k\":"<<c.k<<",\"seed\":"<<c.seed<<",\"bridge\":"<<c.bridge<<",\"branching\":"<<c.branching<<",\"budget\":"<<c.budget<<",\"horizon\":"<<e.H<<",\"epsilon\":"<<c.epsilon
      <<",\"first90\":"<<hit<<",\"stable90\":"<<stable<<",\"auc\":"<<auc<<",\"final_success\":"<<last.e.mean()<<",\"final_branch_success\":";arr(last.e.success);
    std::cout<<",\"final_eval_steps\":";arr(last.e.steps);std::cout<<",\"first_branch_success\":";arr(firstsuccess);
    std::cout<<",\"prefix_at_branch_success\":";arr(prefix_at);std::cout<<",\"frontier_at_branch_success\":";arr(frontier_at);
    std::cout<<",\"task\":[";arr(e.seq[0]);std::cout<<",";arr(e.seq[1]);std::cout<<"],\"updates\":"<<a.updates<<",\"episodes\":"<<episodes<<",\"prefix_steps\":"<<prefix<<",\"frontier_steps\":"<<frontier
      <<",\"binding_steps\":"<<bindings<<",\"wrong_binding_steps\":"<<wrong<<",\"skill_attempts\":"<<skill_attempts<<",\"skill_successes\":"<<skill_success<<",\"skill_other_progress\":"<<skill_other_progress
      <<",\"state_accuracy\":"<<last.d.state_accuracy<<",\"junction_state_accuracy\":"<<last.d.junction_accuracy<<",\"next_goal_coverage\":"<<last.d.goal_coverage<<",\"next_goal_accuracy\":"<<last.d.goal_accuracy
      <<",\"feature_states\":"<<last.d.feature_states<<",\"true_states\":"<<last.d.true_states<<",\"structure_revisions\":0,\"trace_hash\":\""<<trace<<"\",\"q_hash\":\""<<qhash<<"\",\"checkpoints\":[";
    for(size_t i=0;i<checks.size();i++){if(i)std::cout<<",";auto x=checks[i];std::cout<<"["<<x.step<<","<<x.e.mean()<<","<<x.e.success[0]<<","<<x.e.success[1]<<","<<x.d.state_accuracy<<","<<x.d.goal_coverage<<"]";}
    std::cout<<"],\"nodes\":[";for(size_t i=0;i<a.nodes.size();i++){if(i)std::cout<<",";auto n=a.nodes[i];std::cout<<"{\"key\":"<<n.key<<",\"edges\":";arr(n.edges);std::cout<<",\"discovery_steps\":";arr(n.discovered);std::cout<<"}";}
    std::cout<<"],\"skill_context_stats\":[";bool comma=false;for(auto&[key,stats]:skill_stats){if(comma)std::cout<<",";comma=true;std::cout<<"["<<key.first<<","<<key.second;for(auto v:stats)std::cout<<","<<v;std::cout<<"]";}
    std::cout<<"],\"final_policy\":";arr(policy(a,e));
    if(a.count||a.last_event)std::cout<<",\"stochastic_final_success_e005\":"<<stochastic_count_eval(a,e,.05)<<",\"stochastic_final_success_e02\":"<<stochastic_count_eval(a,e,.2);
    std::cout<<"}\n";
}
void selftest(){
    assert(walkable(J)&&physical(17,1).event==0&&physical(21,3).event==1);
    assert(physical(17,0).s==17&&physical(CELLS[2],3).event==-1);
    Config c;c.k=4;Environment e(c);
    for(int b=0;b<2;b++){e.reset(b);auto o=e.step(b==0?1:3);assert(o.progress&&e.depth==1&&e.branch==b);e.step(b==0?1:3);for(int i=0;i<5;i++)e.step(2);assert(e.s==J&&e.depth==1);}
    assert(e.seq[0][1]!=e.seq[1][1]);
    c.method="count_replay";Agent count(c,e);auto x=true_context(e,0,1),y=true_context(e,1,1);assert(count.feature(x)==count.feature(y));
    c.method="learned_replay";Agent learned(c,e);learned.ensure(x.key);learned.ensure(y.key);assert(learned.feature(x)!=learned.feature(y));
    c.method="hand_replay";Agent hand(c,e);assert(hand.feature(x)!=hand.feature(y));assert(hand.feature(true_context(e,0,2))==hand.feature(true_context(e,1,2)));
    std::cout<<"{\"self_tests\":\"passed\"}\n";
}
int main(int argc,char**argv){
    for(int s=0;s<N;s++)for(int a=0;a<4;a++)mv[s][a]=physical(s,a);
    if(argc==2&&std::string(argv[1])=="--self-test"){selftest();return 0;}
    if(argc==2&&std::string(argv[1])=="--map"){std::cout<<"[";bool comma=false;for(int s=0;s<N;s++)if(walkable(s)){if(comma)std::cout<<",";comma=true;std::cout<<"["<<s;for(int a=0;a<4;a++)std::cout<<","<<mv[s][a].s<<","<<mv[s][a].event;std::cout<<"]";}std::cout<<"]\n";return 0;}
    Config c;for(int i=1;i<argc;i+=2){if(i+1>=argc)throw std::runtime_error("missing argument");std::string f=argv[i],v=argv[i+1];
      if(f=="--method")c.method=v;else if(f=="--k")c.k=std::stoi(v);else if(f=="--seed")c.seed=std::stoi(v);else if(f=="--budget")c.budget=std::stoi(v);else if(f=="--every")c.every=std::stoi(v);else if(f=="--hfactor")c.hfactor=std::stoi(v);else if(f=="--bridge")c.bridge=std::stoi(v);else if(f=="--branching")c.branching=std::stoi(v);else if(f=="--epsilon")c.epsilon=std::stod(v);else throw std::runtime_error("unknown flag");}
    const std::set<std::string> methods={"count_replay","last_event_replay","history_replay","hand_replay","learned_replay","delayed_skill","immediate_skill","oracle_skill","history_skill","hand_skill"};
    if(!methods.count(c.method)||c.k<3||c.k>16||c.bridge<0||c.k<c.bridge+2||c.budget<=0||c.every<=0||c.hfactor<12||c.epsilon<0||c.epsilon>1||(c.branching!=0&&c.branching!=1))throw std::runtime_error("invalid config");run(c);
}
