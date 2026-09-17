#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int W=9, N=W*W, M=8, MAXQ=32;
const std::array<int,M> CELLS={0,4,8,36,44,72,76,80};
struct Move { int s,event; };
Move physical(int s,int a) {
    int x=s%W,y=s/W;
    if(a==0)y=std::max(0,y-1);
    if(a==1)x=std::min(W-1,x+1);
    if(a==2)y=std::min(W-1,y+1);
    if(a==3)x=std::max(0,x-1);
    int z=y*W+x, e=-1;
    if(z!=s)for(int j=0;j<M;j++)if(CELLS[j]==z)e=j;
    return {z,e};
}
std::array<std::array<Move,4>,N> moves;
uint64_t mix(uint64_t x) {x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
void hash_add(uint64_t &h,uint64_t x) {h^=mix(x);h*=1099511628211ULL;}
double uniform(uint64_t x) {return (x>>11)*0x1.0p-53;}
struct Config {
    std::string method="count";
    int k=4,seed=0,budget=200000,every=5000,hfactor=24,master_n=5,window=10;
    double slip=0,eps=.2,high=.5,low=.02,eta=.8,alpha=.3,gamma=.98;
};
struct TaskEnvironment {
    std::vector<int> target;
    int s=40,q=0,t=0,H;
    struct Observation {int s,event;bool progress,done,success;};
    explicit TaskEnvironment(const Config &c):H(c.hfactor*c.k) {
        std::mt19937_64 rng(mix(c.seed+101));
        for(int i=0;i<c.k;i++) {int e;do{e=rng()%M;}while(i&&e==target.back());target.push_back(e);}
    }
    void reset(){s=40;q=0;t=0;}
    Observation step(int a) {
        auto m=moves[s][a];s=m.s;t++;
        bool p=m.event==target[q];if(p)q++;
        bool win=q==int(target.size());
        return {s,m.event,p,win||t>=H,win};
    }
};
struct Agent {
    Config c;
    std::vector<double> Q;
    std::vector<double> G;
    struct Sample {int q,s,a,ns;bool p,win;};
    std::vector<Sample> replay;
    std::array<int,MAXQ> edge;
    std::array<bool,MAXQ> terminal{};
    std::array<std::deque<int>,MAXQ> outcomes;
    std::array<long long,MAXQ> stage_steps{},stage_visits{},stage_success{},low_steps{},discovery_step{};
    int stage=0,depth=0;
    bool full_task=false;
    long long updates=0;
    bool flat,adaptive,cf,oracle,goal,local,buffered;
    explicit Agent(const Config &config):c(config),Q(MAXQ*N*4,0),G(M*N*4,0),flat(c.method=="flat"),
        adaptive(c.method.find("progressive")!=std::string::npos||c.method=="count_frontier"),
        cf(c.method.find("qrm")!=std::string::npos),oracle(c.method=="oracle_qrm"||c.method=="oracle_goal"),
        goal(c.method.find("goal")!=std::string::npos),local(c.method.find("option")!=std::string::npos),buffered(c.method.find("replay")!=std::string::npos) {edge.fill(-1);}
    int ix(int q,int s,int a)const{return ((flat?0:q)*N+s)*4+a;}
    double best(int q,int s)const {double v=Q[ix(q,s,0)];for(int a=1;a<4;a++)v=std::max(v,Q[ix(q,s,a)]);return v;}
    double policy_value(int q,int s,int a)const {return goal&&edge[q]>=0&&(c.method!="delayed_goal"||full_task)?G[(edge[q]*N+s)*4+a]:Q[ix(q,s,a)];}
    int greedy(int q,int s)const {int a=0;for(int b=1;b<4;b++)if(policy_value(q,s,b)>policy_value(q,s,a))a=b;return a;}
    bool mastered(int q)const {
        int sum=0;for(int v:outcomes[q])sum+=v;
        return sum>=c.master_n && double(sum)>=c.eta*outcomes[q].size();
    }
    double epsilon()const{return adaptive?(mastered(stage)?c.low:c.high):c.eps;}
    int act(int s,double er,uint64_t ra,uint64_t tie)const {
        if(er<epsilon())return ra%4;
        double v=policy_value(stage,s,greedy(stage,s));int actions[4],n=0;
        for(int a=0;a<4;a++)if(policy_value(stage,s,a)==v)actions[n++]=a;
        return actions[tie%n];
    }
    void one_update(int q,int s,int a,int ns,bool progress,bool success) {
        int nq=q+int(progress);
        double reward=double(progress)-.01;
        double target=reward+((success||(local&&progress))?0:c.gamma*best(nq,ns));
        double &v=Q[ix(q,s,a)];v+=c.alpha*(target-v);updates++;
    }
    void outcome(int q,bool success) {
        outcomes[q].push_back(success);if(int(outcomes[q].size())>c.window)outcomes[q].pop_front();
    }
    void observe(int s,int a,const TaskEnvironment::Observation&o,long long step) {
        int q=stage;stage_steps[q]++;if(adaptive&&mastered(q))low_steps[q]++;
        if(o.success)full_task=true;
        if(o.progress) {
            stage_success[q]++;
            // Count-frontier deliberately needs no learned event transition labels.
            if(c.method!="count_frontier" && edge[q]<0) {edge[q]=o.event;terminal[q]=o.success;discovery_step[q]=step;depth=std::max(depth,q+1);}
        }
        one_update(q,s,a,o.s,o.progress,o.success);
        if(cf)for(int j=0;j<MAXQ-1;j++)if(j!=q&&edge[j]>=0) {
            bool p=o.event==edge[j];one_update(j,s,a,o.s,p,p&&terminal[j]);
        }
        if(goal)for(int g=0;g<M;g++) {
            if(c.method=="known_goal"&&std::find(edge.begin(),edge.end(),g)==edge.end())continue;
            bool p=o.event==g;int z=(g*N+o.s)*4;
            double mx=*std::max_element(G.begin()+z,G.begin()+z+4);
            double target=double(p)-.01+(p?0:c.gamma*mx);
            double &v=G[(g*N+s)*4+a];v+=c.alpha*(target-v);updates++;
        }
        if(buffered) {
            Sample x{q,s,a,o.s,o.progress,o.success};
            if(replay.size()<10000)replay.push_back(x);else replay[(step-1)%10000]=x;
            for(int j=0;j<8;j++) {
                // Independent deterministic replay draws do not consume behavior/environment RNG.
                auto y=replay[mix(uint64_t(step)*8+j+mix(c.seed+303))%replay.size()];
                one_update(y.q,y.s,y.a,y.ns,y.p,y.win);
            }
        }
        if(o.progress||o.done)outcome(q,o.progress);
        if(o.progress)stage++;
        if(o.done)stage=0;
        if(o.progress&&!o.done)stage_visits[stage]++;
        if(o.done)stage_visits[0]++;
    }
};
struct Evaluation {double success,steps;};
Evaluation evaluate(const Agent&a,const TaskEnvironment&env) {
    int k=env.target.size(),H=env.H;
    std::vector<int> policy(k*N);for(int q=0;q<k;q++)for(int s=0;s<N;s++)policy[q*N+s]=a.greedy(q,s);
    if(a.c.slip==0) {
        int s=40,q=0;
        for(int t=0;t<H;t++){auto m=moves[s][policy[q*N+s]];s=m.s;if(m.event==env.target[q])q++;if(q==k)return {1.,double(t+1)};}
        return {0.,double(H)};
    }
    std::vector<double> current(k*N,0),next(k*N,0);current[40]=1.;double success=0,steps=0;
    for(int t=0;t<H;t++) {
        std::fill(next.begin(),next.end(),0);
        for(int q=0;q<k;q++)for(int s=0;s<N;s++) {
            double mass=current[q*N+s];if(mass==0)continue;steps+=mass;
            for(int b=0;b<4;b++) {
                double p=a.c.slip/4+(b==policy[q*N+s]?1-a.c.slip:0);
                if(p==0)continue;
                auto m=moves[s][b];int nq=q+(m.event==env.target[q]);
                if(nq==k)success+=mass*p;else next[nq*N+m.s]+=mass*p;
            }
        }
        current.swap(next);
    }
    return {std::min(1.,success),steps};
}
struct Checkpoint {int step;Evaluation ev;int depth;};
template<class T>void json_array(const T &v,int n=-1) {std::cout<<"[";int i=0;for(auto x:v){if(n>=0&&i>=n)break;if(i++)std::cout<<",";std::cout<<x;}std::cout<<"]";}
void run(const Config &c) {
    TaskEnvironment env(c);Agent a(c);
    if(a.oracle){for(int q=0;q<c.k;q++){a.edge[q]=env.target[q];a.terminal[q]=q==c.k-1;}a.depth=c.k;}
    std::mt19937_64 rng(mix(c.seed+202));
    uint64_t trace=1469598103934665603ULL;
    int episodes=0,train_wins=0,first_train_success=-1;
    long long prefix_steps=0,frontier_steps=0,prefix_at_first=-1,frontier_at_first=-1;
    a.stage_visits[0]=1;
    std::vector<Checkpoint> checks{{0,evaluate(a,env),a.depth}};
    for(int step=1;step<=c.budget;step++) {
        double er=uniform(rng());uint64_t ra=rng(),tie=rng();double sr=uniform(rng());uint64_t sa=rng();
        int s=env.s;
        if(a.edge[a.stage]>=0)prefix_steps++;else frontier_steps++;
        // Oracle-state diagnostic has exactly the information obtainable by counting progress.
        if(c.method=="oracle_state")a.stage=env.q;
        assert(a.stage==env.q);
        int action=a.act(s,er,ra,tie),actual=(sr<c.slip)?int(sa%4):action;
        auto o=env.step(actual);
        hash_add(trace,uint64_t(s+N*(action+4*(int(o.progress)+2*a.stage))));
        a.observe(s,action,o,step);
        if(o.done){episodes++;if(o.success){train_wins++;if(first_train_success<0){first_train_success=step;prefix_at_first=prefix_steps;frontier_at_first=frontier_steps;}}env.reset();}
        if(step%c.every==0||step==c.budget)checks.push_back({step,evaluate(a,env),a.depth});
    }
    uint64_t qhash=1469598103934665603ULL;
    for(double x:a.Q){union{double d;uint64_t u;}z;z.d=x;hash_add(qhash,z.u);}
    if(a.goal)for(double x:a.G){union{double d;uint64_t u;}z;z.d=x;hash_add(qhash,z.u);}
    int hit=-1,stable=-1;double auc=0;
    for(size_t j=1;j<checks.size();j++){
        if(checks[j].ev.success>=.9&&hit<0)hit=checks[j].step;
        if(checks[j].ev.success>=.9&&checks[j-1].ev.success>=.9&&stable<0)stable=checks[j-1].step;
        auc+=(checks[j].step-checks[j-1].step)*(checks[j].ev.success+checks[j-1].ev.success)/2/c.budget;
    }
    std::cout<<std::setprecision(12);
    std::cout<<"{\"method\":\""<<c.method<<"\",\"seed\":"<<c.seed<<",\"k\":"<<c.k<<",\"budget\":"<<c.budget<<",\"horizon\":"<<env.H<<",\"slip\":"<<c.slip
       <<",\"epsilon\":"<<c.eps<<",\"high\":"<<c.high<<",\"low\":"<<c.low<<",\"alpha\":"<<c.alpha<<",\"gamma\":"<<c.gamma<<",\"master_n\":"<<c.master_n<<",\"eta\":"<<c.eta
       <<",\"first90\":"<<hit<<",\"stable90\":"<<stable<<",\"auc\":"<<auc<<",\"final_success\":"<<checks.back().ev.success<<",\"final_eval_steps\":"<<checks.back().ev.steps
       <<",\"updates\":"<<a.updates<<",\"episodes\":"<<episodes<<",\"train_wins\":"<<train_wins<<",\"first_train_success\":"<<first_train_success
       <<",\"prefix_steps_at_first_success\":"<<prefix_at_first<<",\"frontier_steps_at_first_success\":"<<frontier_at_first
       <<",\"trace_hash\":\""<<trace<<"\",\"q_hash\":\""<<qhash<<"\",\"task\":";json_array(env.target);
    std::cout<<",\"learned_edges\":";json_array(a.edge,c.k);
    std::cout<<",\"discovery_steps\":";json_array(a.discovery_step,c.k);
    std::cout<<",\"stage_steps\":";json_array(a.stage_steps,c.k);
    std::cout<<",\"stage_visits\":";json_array(a.stage_visits,c.k);
    std::cout<<",\"stage_successes\":";json_array(a.stage_success,c.k);
    std::cout<<",\"low_exploration_steps\":";json_array(a.low_steps,c.k);
    std::cout<<",\"checkpoints\":[";
    for(size_t j=0;j<checks.size();j++){if(j)std::cout<<",";auto x=checks[j];std::cout<<"["<<x.step<<","<<x.ev.success<<","<<x.ev.steps<<","<<x.depth<<"]";}
    std::cout<<"],\"final_policy\":[";for(int q=0;q<c.k;q++)for(int s=0;s<N;s++){if(q||s)std::cout<<",";std::cout<<a.greedy(q,s);}
    std::cout<<"]}\n";
}
void self_test() {
    assert(physical(0,0).event==-1);assert(physical(1,3).event==0);
    Config c;c.k=3;TaskEnvironment env(c);env.target={0,2,0};
    // Real event-driven trace with wrong event ignored and repeated symbol separated by another goal.
    env.s=1;auto x=env.step(3);assert(x.progress&&env.q==1);
    x=env.step(0);assert(!x.progress&&x.event==-1);
    env.s=3;x=env.step(1);assert(!x.progress&&env.q==1);
    env.s=7;x=env.step(1);assert(x.progress&&env.q==2);
    env.s=1;x=env.step(3);assert(x.success&&x.progress);
    // Unknown edges are not relabeled. Once a different edge is known, exactly one extra update.
    c.method="learned_qrm";Agent a(c);
    TaskEnvironment::Observation o{1,-1,false,false,false};a.observe(2,3,o,1);assert(a.updates==1);
    a.edge[1]=2;a.observe(2,3,o,2);assert(a.updates==3);
    // Bellman terminal versus time-limit bootstrap.
    Agent b(c);b.Q[b.ix(1,0,0)]=10;b.one_update(0,1,3,0,true,true);
    assert(std::abs(b.Q[b.ix(0,1,3)]-.3*.99)<1e-12);
    Agent d(c);d.Q[d.ix(0,0,0)]=10;d.one_update(0,1,3,0,false,false);
    assert(std::abs(d.Q[d.ix(0,1,3)]-.3*(-.01+.98*10))<1e-12);
    c.method="option_count";Agent opt(c);opt.Q[opt.ix(1,0,0)]=10;opt.one_update(0,1,3,0,true,false);
    assert(std::abs(opt.Q[opt.ix(0,1,3)]-.3*.99)<1e-12);
    c.method="goal_reuse";Agent goal(c);goal.observe(1,3,{0,0,true,false,false},1);
    assert(goal.updates==9);assert(std::abs(goal.G[7]-.3*.99)<1e-12);
    assert(goal.edge[0]==0&&goal.edge[1]==-1);
    c.method="count_replay";Agent rep(c);rep.observe(1,3,{0,0,true,false,false},1);assert(rep.updates==9);
    c.method="known_goal";Agent known(c);known.observe(1,3,{0,0,true,false,false},1);assert(known.updates==2);
    assert(known.G[(1*N+1)*4+3]==0);
    c.method="delayed_goal";Agent delayed(c);delayed.edge[0]=0;
    delayed.G[40*4]=2;delayed.Q[delayed.ix(0,40,2)]=1;
    assert(delayed.greedy(0,40)==2);delayed.full_task=true;assert(delayed.greedy(0,40)==0);
    // Exhaustive acceptance check against a combinatorial subsequence oracle.
    for(int length=0;length<=6;length++)for(int code=0;code<(1<<(2*length));code++) {
        std::vector<int> word;int z=code;
        for(int t=0;t<length;t++){word.push_back(z%4-1);z/=4;}
        bool accepted=false;
        for(int i=0;i<length;i++)for(int j=i+1;j<length;j++)for(int h=j+1;h<length;h++)
            if(word[i]==0&&word[j]==2&&word[h]==0)accepted=true;
        env.reset();bool actual=false;
        for(int e:word) {
            TaskEnvironment::Observation obs;
            if(e<0){env.s=40;obs=env.step(0);}
            else {env.s=CELLS[e]+9;obs=env.step(0);}
            if(obs.success){actual=true;break;}
        }
        assert(actual==accepted);
    }
    std::cout<<"{\"cpp_self_tests\":\"passed\"}\n";
}
int main(int argc,char**argv) {
    for(int s=0;s<N;s++)for(int a=0;a<4;a++)moves[s][a]=physical(s,a);
    if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
    Config c;
    for(int i=1;i<argc;i+=2){if(i+1>=argc)throw std::runtime_error("missing value");std::string f=argv[i],v=argv[i+1];
        if(f=="--method")c.method=v;else if(f=="--k")c.k=std::stoi(v);else if(f=="--seed")c.seed=std::stoi(v);
        else if(f=="--budget")c.budget=std::stoi(v);else if(f=="--every")c.every=std::stoi(v);else if(f=="--hfactor")c.hfactor=std::stoi(v);
        else if(f=="--slip")c.slip=std::stod(v);else if(f=="--epsilon")c.eps=std::stod(v);else if(f=="--high")c.high=std::stod(v);else if(f=="--low")c.low=std::stod(v);
        else if(f=="--master-n")c.master_n=std::stoi(v);else if(f=="--eta")c.eta=std::stod(v);else if(f=="--alpha")c.alpha=std::stod(v);else throw std::runtime_error("unknown flag "+f);
    }
    const std::vector<std::string> methods={"flat","count","discovered","oracle_state","progressive","count_frontier","learned_qrm","progressive_qrm","oracle_qrm","goal_reuse","progressive_goal","oracle_goal","option_count","option_progressive","option_qrm","option_progressive_qrm","count_replay","option_replay","delayed_goal","known_goal"};
    if(std::find(methods.begin(),methods.end(),c.method)==methods.end()||c.k<1||c.k>=MAXQ||c.budget<1||c.every<1||c.slip<0||c.slip>1)throw std::runtime_error("invalid config");
    run(c);
}
