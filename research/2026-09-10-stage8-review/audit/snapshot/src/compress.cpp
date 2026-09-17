// Does a smaller task representation cost fewer environment samples?
//
// One tabular Q-learner, one map, one hidden task, sparse terminal reward. The
// arms differ only in what indexes the task dimension of the table: progress
// count, a sliding window of successful events, the bag of them, the full
// history, or the true minimal automaton state. Every arm sees identical
// observations and performs identical numbers of Bellman updates, so a
// difference is attributable to the representation.
//
// The task machine, its minimisation, the map and the balanced start set are
// all computed ahead of time and read from a task file, so the semantics live
// in one place and cannot drift between arms.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

constexpr int MAXF = 4096;

uint64_t mix(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
void hashadd(uint64_t&h,uint64_t x){h^=mix(x);h*=1099511628211ULL;}

struct Task {
    int n=0,d=0,m=0,pa=0,pb=0,alphabet=0,nstates=0,N=0,horizon=0,maxlen=0,minwindow=0;
    std::vector<int> minimal,accepting,failing,trans,moves,events,starts,lengths,
                     product,wrong_product,shuffled;
    std::vector<double> map_marginal,task_marginal;
    void load(const std::string&path){
        std::ifstream in(path);
        if(!in)throw std::runtime_error("cannot open "+path);
        in>>n>>d>>m>>pa>>pb>>alphabet>>nstates;
        minimal.resize(nstates);accepting.resize(nstates);trans.resize(size_t(nstates)*alphabet);
        for(int&x:minimal)in>>x;
        for(int&x:accepting)in>>x;
        failing.resize(nstates);
        for(int&x:failing)in>>x;
        for(size_t i=0;i<trans.size();i++)in>>trans[i];
        in>>N;moves.resize(size_t(N)*4);events.resize(size_t(N)*4);
        for(int s=0;s<N;s++)for(int a=0;a<4;a++)in>>moves[size_t(s)*4+a]>>events[size_t(s)*4+a];
        int S;in>>S;starts.resize(S);lengths.resize(S);
        for(int&x:starts)in>>x;
        for(int&x:lengths)in>>x;
        in>>horizon>>maxlen>>minwindow;
        product.resize(size_t(nstates)*N);
        for(size_t i=0;i<product.size();i++)in>>product[i];
        wrong_product.resize(size_t(nstates)*N);
        for(size_t i=0;i<wrong_product.size();i++)in>>wrong_product[i];
        shuffled.resize(size_t(nstates)*N);
        for(size_t i=0;i<shuffled.size();i++)in>>shuffled[i];
        if(!in)throw std::runtime_error("truncated task file");
        // Marginals of the same table, so every arm is in step units and the
        // only thing that changes is which coordinate the potential may read.
        map_marginal.assign(N,0.0);task_marginal.assign(nstates,0.0);
        for(int q=0;q<nstates;q++)for(int s=0;s<N;s++){
            map_marginal[s]+=double(product[size_t(q)*N+s])/nstates;
            task_marginal[q]+=double(product[size_t(q)*N+s])/N;
        }
    }
    int next_state(int q,int event)const{return event<0?q:trans[size_t(q)*alphabet+event];}
    // Transitions remaining to acceptance, for the privileged shaping arm only.
    std::vector<int> to_accept()const{
        std::vector<int> d(nstates,-1);std::vector<int> frontier;
        for(int q=0;q<nstates;q++)if(accepting[q]){d[q]=0;frontier.push_back(q);}
        while(!frontier.empty()){
            std::vector<int> next;
            for(int q=0;q<nstates;q++){
                if(d[q]>=0)continue;
                for(int e=0;e<alphabet;e++){int z=trans[size_t(q)*alphabet+e];
                    if(z!=q&&d[z]>=0&&(d[q]<0||d[z]+1<d[q]))d[q]=d[z]+1;}
                if(d[q]>=0)next.push_back(q);
            }
            if(next.empty())break;frontier=next;
        }
        for(int&x:d)if(x<0)x=0;
        return d;
    }
};

struct Config {
    std::string task,method="automaton",revision="transfer",goal_select="value";
    int seed=0,budget=200000,every=1000,window=1,theta=20,repartition=100,replay_budget=2000;
    int quota=0,struct_cap=60;double beta=0,risk=0,spatial_scale=-1;
    std::string target="scarcest",shape="none";
    bool strict_children=false,dump_tree=false,shape_normalize=false,doom_continues=false;
    bool crm=false,relabel=true;int replay=5;
    std::string merge_rule="sweep";
    std::string seed_structure;
    double shape_scale=0;
    double epsilon=.02,gamma=.99,alpha=.3,cost=.01,reward_ratio=8;
};

// Learned state merging.
//
// The learner keeps a prefix tree over successful-event histories and merges
// two nodes when the evidence collected at them agrees: same set of events
// observed to advance the task, and successors that are themselves merged. A
// node counts as evidence-bearing only after `theta` visits, so a node nobody
// has looked at yet cannot drag others into a class. Merging is greedy with
// propagation, in the style of evidence-driven state merging, and is recomputed
// from the current evidence whenever that evidence changes, so a merge made on
// thin evidence is undone the moment an observation contradicts it. Those
// undos are counted as structure revisions.
//
// Nothing here reads the task machine. The true state stored on each node is
// written only into the diagnostics.
struct Tree {
    int alphabet;
    std::vector<uint64_t> key;
    std::vector<int> truth,visits,block;
    std::vector<std::vector<int>> succ,advanced,ignored;
    std::vector<int> accepting,danger;
    std::unordered_map<uint64_t,int> ids;
    explicit Tree(int a):alphabet(a){add(1469598103934665603ULL,0);}
    int add(uint64_t k,int truth_state){
        int id=int(key.size());ids[k]=id;key.push_back(k);truth.push_back(truth_state);
        visits.push_back(0);block.push_back(id);accepting.push_back(0);danger.push_back(0);
        succ.emplace_back(alphabet,-1);advanced.emplace_back(alphabet,0);
        ignored.emplace_back(alphabet,0);return id;
    }
    int find_or_add(uint64_t k,int truth_state,bool&fresh){
        auto it=ids.find(k);if(it!=ids.end()){fresh=false;return it->second;}
        fresh=true;return add(k,truth_state);
    }
    int size()const{return int(key.size());}
    uint64_t required(int h)const{uint64_t m=0;for(int e=0;e<alphabet;e++)if(advanced[h][e])m|=1ULL<<e;return m;}
    int observations(int h)const{int n=0;for(int e=0;e<alphabet;e++)n+=advanced[h][e]+ignored[h][e];return n;}
    int tried(int h,int e)const{return advanced[h][e]+ignored[h][e];}
    // A quota turns "nothing contradicted this merge" into "nothing
    // contradicted it although every event was tried here". Without it the
    // rule is closed-world: absence of evidence licenses merging.
    bool quota_met(int h,int q)const{
        if(q<=0)return true;
        for(int e=0;e<alphabet;e++)if(tried(h,e)<q)return false;
        return true;
    }
    int scarcest_untried(int h,int q)const{
        int best=-1,fewest=q;
        for(int e=0;e<alphabet;e++)if(tried(h,e)<fewest){fewest=tried(h,e);best=e;}
        return best;
    }
    // Two nodes conflict when the same event has been seen advancing the task
    // at one and being ignored at the other. This is positive evidence of a
    // behavioural difference; equality of the observed advancing sets is not,
    // because early on almost every node has seen only one advancing event.
    bool conflict(int a,int b)const{
        for(int e=0;e<alphabet;e++){
            if(advanced[a][e]&&ignored[b][e])return true;
            if(ignored[a][e]&&advanced[b][e])return true;
        }
        return false;
    }
};

struct Partition {
    // Evidence-driven merging, rewritten after two defects were demonstrated in
    // the first version.
    //
    // Defect one: the conflict test read only the union-find representative's
    // own evidence, so once A and B were merged, evidence recorded at B was
    // invisible when A was later tested against C. A three-node counterexample
    // merged a pair that directly contradicted each other. The fix is to carry
    // aggregated evidence on the class: uniting two classes unions their
    // advanced, ignored, tried and successor records, and every test reads the
    // class, never a node.
    //
    // Defect two: the evidence quota was checked only on the pair being
    // considered, while the successor pairs dragged in by propagation were
    // united unconditionally. Children could therefore be merged with no
    // evidence at all. The fix checks every implied pair, under a flag so the
    // strict and permissive readings can be compared rather than assumed.
    //
    // Merging is now tentative: the closure is computed on a scratch structure
    // with an undo log, and is committed only if every implied pair passes.
    // Nothing partially applied can survive a failed attempt.
    // Blame travels back along the history that produced it. Marking only the
    // node where the episode failed does nothing, because that node is deep and
    // is never a merge candidate: in every run inspected, each wrongly merged
    // pair had zero failures of its own. The mistake is made shallow, in the
    // unordered phase, and paid for deep, at the commit. So a failure raises the
    // bar on every node along the episode's own path.
    // Risk-weighted commitment. A global evidence threshold trades precision
    // for recall everywhere, and at a high enough setting simply stops merging,
    // which is the full-history arm again. What the irreversible setting calls
    // for is conservatism *where being wrong has proved expensive*: a class that
    // has already led to a failure must clear a higher bar before it may absorb
    // another. The agent observes its own failures, so nothing privileged is
    // used.
    int alphabet=0,quota=0;double risk=0;bool strict_children=true;
    std::vector<int> parent,visits,observations,danger;
    std::vector<uint64_t> adv,ign;
    std::vector<std::vector<int>> tried,succ;
    struct Undo{int loser,winner,visits,observations,danger;uint64_t adv,ign;
                std::vector<int> tried,succ;};
    std::vector<Undo> log;

    void init(const Tree&T){
        int n=T.size();alphabet=T.alphabet;
        parent.resize(n);visits.resize(n);observations.resize(n);danger.resize(n);
        adv.assign(n,0);ign.assign(n,0);tried.assign(n,{});succ.assign(n,{});
        for(int i=0;i<n;i++){
            parent[i]=i;visits[i]=T.visits[i];observations[i]=T.observations(i);
            danger[i]=T.danger[i];
            adv[i]=0;ign[i]=0;tried[i].assign(alphabet,0);succ[i].assign(alphabet,-1);
            for(int e=0;e<alphabet;e++){
                if(T.advanced[i][e])adv[i]|=1ULL<<e;
                if(T.ignored[i][e])ign[i]|=1ULL<<e;
                tried[i][e]=T.tried(i,e);succ[i][e]=T.succ[i][e];
            }
        }
        log.clear();
    }
    int find(int x)const{while(parent[x]!=x)x=parent[x];return x;}
    bool conflict(int a,int b)const{return (adv[a]&ign[b])||(ign[a]&adv[b]);}
    int required_evidence(int a)const{
        double scaled=quota*(1.0+risk*danger[a]);
        return int(scaled+0.5);
    }
    bool quota_met(int a)const{
        int need=required_evidence(a);
        if(need<=0)return true;
        for(int e=0;e<alphabet;e++)if(tried[a][e]<need)return false;
        return true;
    }
    void unite(int a,int b){
        a=find(a);b=find(b);if(a==b)return;
        int w=std::min(a,b),l=std::max(a,b);
        log.push_back({l,w,visits[w],observations[w],danger[w],adv[w],ign[w],tried[w],succ[w]});
        parent[l]=w;adv[w]|=adv[l];ign[w]|=ign[l];
        visits[w]+=visits[l];observations[w]+=observations[l];danger[w]+=danger[l];
        for(int e=0;e<alphabet;e++){
            tried[w][e]+=tried[l][e];
            if(succ[w][e]<0)succ[w][e]=succ[l][e];
        }
    }
    void rollback(size_t mark){
        while(log.size()>mark){
            const Undo&u=log.back();
            parent[u.loser]=u.loser;
            visits[u.winner]=u.visits;observations[u.winner]=u.observations;
            danger[u.winner]=u.danger;
            adv[u.winner]=u.adv;ign[u.winner]=u.ign;
            tried[u.winner]=u.tried;succ[u.winner]=u.succ;
            log.pop_back();
        }
    }

    // Tentatively merge i and j and everything that follows from it. Returns
    // false and leaves the structure untouched if any implied pair conflicts or
    // fails the quota.
    bool attempt(int i,int j){
        size_t mark=log.size();
        std::vector<std::pair<int,int>> stack{{i,j}};
        while(!stack.empty()){
            auto[x,y]=stack.back();stack.pop_back();
            int a=find(x),b=find(y);
            if(a==b)continue;
            if(conflict(a,b)){rollback(mark);return false;}
            if(strict_children&&!(quota_met(a)&&quota_met(b))){rollback(mark);return false;}
            std::vector<std::pair<int,int>> children;
            for(int e=0;e<alphabet;e++){
                int sa=succ[a][e],sb=succ[b][e];
                if(sa>=0&&sb>=0)children.push_back({sa,sb});
            }
            unite(a,b);
            for(auto&ch:children)stack.push_back(ch);
        }
        return true;
    }

    // The same merge with a score attached: how many events the two sides
    // already agreed on. That count is what EDSM ranks candidates by. Returns
    // -1 and leaves nothing applied when the merge is inconsistent.
    int attempt_scored(int i,int j){
        size_t mark=log.size();
        std::vector<std::pair<int,int>> stack{{i,j}};int gained=0;
        while(!stack.empty()){
            auto[x,y]=stack.back();stack.pop_back();
            int a=find(x),b=find(y);
            if(a==b)continue;
            if(conflict(a,b)){rollback(mark);return -1;}
            if(strict_children&&!(quota_met(a)&&quota_met(b))){rollback(mark);return -1;}
            gained+=__builtin_popcountll((adv[a]&adv[b])|(ign[a]&ign[b]));
            std::vector<std::pair<int,int>> children;
            for(int e=0;e<alphabet;e++){
                int sa=succ[a][e],sb=succ[b][e];
                if(sa>=0&&sb>=0)children.push_back({sa,sb});
            }
            unite(a,b);
            for(auto&ch:children)stack.push_back(ch);
        }
        return gained;
    }
    // The standard passive learners, for the comparison the plan required. A red
    // core grows from the root; the blue fringe is everything one transition out
    // of it. RPNI takes the first consistent merge with a red state in index
    // order; EDSM takes the highest-scoring one. A blue state consistent with no
    // red state is promoted into the core. The consistency test, the closure and
    // the rollback are shared with the sweep rule above, so the only thing that
    // differs between the three is which candidates are considered and in what
    // order.
    int rebuild_fringe(const Tree&T,int theta,std::vector<int>&out,bool best_score){
        init(T);
        int n=T.size();
        std::vector<char> red(size_t(n),0);red[size_t(find(0))]=1;
        for(int guard=0;guard<4*n+64;guard++){
            std::vector<int> cores;
            for(int i=0;i<n;i++)if(red[size_t(i)]&&find(i)==i)cores.push_back(i);
            std::vector<int> blue;
            for(int r:cores)for(int e=0;e<alphabet;e++){
                int t=succ[r][e];if(t<0)continue;t=find(t);
                if(!red[size_t(t)]&&std::find(blue.begin(),blue.end(),t)==blue.end())
                    blue.push_back(t);
            }
            std::sort(blue.begin(),blue.end());
            bool changed=false;
            for(int b0:blue){
                int b=find(b0);
                if(red[size_t(b)])continue;
                if(visits[b]<theta||observations[b]<theta)continue;
                int pick=-1,pick_score=-1;
                for(int r0:cores){
                    int r=find(r0);if(r==b)continue;
                    size_t mark=log.size();
                    int sc=attempt_scored(r,b);
                    if(sc<0)continue;
                    rollback(mark);
                    if(sc>pick_score){pick_score=sc;pick=r;}
                    if(!best_score)break;
                }
                if(pick>=0){attempt(pick,b);red[size_t(find(pick))]=1;}
                else red[size_t(b)]=1;
                changed=true;break;
            }
            if(!changed)break;
        }
        out.assign(n,0);std::unordered_map<int,int> label;
        for(int i=0;i<n;i++){int r=find(i);auto it=label.find(r);
            if(it==label.end()){int v=int(label.size());label[r]=v;out[i]=v;}else out[i]=it->second;}
        return int(label.size());
    }

    int rebuild(const Tree&T,int theta,std::vector<int>&out){
        init(T);
        int n=T.size();
        for(int i=0;i<n;i++){
            if(T.visits[i]<theta||T.observations(i)<theta)continue;
            for(int j=i+1;j<n;j++){
                if(T.visits[j]<theta||T.observations(j)<theta)continue;
                int a=find(i),b=find(j);
                if(a==b)continue;
                if(!quota_met(a)||!quota_met(b))continue;
                if(conflict(a,b))continue;
                attempt(i,j);
            }
        }
        out.assign(n,0);std::unordered_map<int,int> label;
        for(int i=0;i<n;i++){int r=find(i);auto it=label.find(r);
            if(it==label.end()){int v=int(label.size());label[r]=v;out[i]=v;}else out[i]=it->second;}
        return int(label.size());
    }
};

struct Episode { std::vector<int> events; };

// Feature key from the successful-event prefix. The oracle arm folds the same
// prefix through the task machine and returns its Myhill-Nerode class.
struct Encoder {
    const Task&T;const Config&c;int mode;
    Encoder(const Task&t,const Config&c):T(t),c(c){
        mode = c.method=="count"?0:c.method=="window"?1:c.method=="bag"?2:
               c.method=="history"?3:c.method=="automaton"?4:
               (c.method=="merged"||c.method=="merged_explore")?5:
               c.method=="rawhistory"?6:c.method=="goal"?7:-1;
        if(mode<0)throw std::runtime_error("unknown method "+c.method);
    }
    // Node identity for the prefix tree. Identical to the history key for every
    // method that tracks a tree today, and stated separately so that a method
    // whose value table is indexed on something coarser still gets a real tree.
    uint64_t hist_key(const std::vector<int>&h,int upto)const{
        uint64_t k=1469598103934665603ULL;
        for(int i=0;i<upto;i++)hashadd(k,uint64_t(h[i]+1));
        return k;
    }
    // The feature key of a task class named directly rather than reached by
    // replaying a history. Counterfactual updates need this: they ask what a
    // transition would have done from a state the agent was never in.
    uint64_t class_key(int cls)const{
        uint64_t k=1469598103934665603ULL;hashadd(k,uint64_t(cls));return k;
    }
    uint64_t key(const std::vector<int>&h,int upto)const{
        uint64_t k=1469598103934665603ULL;
        if(mode==0){hashadd(k,uint64_t(upto));return k;}
        if(mode==1){
            for(int i=std::max(0,upto-c.window);i<upto;i++)hashadd(k,uint64_t(h[i]+1));
            hashadd(k,uint64_t(upto)+1000000);return k;
        }
        if(mode==2){
            std::vector<int> count(T.alphabet,0);
            for(int i=0;i<upto;i++)count[h[i]]++;
            for(int e=0;e<T.alphabet;e++)hashadd(k,uint64_t(count[e]));
            return k;
        }
        if(mode==3||mode==5||mode==6){for(int i=0;i<upto;i++)hashadd(k,uint64_t(h[i]+1));return k;}
        int q=0;for(int i=0;i<upto;i++)q=T.next_state(q,h[i]);
        if(mode==7){
            // Index by the goal the task now requires, not by the task state.
            int g=-1;
            for(int e=0;e<T.alphabet;e++)
                if(T.next_state(q,e)!=q&&!T.failing[T.next_state(q,e)]){g=e;break;}
            hashadd(k,uint64_t(g+1));return k;
        }
        hashadd(k,uint64_t(T.minimal[q]));return k;
    }
};

// The buffer keeps the physical transition plus indices into its episode's
// successful-event trace, never a computed feature. Features are derived at
// replay time under the current encoder, so a later learner that revises the
// representation can reinterpret the same rows without re-collecting them.
// What the agent may carry from one map to another. The tree records which
// event advanced the task after which history: a statement about the task, not
// about any floor plan. Cell-indexed tables, the learned map graph and the
// product distances are all map-specific and are never carried.
//
// The file is a plain table written by src/structure.py, not JSON, because a
// hand-rolled parser here was a source of bugs and carries no benefit.
// Layout: alphabet, node count, then per node visits, accepting, advanced[],
// ignored[], successor[]. Node keys are recomputed from the edges by this run's
// own encoder rather than trusted from the file.
int seed_tree(Tree&tree,const Task&T,const std::string&source,std::vector<int>&blocks){
    std::ifstream in(source);
    if(!in)throw std::runtime_error("cannot open structure "+source);
    int alphabet=0,n=0;in>>alphabet>>n;
    if(alphabet!=tree.alphabet)throw std::runtime_error("alphabet mismatch");
    std::vector<int> vis(n),acc(n);
    std::vector<std::vector<int>> adv(n),ign(n),succ(n);
    for(int i=0;i<n;i++){
        in>>vis[i]>>acc[i];
        adv[i].resize(alphabet);ign[i].resize(alphabet);succ[i].resize(alphabet);
        for(int e=0;e<alphabet;e++)in>>adv[i][e];
        for(int e=0;e<alphabet;e++)in>>ign[i][e];
        for(int e=0;e<alphabet;e++)in>>succ[i][e];
    }
    if(!in)throw std::runtime_error("truncated structure file");
    std::vector<std::vector<int>> route(n);
    std::vector<int> order{0};
    for(size_t k=0;k<order.size();k++){
        int i=order[k];
        for(int e=0;e<alphabet;e++){
            int child=succ[i][e];
            if(child<=0||child>=n||!route[child].empty())continue;
            route[child]=route[i];route[child].push_back(e);
            order.push_back(child);
        }
    }
    blocks.assign(1,0);
    std::vector<int> remap(n,-1);remap[0]=0;
    tree.visits[0]=vis[0];tree.accepting[0]=acc[0];
    tree.advanced[0]=adv[0];tree.ignored[0]=ign[0];
    // Diagnostics only: the true class of a seeded node has to be recomputed by
    // replaying its route through the task machine, or every seeded node claims
    // class zero and the merge scores are meaningless.
    auto truth_of=[&](const std::vector<int>&events){
        int q=0;for(int e:events)q=T.next_state(q,e);return q;};
    uint64_t base=1469598103934665603ULL;
    auto key_of=[&](const std::vector<int>&events){
        uint64_t k=base;for(int e:events)hashadd(k,uint64_t(e+1));return k;};
    for(size_t k=1;k<order.size();k++){
        int i=order[k];bool fresh=false;
        int id=tree.find_or_add(key_of(route[i]),truth_of(route[i]),fresh);
        remap[i]=id;
        if(fresh)blocks.push_back(int(blocks.size()));
        tree.visits[id]=vis[i];tree.accepting[id]=acc[i];tree.truth[id]=truth_of(route[i]);
        tree.advanced[id]=adv[i];tree.ignored[id]=ign[i];
    }
    for(size_t k=0;k<order.size();k++){
        int i=order[k];
        for(int e=0;e<alphabet;e++){
            int child=succ[i][e];
            if(child>=0&&child<n&&remap[child]>=0&&remap[i]>=0)
                tree.succ[remap[i]][e]=remap[child];
        }
    }
    int longest=0;
    for(size_t k=0;k<order.size();k++){int i=order[k];
        if(acc[i])longest=std::max(longest,int(route[i].size()));}
    return longest;
}

// `pre`/`post` index the episode's successful-event trace and `rpre`/`rpost`
// its trace of every observed event. Which one a run reads is the difference
// between a baseline that has been told non-progress events are irrelevant and
// one that has been told nothing.
struct Raw { int s,a,ns,ev,episode,pre,post,rpre,rpost;bool win,dead; };

struct Learner {
    const Task&T;Config c;Encoder enc;
    std::unordered_map<uint64_t,int> ids;std::vector<double> Q;
    std::vector<Raw> buffer;std::vector<std::vector<int>>*trace=nullptr,*rawtrace=nullptr;
    bool raw_history=false;
    long long updates=0,revision_updates=0;double reward_success;
    Tree tree;Partition part;std::vector<int> blocks;std::vector<double> G;
    long long skill_updates=0,structural_steps=0;int struct_target=-1,struct_left=0;
    // The automaton read as a goal provider rather than as a table index. The
    // agent then holds no task-indexed value function at all: it holds the
    // goal-conditioned skills every arm already trains, and the automaton
    // supplies which skill to run. The task dimension of the value table is the
    // event vocabulary, not the number of task states. Tracking the task state
    // is still required to name the goal, but that is a constant-memory fold
    // over the event stream rather than a dimension of the table.
    bool goal_mode=false;
    // Every event that advances the task from here without failing it. More
    // than one is live whenever the task leaves an order free.
    int allowed(int q,int*out)const{
        int k=0;
        for(int e=0;e<T.alphabet;e++){int n=T.next_state(q,e);
            if(n!=q&&!T.failing[n])out[k++]=e;}
        return k;
    }
    // With several goals live the automaton does not say which to take, so the
    // choice is the agent's. "first" fixes an arbitrary order and is the naive
    // reading; "value" asks the skills which goal is nearest from here, which
    // needs no oracle beyond the skills themselves.
    int select_goal(int q,int s,const std::vector<int>&hist,int node=-1)const{
        int cand[64];int k;
        if(c.goal_select=="learned"){
            // The goal named by the automaton the agent is building, not by the
            // true one. At a history it has advanced from before, the frontier
            // is whatever it saw advance there. At a history it knows nothing
            // about, everything not yet seen to be ignored is a candidate,
            // which is optimism in the face of an unbuilt automaton.
            k=0;
            if(node>=0&&node<int(tree.advanced.size())){
                // Among the events seen to advance here, prefer the one seen to
                // advance most often. A rarely used frontier event is one whose
                // skill has little data, and asking for an untrained skill
                // deadlocks the policy: the agent walks toward a goal it does
                // not know how to reach and never arrives.
                int bestn=0;
                for(int e=0;e<T.alphabet;e++)bestn=std::max(bestn,tree.advanced[node][e]);
                if(bestn>0)for(int e=0;e<T.alphabet;e++)
                    if(tree.advanced[node][e]==bestn)cand[k++]=e;
                if(k==0)for(int e=0;e<T.alphabet;e++)if(tree.ignored[node][e]==0)cand[k++]=e;
            }
            if(k==0)for(int e=0;e<T.alphabet;e++)cand[k++]=e;
        }
        else
        if(c.goal_select=="untried"){
            // The control that has eaten several of this project's claims: name
            // the goal without consulting the automaton at all. An event this
            // episode has not yet collected is a goal any agent can name from
            // the progress feedback it already receives.
            k=0;
            for(int e=0;e<T.alphabet;e++){
                bool had=false;for(int seen:hist)if(seen==e){had=true;break;}
                if(!had)cand[k++]=e;
            }
            if(k==0)for(int e=0;e<T.alphabet;e++)cand[k++]=e;
        }else k=allowed(q,cand);
        if(k==0)return -1;
        if(k==1||c.goal_select=="first")return cand[0];
        int bg=cand[0];double bv=-1e300;
        for(int i=0;i<k;i++){
            double v=*std::max_element(G.begin()+(size_t(cand[i])*T.N+s)*4,
                                       G.begin()+(size_t(cand[i])*T.N+s)*4+4);
            if(v>bv){bv=v;bg=cand[i];}
        }
        return bg;
    }
    void skill_replay(const Raw&x){
        if(c.relabel){train_skills(x.s,x.a,x.ns,x.ev);return;}
        // Without relabelling the replayed row retrains only the goal that was
        // actually being pursued when it was collected, recovered by folding its
        // episode's own event prefix through the machine.
        const std::vector<int>&h=(*trace)[x.episode];
        int q=0;for(int i=0;i<x.pre&&i<int(h.size());i++)q=T.next_state(q,h[i]);
        int g=select_goal(q,x.s,h,-1);
        if(g>=0)train_one_skill(g,x.s,x.a,x.ns,x.ev);
    }
    std::vector<int> accept_dist;int deepest=0,solved_length=0;
    // Two hypotheses, deliberately different. The one that drives the value
    // table is conservative: it must not merge without evidence. The one that
    // drives which experiment to run next is optimistic: it merges on the
    // slightest agreement precisely so that it proposes equivalences worth
    // testing. Asking about a merge you already refuse to make teaches nothing.
    // Distance to a discovered accepting node, over discovered edges only. This
    // is what a task-graph potential needs and a progress-depth potential
    // cannot express: with branch tails of different lengths, two histories at
    // the same depth owe different numbers of transitions. Nodes with no known
    // route to an accepting node fall back to the depth potential, so the two
    // arms differ only where the learner actually knows something.
    std::vector<int> graph_dist;
    // Steps still owed from (task node, cell), estimated by counting backwards
    // along episodes that finished. Model-free and privileged in nothing: it
    // uses only the cells the agent stood on and the successes it achieved. The
    // oracle product table is the ceiling this is aiming at.
    std::vector<int> learned_product;
    // Computed, not learned. Every step reveals one map edge for free, and the
    // prefix tree records the task edges the agent has confirmed. Backward
    // reachability over the product of the two gives steps-to-target without
    // ever having finished the task, which is what counting backwards from
    // successes cannot do. The target is an accepting node once one is known
    // and the deepest node discovered so far before that, so the potential is
    // optimistic about the frontier exactly as the depth potential is.
    std::vector<int> map_next,map_event,product_dist;
    // Distance over the learned map from every cell to the nearest cell bearing
    // each label. One backward pass per label, recomputed when the map changes.
    std::vector<int> label_dist;
    void refresh_labels(){
        label_dist.assign(size_t(T.alphabet)*T.N,-1);
        for(int e=0;e<T.alphabet;e++){
            std::vector<int> frontier;
            for(int s=0;s<T.N;s++)for(int a=0;a<4;a++)
                if(map_event[size_t(s)*4+a]==e&&map_next[size_t(s)*4+a]>=0){
                    size_t idx=size_t(e)*T.N+map_next[size_t(s)*4+a];
                    if(label_dist[idx]<0){label_dist[idx]=0;frontier.push_back(map_next[size_t(s)*4+a]);}
                }
            size_t head=0;
            while(head<frontier.size()){
                int cell=frontier[head++];
                int d=label_dist[size_t(e)*T.N+cell];
                for(int s=0;s<T.N;s++)for(int a=0;a<4;a++){
                    if(map_next[size_t(s)*4+a]!=cell)continue;
                    size_t idx=size_t(e)*T.N+s;
                    if(label_dist[idx]<0){label_dist[idx]=d+1;frontier.push_back(s);}
                }
            }
        }
    }
    void learn_edge(int cell,int action,int next,int event){
        size_t i=size_t(cell)*4+action;
        map_next[i]=next;map_event[i]=event;
    }
    void refresh_product(){
        int nodes=tree.size();
        size_t total=size_t(nodes)*T.N;
        product_dist.assign(total,-1);
        std::vector<size_t> frontier;
        int best_depth=-1;
        for(int i=0;i<nodes;i++)if(tree.accepting[i])best_depth=1<<30;
        if(best_depth<0)for(int i=0;i<nodes;i++)best_depth=std::max(best_depth,depth_of(i));
        for(int i=0;i<nodes;i++){
            bool target=tree.accepting[i]||(best_depth<(1<<30)&&depth_of(i)==best_depth);
            if(!target)continue;
            for(int s=0;s<T.N;s++){product_dist[size_t(i)*T.N+s]=0;frontier.push_back(size_t(i)*T.N+s);}
        }
        // Backward pass: which (node, cell) can reach a target in d steps.
        std::vector<std::vector<size_t>> back(total);
        for(int i=0;i<nodes;i++)for(int s=0;s<T.N;s++)for(int a=0;a<4;a++){
            size_t e=size_t(s)*4+a;
            if(map_next[e]<0)continue;
            int ev=map_event[e],dest=i;
            if(ev>=0&&tree.succ[i][ev]>=0)dest=tree.succ[i][ev];
            back[size_t(dest)*T.N+map_next[e]].push_back(size_t(i)*T.N+s);
        }
        size_t head=0;
        while(head<frontier.size()){
            size_t z=frontier[head++];
            for(size_t prev:back[z])
                if(product_dist[prev]<0){product_dist[prev]=product_dist[z]+1;frontier.push_back(prev);}
        }
    }
    int depth_of(int node)const{
        auto it=node_depth.find(node);return it==node_depth.end()?0:it->second;
    }
    std::unordered_map<int,int> node_depth;
    std::vector<std::pair<int,int>> episode_path;      // (node, cell) this episode
    std::vector<int> node_path;                        // tree nodes this episode
    long long product_entries=0,total_steps=0,total_events=0;
    // The learned table is in environment steps; the fallback is a count of
    // events still owed. Mixing the two at one scale puts a cliff at the edge of
    // what has been learned. Converting with the agent's own measured steps per
    // progress event keeps the potential in one unit throughout.
    double steps_per_event()const{
        return total_events>0?double(total_steps)/double(total_events):1.0;
    }
    void note_step(int node,int cell){episode_path.push_back({node,cell});}
    void credit_success(){
        int len=int(episode_path.size());
        for(int t=0;t<len;t++){
            auto[node,cell]=episode_path[size_t(t)];
            if(node<0)continue;
            size_t idx=size_t(node)*T.N+cell;
            if(idx>=learned_product.size())learned_product.resize(idx+T.N,-1);
            int togo=len-t;
            if(learned_product[idx]<0||togo<learned_product[idx]){
                if(learned_product[idx]<0)product_entries++;
                learned_product[idx]=togo;
            }
        }
    }
    void refresh_graph(){
        int n=tree.size();
        graph_dist.assign(n,-1);
        std::vector<int> frontier;
        for(int i=0;i<n;i++)if(tree.accepting[i]){graph_dist[i]=0;frontier.push_back(i);}
        while(!frontier.empty()){
            std::vector<int> next;
            for(int i=0;i<n;i++){
                if(graph_dist[i]>=0)continue;
                int best=-1;
                for(int e=0;e<T.alphabet;e++){
                    int z=tree.succ[i][e];
                    if(z>=0&&graph_dist[z]>=0&&(best<0||graph_dist[z]+1<best))best=graph_dist[z]+1;
                }
                if(best>=0){graph_dist[i]=best;next.push_back(i);}
            }
            if(next.empty())break;
            frontier=next;
        }
    }

    Partition guess;std::vector<int> hyp;
    std::vector<uint64_t> block_advanced,block_ignored;
    void refresh_hypothesis(){
        guess.quota=0;
        guess.rebuild(tree,c.theta,hyp);
        int k=0;for(int b:hyp)k=std::max(k,b+1);
        block_advanced.assign(size_t(k),0);block_ignored.assign(size_t(k),0);
        for(int i=0;i<tree.size();i++)
            for(int e=0;e<T.alphabet;e++){
                if(tree.advanced[i][e])block_advanced[hyp[i]]|=1ULL<<e;
                if(tree.ignored[i][e])block_ignored[hyp[i]]|=1ULL<<e;
            }
    }
    // The experiment that would settle the most doubtful merge available here:
    // an event whose outcome is already known somewhere in this node's
    // hypothesised class but has never been tried at this node. Running it
    // either confirms the class or splits it.
    int separating_experiment(int node)const{
        if(node>=int(hyp.size()))return -1;
        uint64_t known=block_advanced[hyp[node]]|block_ignored[hyp[node]];
        uint64_t positive=block_advanced[hyp[node]];
        int fallback=-1;
        for(int e=0;e<T.alphabet;e++){
            if(tree.tried(node,e))continue;
            if(positive>>e&1)return e;
            if((known>>e&1)&&fallback<0)fallback=e;
        }
        return fallback;
    }
    int classes=0,revisions=0,first_merge=-1;bool merging=false,tracking=false,dirty=true;
    Learner(const Task&t,const Config&c):T(t),c(c),enc(t,c),Q(size_t(MAXF)*t.N*4,0.0),
        reward_success(c.reward_ratio*c.cost*t.maxlen),tree(t.alphabet){
        raw_history=c.method=="rawhistory";
        goal_mode=c.method=="goal";
        merging=c.method=="merged"||c.method=="merged_explore";
        // The prefix tree is also maintained without merging, so an arm can buy
        // structural evidence and still index its table by full history. That
        // control separates "the detours improved exploration" from "the merges
        // improved the representation".
        tracking=merging||c.beta>0||c.shape=="learned"||c.shape=="learnedproduct"
                 ||c.shape=="computedproduct"||c.shape=="guided"
                 ||c.goal_select=="learned";
        if(c.crm){
            if(goal_mode)throw std::runtime_error("--crm is a rule for the index reading; "
                "the goal reading's counterpart is --relabel");
            if(c.shape!="none")throw std::runtime_error("--crm with shaping is not implemented: "
                "a counterfactual state has no history to evaluate a potential on");
            if(!merging)build_crm_reps();
        }
        if(c.shape=="learnedproduct")learned_product.assign(size_t(t.N)*64,-1);
        if(c.shape=="computedproduct"||c.shape=="guided"){
            map_next.assign(size_t(t.N)*4,-1);map_event.assign(size_t(t.N)*4,-1);
        }
        part.quota=c.quota;part.strict_children=c.strict_children;
        part.risk=c.risk;guess.risk=c.risk;
        guess.strict_children=c.strict_children;
        G.assign(size_t(t.alphabet)*t.N*4,0.0);
        if(c.shape=="oracle")accept_dist=t.to_accept();
        blocks.assign(1,0);classes=1;
    }
    // The table is finite. A representation that keeps producing new states
    // eventually fills it, and everything after that collides into one row.
    // That is what happens to any bounded learner facing an unbounded feature
    // space, so it is reported rather than treated as an error. The map is
    // allowed to keep growing so the number of distinct states can be counted.
    long long overflow_hits=0;
    int id(uint64_t key){auto it=ids.find(key);if(it!=ids.end())return it->second;
        int v=int(ids.size());ids[key]=v;
        if(v>=MAXF){overflow_hits++;return MAXF-1;}
        return v;}
    int feature(const std::vector<int>&h,int upto){
        if(!merging)return id(enc.key(h,upto));
        auto it=tree.ids.find(enc.hist_key(h,upto));
        return it==tree.ids.end()?0:blocks[it->second];
    }
    // Read-only lookup for frozen-policy evaluation: an unseen feature must not
    // be allocated, and its values are the zero-initialised ones.
    int lookup(const std::vector<int>&h,int upto)const{
        if(merging){auto it=tree.ids.find(enc.hist_key(h,upto));
            return it==tree.ids.end()?-1:blocks[it->second];}
        auto it=ids.find(enc.key(h,upto));
        if(it==ids.end())return -1;
        return it->second>=MAXF?MAXF-1:it->second;}

    // Repartition, then carry values across the relabelling under the chosen
    // policy. Reset keeps nothing, transfer copies each new class from one of
    // its members' old classes, replay rebuilds from the raw buffer. Replay's
    // extra Bellman updates are reported separately so the per-step budget
    // stays identical across arms.
    void repartition(int step){
        std::vector<int> fresh;
        int n=c.merge_rule=="rpni"?part.rebuild_fringe(tree,c.theta,fresh,false)
             :c.merge_rule=="edsm"?part.rebuild_fringe(tree,c.theta,fresh,true)
             :part.rebuild(tree,c.theta,fresh);
        if(fresh==blocks&&n==classes){dirty=false;return;}
        if(first_merge<0&&n<tree.size())first_merge=step;
        // A revision is a class being torn apart, not a node being added: two
        // nodes that used to share a class no longer do.
        bool split=false;
        for(size_t i=0;i<blocks.size()&&!split;i++)
            for(size_t j=i+1;j<blocks.size();j++)
                if(blocks[i]==blocks[j]&&fresh[i]!=fresh[j]){split=true;break;}
        if(split)revisions++;
        std::vector<double> old;old.swap(Q);Q.assign(size_t(MAXF)*T.N*4,0.0);
        if(c.revision=="jirp"){
            // JIRP transfers values between hypothesis states that are
            // equivalent with respect to future behaviour, not between a new
            // class and whichever old class one of its members happened to sit
            // in. The signature below is that equivalence approximated by three
            // rounds of refinement: a class is described by whether it accepts
            // and by the descriptions of what each event leads to.
            auto signature=[&](const std::vector<int>&part,int ncls){
                std::vector<std::vector<int>> cs(size_t(ncls),
                    std::vector<int>(size_t(tree.alphabet),-1));
                std::vector<char> acc(size_t(ncls),0);
                std::vector<uint64_t> am(size_t(ncls),0),im(size_t(ncls),0);
                for(int nd=0;nd<int(part.size());nd++){
                    int b=part[size_t(nd)];if(b<0||b>=ncls)continue;
                    if(nd<int(tree.accepting.size())&&tree.accepting[size_t(nd)])acc[size_t(b)]=1;
                    for(int e=0;e<tree.alphabet;e++){
                        if(tree.advanced[size_t(nd)][size_t(e)])am[size_t(b)]|=1ULL<<e;
                        if(tree.ignored[size_t(nd)][size_t(e)])im[size_t(b)]|=1ULL<<e;
                        int t=tree.succ[size_t(nd)][size_t(e)];
                        if(t>=0&&t<int(part.size())&&cs[size_t(b)][size_t(e)]<0)
                            cs[size_t(b)][size_t(e)]=part[size_t(t)];
                    }
                }
                std::vector<uint64_t> h(size_t(ncls),0ULL);
                // The observable behaviour of the class itself, before any
                // refinement: whether it accepts, and which events it has been
                // seen to advance on and to ignore.
                for(int b=0;b<ncls;b++){
                    uint64_t k=1469598103934665603ULL;
                    hashadd(k,acc[size_t(b)]?1ULL:0ULL);
                    hashadd(k,am[size_t(b)]);hashadd(k,im[size_t(b)]);
                    h[size_t(b)]=k;
                }
                for(int it=0;it<3;it++){
                    std::vector<uint64_t> nh(size_t(ncls),0ULL);
                    for(int b=0;b<ncls;b++){
                        uint64_t k=1469598103934665603ULL;hashadd(k,h[size_t(b)]);
                        for(int e=0;e<tree.alphabet;e++){
                            int t=cs[size_t(b)][size_t(e)];
                            hashadd(k,t<0?0xFFFFFFFFULL:h[size_t(t)]);
                        }
                        nh[size_t(b)]=k;
                    }
                    h.swap(nh);
                }
                return h;
            };
            std::vector<int> oldpart(blocks.begin(),blocks.end());
            std::vector<uint64_t> so=signature(oldpart,classes),sn=signature(fresh,n);
            std::unordered_map<uint64_t,int> once;
            for(int b=0;b<classes;b++){
                auto it=once.find(so[size_t(b)]);
                if(it==once.end())once[so[size_t(b)]]=b;else it->second=-1;
            }
            for(int b=0;b<n;b++){
                auto it=once.find(sn[size_t(b)]);
                if(it==once.end()||it->second<0)continue;
                int src=it->second;
                if(src<MAXF&&b<MAXF)
                    std::copy(old.begin()+size_t(src)*T.N*4,old.begin()+size_t(src+1)*T.N*4,
                              Q.begin()+size_t(b)*T.N*4);
            }
        }
        if(c.revision=="transfer"){
            std::vector<int> source(size_t(n),-1);
            for(int i=int(fresh.size())-1;i>=0;i--)if(i<int(blocks.size()))source[fresh[i]]=blocks[i];
            for(int b=0;b<n;b++)if(source[b]>=0&&source[b]<MAXF)
                std::copy(old.begin()+size_t(source[b])*T.N*4,old.begin()+size_t(source[b]+1)*T.N*4,
                          Q.begin()+size_t(b)*T.N*4);
        }
        blocks=fresh;classes=n;dirty=false;
        if(c.revision=="replay"&&!buffer.empty()){
            long long before=updates;
            for(int j=0;j<c.replay_budget;j++)
                apply(buffer[mix(uint64_t(step)*7919+j)%buffer.size()]);
            revision_updates+=updates-before;updates=before;
        }
    }
    double q(int f,int s,int a)const{
        if(f<0)return 0.0;
        if(f>=MAXF)f=MAXF-1;
        return Q[(size_t(f)*T.N+s)*4+a];}
    double best(int f,int s)const{double b=q(f,s,0);for(int a=1;a<4;a++)b=std::max(b,q(f,s,a));return b;}
    int greedy(int f,int s)const{int a=0;for(int b=1;b<4;b++)if(q(f,s,b)>q(f,s,a))a=b;return a;}
    // Goal-conditioned navigation, relabelled from every physical transition.
    // Every arm trains these so the update budget is identical; only the
    // structural-exploration arm is allowed to act on them.
    void train_one_skill(int g,int s,int a,int ns,int event){
        bool hit=event==g;
        size_t base=(size_t(g)*T.N+ns)*4;
        double best_next=*std::max_element(G.begin()+base,G.begin()+base+4);
        double target=double(hit)-c.cost+(hit?0:c.gamma*best_next);
        double&v=G[(size_t(g)*T.N+s)*4+a];v+=c.alpha*(target-v);skill_updates++;
    }
    void train_skills(int s,int a,int ns,int event){
        for(int g=0;g<T.alphabet;g++){
            bool hit=event==g;
            size_t base=(size_t(g)*T.N+ns)*4;
            double best_next=*std::max_element(G.begin()+base,G.begin()+base+4);
            double target=double(hit)-c.cost+(hit?0:c.gamma*best_next);
            double&v=G[(size_t(g)*T.N+s)*4+a];v+=c.alpha*(target-v);skill_updates++;
        }
    }
    int skill_action(int g,int s)const{
        int a=0;for(int b=1;b<4;b++)if(G[(size_t(g)*T.N+s)*4+b]>G[(size_t(g)*T.N+s)*4+a])a=b;return a;
    }

    // Potential-based shaping. The base reward never changes; the potential is
    // recomputed at replay time from the current structure, so a shaped reward
    // is never frozen into the buffer.
    //
    // Three named potentials, because the first version conflated them. `count`
    // is worth nothing until the task has been solved once, since only then is
    // its horizon known; that is exactly what the original code did, and naming
    // it stops it being read as evidence about partial structure. `frontier`
    // adds the optimistic pre-success term the plan actually called for: before
    // any success the deepest history seen stands in for the horizon, so going
    // deeper than usual is rewarded. In this task family every accepted history
    // has the same length, so depth is graph distance and `frontier` is not a
    // test of a general graph-distance potential. `oracle` reads the true
    // distance to acceptance and is a reference, not a method.
    // Under discounting, potential-based shaping is not invariant to the
    // potential's scale or offset: the term for a step that changes nothing is
    // (gamma-1)*Phi, a per-step bonus proportional to how far the state is from
    // acceptance. Two potentials that disagree about total distance therefore
    // deliver different amounts of that bonus even at the same nominal scale.
    // Normalising each potential by its own value at the start makes every arm
    // span the same range, which is the comparison that isolates *what the
    // potential knows* from *how big its numbers are*.
    // `product` is the privileged reference the project's earlier work points
    // at: distance in the product of task machine and map, which is environment
    // steps still owed rather than task transitions still owed. It needs the
    // current cell, so it is the only potential here that is not a function of
    // the event history alone.
    double raw_potential(const std::vector<int>&h,int upto,int cell)const{
        if(c.shape=="none")return 0;
        if(c.shape=="product"||c.shape=="wrongproduct"||c.shape=="taskstep"||c.shape=="shuffled"){
            int q=0;for(int i=0;i<upto;i++)q=T.next_state(q,h[i]);
            if(c.shape=="taskstep")return -c.shape_scale*T.task_marginal[q];
            const std::vector<int>&tab=c.shape=="product"?T.product:
                                       c.shape=="shuffled"?T.shuffled:T.wrong_product;
            return -c.shape_scale*double(tab[size_t(q)*T.N+cell]);
        }
        if(c.shape=="mapdist")return -c.shape_scale*T.map_marginal[cell];
        // "guided": the spatial half points at an event this task state has not
        // yet tried, and the depth half keeps an optimistic margin past the
        // deepest history seen. Neither half alone worked: a potential aimed at
        // the frontier makes the frontier an attractor, and a depth potential
        // has no spatial gradient at all. The potential changes as evidence
        // arrives, so it is recomputed at replay time like the others.
        if(c.shape=="guided"){
            // Two phases, because the spatial half is answering two
            // different questions. Before the task has ever been finished it
            // points at an event this state has not tried, which is
            // exploration. Afterwards it points at an event already known to
            // advance the task from here, which is execution. Leaving it on
            // exploration after the first success pulls the agent off the route
            // it has just learned, and that is what the first version did.
            // The spatial half is switched off once the task has been finished
            // once. Pointing it at the next required event instead was tried and
            // did not help: its target jumps whenever a node or a tried-set
            // changes, and those jumps inject shaping spikes that cost more than
            // the guidance is worth. Before the first success there is nothing
            // to disturb, and that is where it pays.
            auto it=tree.ids.find(enc.hist_key(h,upto));
            double spatial=0;
            if(!solved_length&&it!=tree.ids.end()&&!label_dist.empty()){
                int node=it->second,best=-1;
                for(int e=0;e<T.alphabet;e++){
                    if(tree.tried(node,e))continue;
                    int d=label_dist[size_t(e)*T.N+cell];
                    if(d>=0&&(best<0||d<best))best=d;
                }
                if(best>0)spatial=best;
            }
            // The two halves are in the same unit but not on the same natural
            // scale, and sharing one multiplier let the spatial half's larger
            // range hold the depth half far below its own optimum. They are
            // scaled separately; a negative spatial scale means "same as the
            // depth half", which recovers the single-scale version.
            // No unit conversion between the halves. Converting the depth term
            // into steps needed an average steps-per-event, which is wildly
            // unstable early on — a few hundred steps before the first event
            // makes it enormous and the potential explodes. The depth half is
            // left in events and the spatial half in steps, and the two scales
            // absorb the difference.
            int horizon=solved_length?solved_length:deepest+1;
            double remaining=double(std::max(0,horizon-upto));
            double sscale=c.spatial_scale>=0?c.spatial_scale:c.shape_scale;
            return -(c.shape_scale*remaining+sscale*spatial);
        }
        if(c.shape=="computedproduct"){
            auto it=tree.ids.find(enc.hist_key(h,upto));
            if(it!=tree.ids.end()){
                size_t idx=size_t(it->second)*T.N+cell;
                if(idx<product_dist.size()&&product_dist[idx]>=0)
                    return -c.shape_scale*double(product_dist[idx]);
            }
            int horizon=solved_length?solved_length:std::max(deepest,upto);
            return -c.shape_scale*double(std::max(0,horizon-upto))*steps_per_event();
        }
        if(c.shape=="learnedproduct"){
            auto it=tree.ids.find(enc.hist_key(h,upto));
            if(it!=tree.ids.end()){
                size_t idx=size_t(it->second)*T.N+cell;
                if(idx<learned_product.size()&&learned_product[idx]>=0)
                    return -c.shape_scale*double(learned_product[idx]);
            }
            int horizon=solved_length?solved_length:std::max(deepest,upto);
            return -c.shape_scale*double(std::max(0,horizon-upto))*steps_per_event();
        }
        if(c.shape=="oracle"){
            int q=0;for(int i=0;i<upto;i++)q=T.next_state(q,h[i]);
            return -c.shape_scale*accept_dist[q];
        }
        int horizon;
        if(solved_length)horizon=solved_length;
        else if(c.shape=="frontier"||c.shape=="learned")horizon=std::max(deepest,upto);
        else horizon=upto;                 // "count": no signal before a success
        double depth_potential=-c.shape_scale*double(std::max(0,horizon-upto));
        if(c.shape!="learned")return depth_potential;
        auto it=tree.ids.find(enc.hist_key(h,upto));
        if(it==tree.ids.end()||it->second>=int(graph_dist.size()))return depth_potential;
        int dist=graph_dist[it->second];
        return dist<0?depth_potential:-c.shape_scale*double(dist);
    }
    double potential(const std::vector<int>&h,int upto,int cell)const{
        double v=raw_potential(h,upto,cell);
        if(!c.shape_normalize||c.shape=="none")return v;
        static thread_local std::vector<int> empty;
        double span=-raw_potential(empty,0,cell);
        return span>1e-12?v*(c.shape_scale/span):v;
    }

    // One representative task state per class of the machine the agent is
    // given. Built once; the counterfactual update walks it every step.
    std::vector<int> crm_reps;
    void build_crm_reps(){
        std::vector<char> seen(size_t(T.nstates),0);
        for(int q=0;q<T.nstates;q++){int cls=T.minimal[q];
            if(cls>=0&&cls<T.nstates&&!seen[size_t(cls)]){seen[size_t(cls)]=1;crm_reps.push_back(q);}}
    }
    // Counterfactual experience for reward machines: one physical transition is
    // evidence about every task state at once, because the machine says what
    // that same transition would have done had the task been elsewhere. This is
    // the update rule the goal reading gets for free across goals, brought back
    // to the reading that indexes the value table by task state.
    void apply_counterfactual(const Raw&x){
        int ev=x.ev;
        if(!merging){
            for(int u:crm_reps){
                int nu=T.next_state(u,ev);
                bool win=T.accepting[nu]!=0,dead=T.failing[nu]!=0;
                int fi=id(enc.class_key(T.minimal[u])),fj=id(enc.class_key(T.minimal[nu]));
                double r=win?reward_success:-c.cost;
                double target=r+((win||dead)?0:c.gamma*best(fj,x.ns));
                if(fi>=MAXF)fi=MAXF-1;
                double&v=Q[(size_t(fi)*T.N+x.s)*4+x.a];v+=c.alpha*(target-v);updates++;
            }
            return;
        }
        // The learned machine can only supply counterfactual evidence where it
        // has some: a class whose representative history has never seen this
        // event has no opinion about it and is skipped. That gap is the honest
        // difference between running this rule on a known machine and on one
        // being inferred alongside the policy.
        std::vector<int> rep(size_t(classes),-1);
        for(int nnode=0;nnode<int(blocks.size());nnode++){
            int b=blocks[nnode];
            if(b>=0&&b<classes&&rep[size_t(b)]<0)rep[size_t(b)]=nnode;
        }
        for(int b=0;b<classes;b++){
            int u=rep[size_t(b)];if(u<0)continue;
            if(ev<0){
                double target=-c.cost+c.gamma*best(b,x.ns);
                double&v=Q[(size_t(b)*T.N+x.s)*4+x.a];v+=c.alpha*(target-v);updates++;continue;
            }
            int fj=-1;bool win=false;
            if(tree.advanced[u][ev]>0&&tree.succ[u][ev]>=0){
                int child=tree.succ[u][ev];
                fj=blocks[size_t(child)];win=tree.accepting[size_t(child)]>0;
            }else if(tree.ignored[u][ev]>0){fj=b;}
            else continue;
            double r=win?reward_success:-c.cost;
            double target=r+(win?0:c.gamma*best(fj,x.ns));
            double&v=Q[(size_t(b)*T.N+x.s)*4+x.a];v+=c.alpha*(target-v);updates++;
        }
    }
    void apply(const Raw&x){
        if(goal_mode)return;
        const std::vector<int>&h=raw_history?(*rawtrace)[x.episode]:(*trace)[x.episode];
        int pre=raw_history?x.rpre:x.pre, post=raw_history?x.rpost:x.post;
        int fi=feature(h,pre),fj=feature(h,post);
        double r=x.win?reward_success:-c.cost;
        // A failed commit ends the episode with nothing further to collect, so
        // it bootstraps from zero exactly as a success does, and the potential
        // at an absorbing state is zero for both.
        bool terminal=x.win||x.dead;
        const std::vector<int>&ph=(*trace)[x.episode];
        if(c.shape!="none")r+=(terminal?0.0:c.gamma*potential(ph,x.post,x.ns))-potential(ph,x.pre,x.s);
        double target=r+(terminal?0:c.gamma*best(fj,x.ns));
        if(fi>=MAXF)fi=MAXF-1;
        double&v=Q[(size_t(fi)*T.N+x.s)*4+x.a];v+=c.alpha*(target-v);updates++;
    }
};

int main(int argc,char**argv){
    Config c;
    for(int i=1;i<argc;i+=2){
        std::string k=argv[i],v=i+1<argc?argv[i+1]:"";
        if(k=="--task")c.task=v;else if(k=="--method")c.method=v;
        else if(k=="--seed")c.seed=std::stoi(v);else if(k=="--budget")c.budget=std::stoi(v);
        else if(k=="--every")c.every=std::stoi(v);else if(k=="--window")c.window=std::stoi(v);
        else if(k=="--epsilon")c.epsilon=std::stod(v);else if(k=="--gamma")c.gamma=std::stod(v);
        else if(k=="--alpha")c.alpha=std::stod(v);else if(k=="--cost")c.cost=std::stod(v);
        else if(k=="--reward-ratio")c.reward_ratio=std::stod(v);
        else if(k=="--theta")c.theta=std::stoi(v);
        else if(k=="--revision")c.revision=v;
        else if(k=="--repartition")c.repartition=std::stoi(v);
        else if(k=="--replay-budget")c.replay_budget=std::stoi(v);
        else if(k=="--quota")c.quota=std::stoi(v);
        else if(k=="--beta")c.beta=std::stod(v);
        else if(k=="--struct-cap")c.struct_cap=std::stoi(v);
        else if(k=="--target")c.target=v;
        else if(k=="--shape")c.shape=v;
        else if(k=="--shape-scale")c.shape_scale=std::stod(v);
        else if(k=="--strict-children")c.strict_children=std::stoi(v)!=0;
        else if(k=="--dump-tree")c.dump_tree=std::stoi(v)!=0;
        else if(k=="--shape-normalize")c.shape_normalize=std::stoi(v)!=0;
        else if(k=="--seed-structure")c.seed_structure=v;
        else if(k=="--doom-continues")c.doom_continues=std::stoi(v)!=0;
        else if(k=="--risk")c.risk=std::stod(v);
        else if(k=="--spatial-scale")c.spatial_scale=std::stod(v);
        else if(k=="--goal-select")c.goal_select=v;
        else if(k=="--crm")c.crm=std::stoi(v)!=0;
        else if(k=="--relabel")c.relabel=std::stoi(v)!=0;
        else if(k=="--replay")c.replay=std::stoi(v);
        else if(k=="--merge-rule")c.merge_rule=v;
        else throw std::runtime_error("unknown flag "+k);
    }
    Task T;T.load(c.task);
    Learner L(T,c);
    if(!c.seed_structure.empty()){
        L.tracking=true;
        int longest=seed_tree(L.tree,T,c.seed_structure,L.blocks);
        L.dirty=true;
        if(longest>0){L.solved_length=longest;L.deepest=longest;}
    }
    const int S=int(T.starts.size());
    int node=0;
    std::vector<std::vector<int>> episodes;episodes.emplace_back();
    std::vector<std::vector<int>> raw_episodes;raw_episodes.emplace_back();
    L.trace=&episodes;L.rawtrace=&raw_episodes;
    int s=T.starts[0],qstate=0,t=0,episode=0,which=0;
    uint64_t stream=mix(uint64_t(c.seed)*2654435761u+7);
    uint64_t trace=1469598103934665603ULL;
    std::vector<std::array<double,3>> checks;

    auto reset=[&](int index){which=index%S;s=T.starts[which];qstate=0;t=0;node=0;
        L.episode_path.clear();L.node_path.assign(1,0);
        episodes.emplace_back();raw_episodes.emplace_back();
        episode=int(episodes.size())-1;};

    auto rollout=[&](int start)->int{
        int cs=T.starts[start],cq=0;std::vector<int> h;
        std::vector<int> rh;int cn=0;
        for(int step=0;step<T.horizon;step++){
            int a;
            if(L.goal_mode){
                int g=L.select_goal(cq,cs,h,cn);
                a=g<0?0:L.skill_action(g,cs);
            }else{
                int f=L.raw_history?L.lookup(rh,int(rh.size())):L.lookup(h,int(h.size()));
                a=L.greedy(f,cs);
            }
            int ns=T.moves[size_t(cs)*4+a],ev=T.events[size_t(cs)*4+a];
            int nq=T.next_state(cq,ev);
            if(nq!=cq&&!T.failing[nq]&&L.tracking&&cn>=0&&cn<int(L.tree.succ.size()))
                cn=L.tree.succ[cn][ev];
            if(nq!=cq)h.push_back(ev);
            if(ev>=0)rh.push_back(ev);
            cs=ns;cq=nq;
            if(T.failing[cq])return -1;
            if(T.accepting[cq])return step+1;
        }
        return -1;
    };

    int first90=-1,stable90=-1,hits=0,last_repartition=0,first_success=-1,first_failure=-1;
    long long failures=0,doomed_steps=0;double auc=0,last=0;
    for(int step=1;step<=c.budget;step++){
        std::vector<int>&hist=episodes[episode];
        std::vector<int>&rawhist=raw_episodes[episode];
        int pre=int(hist.size()),rpre=int(rawhist.size());
        if(L.tracking){
            if(++L.tree.visits[node]==c.theta)L.dirty=true;
            if(L.dirty&&step-last_repartition>=c.repartition){
                if(c.target=="separating")L.refresh_hypothesis();
                if(c.shape=="learned")L.refresh_graph();
                if(c.shape=="computedproduct")L.refresh_product();
                if(c.shape=="guided")L.refresh_labels();
                if(L.merging)L.repartition(step);else L.dirty=false;
                last_repartition=step;
            }
        }
        if(c.shape=="learnedproduct"){L.note_step(node,s);L.total_steps++;}
        if(c.shape=="computedproduct"||c.shape=="guided")L.total_steps++;
        int f=L.raw_history?L.feature(rawhist,rpre):L.feature(hist,pre);
        stream=mix(stream);
        double roll=(stream>>11)*0x1.0p-53;
        int a;
        // Directed structural exploration. Task-driven behaviour never produces
        // the negative evidence a merger needs, and raising epsilon does not
        // produce it either, so the agent must deliberately walk to an untried
        // event and try it. Every step of that detour is charged to the same
        // environment budget as any other step.
        bool structural=false;
        if(c.beta>0){
            if(L.struct_left<=0){
                L.struct_target=-1;
                if(roll<c.beta){
                    // "scarcest" picks the event this state has tried least, which
                    // is the evidence a merger is missing. "random" picks any
                    // event, which buys skill-driven navigation without any
                    // structural targeting: the control that says whether the
                    // targeting matters or only the detours do.
                    int e;
                    // "fresh" needs no structure at all: pick uniformly among
                    // labels this episode has not already collected. If the
                    // directed detours are really buying structural evidence,
                    // this should do clearly worse than the rules that consult
                    // the hypothesis. If it matches them, the detours were
                    // never about structure.
                    if(c.target=="fresh"){
                        int cand[16],k=0;
                        for(int g=0;g<T.alphabet;g++){
                            bool had=false;
                            for(int seen:hist)if(seen==g){had=true;break;}
                            if(!had)cand[k++]=g;
                        }
                        e=k?cand[mix(stream+6)%uint64_t(k)]:int(mix(stream+5)%uint64_t(T.alphabet));
                    }
                    else if(c.target=="random")e=int(mix(stream+5)%uint64_t(T.alphabet));
                    else if(c.target=="separating"){
                        e=L.separating_experiment(node);
                        if(e<0)e=L.tree.scarcest_untried(node,std::max(1,c.quota));
                    }else e=L.tree.scarcest_untried(node,c.quota);
                    if(e>=0){L.struct_target=e;L.struct_left=c.struct_cap;}
                }
            }
            if(L.struct_target>=0){structural=true;L.struct_left--;L.structural_steps++;
                a=L.skill_action(L.struct_target,s);}
        }
        int active_goal=L.goal_mode?L.select_goal(qstate,s,hist,node):-1;
        if(!structural){
            if(roll<c.epsilon)a=int(mix(stream+1)%4);
            else if(L.goal_mode){
                int g=active_goal;
                if(g<0)a=int(mix(stream+2)%4);
                else{
                    size_t base=(size_t(g)*L.T.N+s)*4;
                    double v=*std::max_element(L.G.begin()+base,L.G.begin()+base+4);
                    int cand[4],k=0;
                    for(int b=0;b<4;b++)if(L.G[base+b]==v)cand[k++]=b;
                    a=cand[mix(stream+2)%uint64_t(k)];
                }
            }
            else{double v=L.q(f,s,L.greedy(f,s));int cand[4],k=0;
                for(int b=0;b<4;b++)if(L.q(f,s,b)==v)cand[k++]=b;
                a=cand[mix(stream+2)%uint64_t(k)];}
        }
        int ns=T.moves[size_t(s)*4+a],ev=T.events[size_t(s)*4+a];
        int nq=T.next_state(qstate,ev);
        // Entering the failure sink is a change of task state but it is not
        // progress. Reporting it as progress made the fatal label look like an
        // advancing one, which let the two branch states merge and left no
        // evidence that could ever refute the merge.
        bool fell=T.failing[nq]!=0;
        bool progress=nq!=qstate&&!fell;
        if(c.shape=="computedproduct"||c.shape=="guided"){
            if(L.map_next[size_t(s)*4+a]<0)L.dirty=true;
            L.learn_edge(s,a,ns,ev);
        }
        if(L.goal_mode&&!c.relabel){if(active_goal>=0)L.train_one_skill(active_goal,s,a,ns,ev);}
        else L.train_skills(s,a,ns,ev);
        if(L.struct_target>=0&&ev==L.struct_target){L.struct_target=-1;L.struct_left=0;}
        if(L.tracking&&ev>=0&&!progress){
            if(!L.tree.ignored[node][ev]++)L.dirty=true;
        }
        if(progress){
            if(L.tracking){
                if(!L.tree.advanced[node][ev]++)L.dirty=true;
                uint64_t k=L.enc.hist_key(hist,int(hist.size()));
                std::vector<int> extended=hist;extended.push_back(ev);
                uint64_t ek=L.enc.hist_key(extended,int(extended.size()));(void)k;
                bool fresh=false;int child=L.tree.find_or_add(ek,nq,fresh);
                if(fresh){L.blocks.push_back(int(L.blocks.size()));L.dirty=true;
                          L.node_depth[child]=int(hist.size())+1;}
                if(!L.merging)L.blocks.back()=int(L.blocks.size())-1;
                L.tree.succ[node][ev]=child;node=child;L.node_path.push_back(node);
            }
            hist.push_back(ev);
        }
        if(progress){L.deepest=std::max(L.deepest,int(hist.size()));L.total_events++;}
        bool win=T.accepting[nq]!=0;
        bool dead=fell;
        if(dead){failures++;
            if(L.tracking){
                for(int touched:L.node_path)
                    if(touched<int(L.tree.danger.size()))L.tree.danger[touched]++;
                if(node<int(L.tree.danger.size()))L.tree.danger[node]++;
                L.dirty=true;
            }}
        if(win&&L.tracking&&node<int(L.tree.accepting.size())&&!L.tree.accepting[node]++)L.dirty=true;
        // Two different facts: when this run first finished, and how long a
        // finished task is. A transferred structure supplies the second without
        // supplying the first.
        if(win&&first_success<0)first_success=step;
        if(win&&!L.solved_length)L.solved_length=int(hist.size());
        if(ev>=0)rawhist.push_back(ev);
        Raw online{s,a,ns,ev,episode,pre,int(hist.size()),rpre,int(rawhist.size()),win,dead};
        (void)f;
        if(c.crm)L.apply_counterfactual(online);else L.apply(online);
        if(L.buffer.size()<20000)L.buffer.push_back(online);else L.buffer[size_t(step)%20000]=online;
        if(!c.crm)for(int j=0;j<c.replay;j++)
            L.apply(L.buffer[mix(uint64_t(step)*5+j+mix(c.seed+333))%L.buffer.size()]);
        // The goal arm has no task-indexed table, so its six replayed updates go
        // to the skills instead. One replayed row retrains all |events| skills,
        // which is the same six updates the other arms spend on their task Q.
        if(L.goal_mode)L.skill_replay(L.buffer[mix(uint64_t(step)*11+mix(c.seed+777))%L.buffer.size()]);
        hashadd(trace,uint64_t(s*4+a));
        s=ns;qstate=nq;t++;
        if(win&&c.shape=="learnedproduct")L.credit_success();
        // Three costs for a wrong commit, which is the dose this experiment
        // varies. Ignored: nothing happens. Terminating: the episode ends, which
        // hands the agent a free early-stop signal. Continuing: the episode runs
        // on with success now impossible, so the remaining steps are wasted, and
        // that waste is what the project's earlier dose-response measured.
        if(dead)doomed_steps+=T.horizon-t;
        bool stop=win||t>=T.horizon||(dead&&!c.doom_continues);
        if(dead&&c.doom_continues)stop=(t>=T.horizon);
        if(dead&&first_failure<0)first_failure=step;
        if(stop)reset(++which);
        if(step%c.every==0){
            int solved=0;double steps=0;
            for(int i=0;i<S;i++){int r=rollout(i);if(r>0){solved++;steps+=r;}}
            double rate=double(solved)/S;
            checks.push_back({double(step),rate,solved?steps/solved:0});
            auc+=rate*c.every;
            if(rate>=.9){if(first90<0)first90=step;if(last>=.9&&stable90<0)stable90=step;}
            last=rate;
        }
    }
    // Diagnostics compare the learner's partition of the nodes it actually
    // visited against the true Myhill-Nerode classes of those nodes. Never fed
    // back into learning.
    double precision=1,recall=1;int agree=0,learned_pairs=0,true_pairs=0;
    for(int i=0;i<L.tree.size();i++)for(int j=i+1;j<L.tree.size();j++){
        bool same_true=T.minimal[L.tree.truth[i]]==T.minimal[L.tree.truth[j]];
        bool same_learned=L.blocks[i]==L.blocks[j];
        if(same_learned)learned_pairs++;
        if(same_true)true_pairs++;
        if(same_learned&&same_true)agree++;
    }
    if(learned_pairs)precision=double(agree)/learned_pairs;
    if(true_pairs)recall=double(agree)/true_pairs;
    std::set<int> reached;for(int i=0;i<L.tree.size();i++)reached.insert(T.minimal[L.tree.truth[i]]);
    // Structural coverage: of the (node, event) pairs the learner would need to
    // separate states, how many has task-driven behaviour actually tried?
    int tried=0,total=0;
    for(int i=0;i<L.tree.size();i++){
        if(L.tree.visits[i]<c.theta)continue;
        for(int e=0;e<T.alphabet;e++){total++;if(L.tree.advanced[i][e]||L.tree.ignored[i][e])tried++;}
    }
    double coverage=total?double(tried)/total:0;
    double final_rate=checks.empty()?0:checks.back()[1];
    std::cout<<"{\"task\":\""<<c.task<<"\",\"method\":\""<<c.method<<"\",\"window\":"<<c.window
             <<",\"seed\":"<<c.seed<<",\"budget\":"<<c.budget<<",\"epsilon\":"<<c.epsilon
             <<",\"gamma\":"<<c.gamma<<",\"reward_ratio\":"<<c.reward_ratio
             <<",\"first90\":"<<first90<<",\"stable90\":"<<stable90
             <<",\"first_success\":"<<first_success
             <<",\"failures\":"<<failures<<",\"first_failure\":"<<first_failure
             <<",\"doomed_steps\":"<<(c.doom_continues?doomed_steps:0)
             <<",\"doom_continues\":"<<(c.doom_continues?1:0)
             <<",\"steps_after_first_success\":"<<((first90>0&&first_success>0)?first90-first_success:-1)
             <<",\"auc\":"<<auc/c.budget<<",\"final_success\":"<<final_rate
             <<",\"features\":"<<(L.merging?size_t(L.classes):L.ids.size())
             <<",\"feature_cap\":"<<MAXF<<",\"overflow_hits\":"<<L.overflow_hits
             <<",\"nodes\":"<<L.tree.size()<<",\"classes\":"<<L.classes
             <<",\"true_reached\":"<<reached.size()<<",\"revisions\":"<<L.revisions
             <<",\"first_merge\":"<<L.first_merge
             <<",\"merge_precision\":"<<precision<<",\"merge_recall\":"<<recall
             <<",\"structural_coverage\":"<<coverage
             <<",\"hypothesis_classes\":"<<(L.hyp.empty()?0:*std::max_element(L.hyp.begin(),L.hyp.end())+1)
             <<",\"product_entries\":"<<L.product_entries
             <<",\"structural_steps\":"<<L.structural_steps
             <<",\"skill_updates\":"<<L.skill_updates
             <<",\"seeded_nodes\":"<<(c.seed_structure.empty()?0:L.tree.size())
             <<",\"quota\":"<<c.quota<<",\"risk\":"<<c.risk<<",\"beta\":"<<c.beta
             <<",\"danger_nodes\":"<<[&]{int k=0;for(int v:L.tree.danger)if(v)k++;return k;}()
             <<",\"danger_total\":"<<[&]{long long k=0;for(int v:L.tree.danger)k+=v;return k;}()<<",\"target\":\""<<c.target<<"\",\"shape\":\""<<c.shape
             <<"\",\"shape_scale\":"<<c.shape_scale
             <<",\"minimality_gap\":"<<(reached.empty()?0:double(L.classes)/double(reached.size()))
             <<",\"revision_updates\":"<<L.revision_updates
             <<",\"crm\":"<<(c.crm?1:0)<<",\"relabel\":"<<(c.relabel?1:0)
             <<",\"replay\":"<<c.replay<<",\"merge_rule\":\""<<c.merge_rule<<"\""<<",\"theta\":"<<c.theta<<",\"revision\":\""<<c.revision<<"\""
             <<",\"updates\":"<<L.updates
             <<",\"episodes\":"<<episodes.size()<<",\"starts\":"<<S
             <<",\"horizon\":"<<T.horizon<<",\"reward_success\":"<<L.reward_success
             <<",\"trace_hash\":\""<<trace<<"\",\"checkpoints\":[";
    for(size_t i=0;i<checks.size();i++)std::cout<<(i?",":"")<<"["<<int(checks[i][0])<<","<<checks[i][1]<<","<<checks[i][2]<<"]";
    std::cout<<"]";
    if(c.dump_tree){
        // The evidence and the partition the learner ended with, so an
        // independent implementation can be checked against this one.
        std::cout<<",\"tree\":{\"alphabet\":"<<T.alphabet<<",\"nodes\":[";
        for(int i=0;i<L.tree.size();i++){
            std::cout<<(i?",":"")<<"{\"visits\":"<<L.tree.visits[i]
                     <<",\"accepting\":"<<L.tree.accepting[i]
                     <<",\"danger\":"<<L.tree.danger[i]
                     <<",\"truth\":"<<T.minimal[L.tree.truth[i]]<<",\"advanced\":[";
            for(int e=0;e<T.alphabet;e++)std::cout<<(e?",":"")<<L.tree.advanced[i][e];
            std::cout<<"],\"ignored\":[";
            for(int e=0;e<T.alphabet;e++)std::cout<<(e?",":"")<<L.tree.ignored[i][e];
            std::cout<<"],\"succ\":[";
            for(int e=0;e<T.alphabet;e++)std::cout<<(e?",":"")<<L.tree.succ[i][e];
            std::cout<<"]}";
        }
        std::cout<<"],\"blocks\":[";
        for(size_t i=0;i<L.blocks.size();i++)std::cout<<(i?",":"")<<L.blocks[i];
        std::cout<<"],\"theta\":"<<c.theta<<",\"quota\":"<<c.quota
                 <<",\"strict_children\":"<<(c.strict_children?1:0)<<"}";
    }
    std::cout<<"}"<<std::endl;
    (void)hits;
    return 0;
}
