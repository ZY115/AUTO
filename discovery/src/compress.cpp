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
#include <map>
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
    // Two orthogonal switches for the mechanism ablation. `skill_key`
    // decides whether one goal skill is shared across task states or a
    // separate one is learned per state; `route` decides whether the
    // automaton names the goal or the agent learns to pick it itself.
    std::string skill_key="shared",route="automaton";int option_cap=64;
    // Robustness control for the irreversibility round: the learned router
    // may propose the hazard itself as a goal, which no automaton-routed arm
    // can ever do. Turning this on hands it that knowledge for free, which is
    // oracle help, so it is a check and never the headline arm.
    int meta_safe=0;
    int seed=0,budget=200000,every=1000,window=1,theta=20,repartition=100,replay_budget=2000;
    int quota=0,struct_cap=60;double beta=0,risk=0,spatial_scale=-1;
    std::string target="scarcest",shape="none";
    bool strict_children=false,dump_tree=false,shape_normalize=false,doom_continues=false;
    bool crm=false,relabel=true,source_frequency=true;int replay=5;
    // P0 audit switches. `cache_reuse=0` forbids re-using a label recorded
    // earlier at this node, so every firing is either paid for or unknown; that
    // separates "the query policy is poor" from "the cache is wrong".
    // `cache_key=truestate` files labels under the true task state instead of
    // the agent's history node. It is privileged and exists only as a ceiling:
    // it is what perfect evidence attribution would look like.
    bool cache_reuse=true,dump_queries=false;std::string cache_key="node";
    // The history a representation reads was tied to the method name, so only
    // `rawhistory` ever saw the full event stream and `count`/`window` silently
    // read the verification-filtered progress history instead. Source and
    // encoding are now separate declarations.
    std::string history_source="auto";
    // A window that also hashes the total prefix length is not a window. Kept
    // off by default so every earlier run reproduces bitwise.
    bool window_pure=false;
    // How many times the agent may ask "did that event advance the task?".
    // Negative means the question is free and always answered, which is what
    // every experiment before this one assumed.
    long long verify_budget=-1;
    // Which firings to spend a query on, and what to believe about a firing no
    // query was spent on. They are different choices and are varied separately.
    std::string verify_policy="arrival",verify_fallback="own",verify_partition="sweep";
    double verify_rate=.25;int verify_support=1;
    std::string merge_rule="sweep";int sig_support=1;
    std::string seed_structure;
    double shape_scale=0;
    double epsilon=.02,gamma=.99,alpha=.3,cost=.01,reward_ratio=8;
    // Set the success reward directly instead of deriving it from the step cost,
    // so a published protocol's "1 on success, 0 otherwise" can be matched
    // exactly. Negative means "derive it as before".
    double reward_success_override=-1;
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
    // Three verdicts, not two. An event that ends the task is not an event
    // that does nothing, and a node that has seen one must never merge with a
    // node that has seen the other. Folding fatal into ignored was the defect
    // that let the failure sink look like a no-op.
    std::vector<std::vector<int>> succ,advanced,ignored,fatal;
    // How often an event advanced here, used only to rank candidate goals. It is
    // separate from `advanced`, which is evidence: evidence says the event can
    // advance, the count says how often it did, and a count carried from another
    // map is a preference of that map's layout and collection policy rather than
    // a fact about the task. Keeping them apart is what lets an arm transfer the
    // rule without transferring the preference.
    std::vector<std::vector<int>> rank;
    std::vector<int> accepting,danger;
    std::unordered_map<uint64_t,int> ids;
    explicit Tree(int a):alphabet(a){add(1469598103934665603ULL,0);}
    int add(uint64_t k,int truth_state){
        int id=int(key.size());ids[k]=id;key.push_back(k);truth.push_back(truth_state);
        visits.push_back(0);block.push_back(id);accepting.push_back(0);danger.push_back(0);
        succ.emplace_back(alphabet,-1);advanced.emplace_back(alphabet,0);
        ignored.emplace_back(alphabet,0);fatal.emplace_back(alphabet,0);
        rank.emplace_back(alphabet,0);return id;
    }
    int find_or_add(uint64_t k,int truth_state,bool&fresh){
        auto it=ids.find(k);if(it!=ids.end()){fresh=false;return it->second;}
        fresh=true;return add(k,truth_state);
    }
    int size()const{return int(key.size());}
    uint64_t required(int h)const{uint64_t m=0;for(int e=0;e<alphabet;e++)if(advanced[h][e])m|=1ULL<<e;return m;}
    int observations(int h)const{int n=0;for(int e=0;e<alphabet;e++)n+=tried(h,e);return n;}
    // Unknown is a fourth outcome and is not "ignored". A pair the agent has
    // never paid to check carries no evidence at all, so nothing downstream can
    // mistake "not checked" for "checked and it does nothing".
    bool labelled(int h,int e)const{return advanced[h][e]||ignored[h][e]||fatal[h][e];}
    int tried(int h,int e)const{return advanced[h][e]+ignored[h][e]+fatal[h][e];}
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
            if(advanced[a][e]&&(ignored[b][e]||fatal[b][e]))return true;
            if(ignored[a][e]&&(advanced[b][e]||fatal[b][e]))return true;
            if(fatal[a][e]&&(advanced[b][e]||ignored[b][e]))return true;
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
    std::vector<uint64_t> adv,ign,fat;
    // Counts, not just bits: the arms that read a class must rank its events
    // the same way the arms that read a node do, or they differ in two ways
    // at once and the comparison measures the tie-break.
    std::vector<std::vector<int>> tried,succ,advn,ignn;
    struct Undo{int loser,winner,visits,observations,danger;uint64_t adv,ign,fat;
                std::vector<int> tried,succ,advn,ignn;};
    std::vector<Undo> log;

    void init(const Tree&T){
        int n=T.size();alphabet=T.alphabet;
        parent.resize(n);visits.resize(n);observations.resize(n);danger.resize(n);
        adv.assign(n,0);ign.assign(n,0);fat.assign(n,0);
        tried.assign(n,{});succ.assign(n,{});advn.assign(n,{});ignn.assign(n,{});
        for(int i=0;i<n;i++){
            parent[i]=i;visits[i]=T.visits[i];observations[i]=T.observations(i);
            danger[i]=T.danger[i];
            adv[i]=0;ign[i]=0;fat[i]=0;tried[i].assign(alphabet,0);
            succ[i].assign(alphabet,-1);advn[i].assign(alphabet,0);
            ignn[i].assign(alphabet,0);
            for(int e=0;e<alphabet;e++){
                if(T.advanced[i][e])adv[i]|=1ULL<<e;
                if(T.ignored[i][e])ign[i]|=1ULL<<e;
                if(T.fatal[i][e])fat[i]|=1ULL<<e;
                tried[i][e]=T.tried(i,e);succ[i][e]=T.succ[i][e];
                advn[i][e]=T.rank[i][e];
                ignn[i][e]=T.ignored[i][e]+T.fatal[i][e];
            }
        }
        log.clear();
    }
    int find(int x)const{while(parent[x]!=x)x=parent[x];return x;}
    bool conflict(int a,int b)const{
        return (adv[a]&(ign[b]|fat[b]))||(ign[a]&(adv[b]|fat[b]))||(fat[a]&(adv[b]|ign[b]));
    }
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
        log.push_back({l,w,visits[w],observations[w],danger[w],adv[w],ign[w],fat[w],
                       tried[w],succ[w],advn[w],ignn[w]});
        parent[l]=w;adv[w]|=adv[l];ign[w]|=ign[l];fat[w]|=fat[l];
        visits[w]+=visits[l];observations[w]+=observations[l];danger[w]+=danger[l];
        for(int e=0;e<alphabet;e++){
            tried[w][e]+=tried[l][e];advn[w][e]+=advn[l][e];ignn[w][e]+=ignn[l][e];
            if(succ[w][e]<0)succ[w][e]=succ[l][e];
        }
    }
    void rollback(size_t mark){
        while(log.size()>mark){
            const Undo&u=log.back();
            parent[u.loser]=u.loser;
            visits[u.winner]=u.visits;observations[u.winner]=u.observations;
            danger[u.winner]=u.danger;
            adv[u.winner]=u.adv;ign[u.winner]=u.ign;fat[u.winner]=u.fat;
            tried[u.winner]=u.tried;succ[u.winner]=u.succ;advn[u.winner]=u.advn;
            ignn[u.winner]=u.ignn;
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

    // Adopt a partition given from outside, with no conflict test. Used for the
    // true partition, which is a ceiling rather than a method.
    int adopt(const Tree&T,const std::vector<int>&given,std::vector<int>&out){
        init(T);
        int n=T.size();
        std::unordered_map<int,int> first;
        for(int i=0;i<n&&i<int(given.size());i++){
            auto it=first.find(given[size_t(i)]);
            if(it==first.end())first[given[size_t(i)]]=i;else unite(it->second,i);
        }
        out.assign(n,0);std::unordered_map<int,int> label;
        for(int i=0;i<n;i++){int r=find(i);auto it=label.find(r);
            if(it==label.end()){int v=int(label.size());label[r]=v;out[i]=v;}else out[i]=it->second;}
        return int(label.size());
    }
    // A learned task-state abstraction built from partial feedback signatures.
    //
    // Step 30 measured the space this has to fill: filing a paid answer by the
    // progress history costs 177 queries where the true task state costs 86, and
    // the whole gap is over-splitting. What attribution needs is therefore
    // different from what the sweep rule provides: it must share from the *first*
    // observation, because the value is in not buying the second one, so there is
    // no visit threshold here.
    //
    // A node's signature is its response vector: for each event, advance, ignore,
    // fatal, or unknown. Two nodes may share when no event has both of them
    // labelled differently, **and** they agree on at least `support` events. The
    // support gate is what keeps a brand-new node, whose signature is all
    // unknown and therefore compatible with everything, from joining an
    // arbitrary class.
    //
    // Compatibility alone is not task-state equivalence: equivalence is about
    // future behaviour. So after the signature pass the partition is refined to a
    // fixed point — two nodes cannot share if any event sends them to different
    // classes — which is the Moore construction on the hypothesis the agent
    // currently holds.
    int rebuild_signature(const Tree&T,std::vector<int>&out,int support){
        init(T);
        int n=T.size();
        auto verdict=[&](int i,int e)->int{
            if(T.advanced[i][e])return 1;
            if(T.ignored[i][e])return 2;
            if(T.fatal[i][e])return 3;
            return 0;
        };
        auto ok=[&](int a,int b){
            int agree=0;
            for(int e=0;e<alphabet;e++){
                int x=verdict(a,e),y=verdict(b,e);
                if(x&&y){if(x!=y)return false;agree++;}
            }
            return agree>=support;
        };
        // Merge with successor closure, not "group then refine". Grouping nodes
        // independently and refining afterwards cannot hold in a prefix tree: two
        // merged nodes have distinct successors, which land in different classes
        // and split the parents straight back. A merge has to carry its
        // successors with it as it is made, and roll back whole if any implied
        // pair contradicts. This is the classic state-merging shape, with the
        // signature agreement gate replacing the visit quota.
        std::vector<int> reps;
        for(int i=0;i<n;i++){
            bool joined=false;
            for(int r:reps){
                if(find(r)==find(i))continue;
                if(!ok(i,r))continue;
                size_t mark=log.size();
                std::vector<std::pair<int,int>> stack{{i,r}};
                bool good=true;
                while(!stack.empty()&&good){
                    auto[x,y]=stack.back();stack.pop_back();
                    int a=find(x),b=find(y);
                    if(a==b)continue;
                    if(conflict(a,b)){good=false;break;}
                    std::vector<std::pair<int,int>> kids;
                    for(int e=0;e<alphabet;e++){
                        int sa=succ[size_t(a)][size_t(e)],sb=succ[size_t(b)][size_t(e)];
                        if(sa>=0&&sb>=0)kids.push_back({sa,sb});
                    }
                    unite(a,b);
                    for(auto&k:kids)stack.push_back(k);
                }
                if(!good){rollback(mark);continue;}
                joined=true;break;
            }
            if(!joined)reps.push_back(i);
        }
        // Publish through the union-find so the class-level evidence aggregates
        // the same way every other rule's does.
        out.assign(size_t(n),0);std::unordered_map<int,int> label;
        for(int i=0;i<n;i++){int r=find(i);auto it=label.find(r);
            if(it==label.end()){int v=int(label.size());label[r]=v;out[size_t(i)]=v;}
            else out[size_t(i)]=it->second;}
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
            if(!c.window_pure)hashadd(k,uint64_t(upto)+1000000);
            return k;
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
// ignored[], fatal[], successor[]. Node keys are recomputed from the edges by
// this run's own encoder rather than trusted from the file.
int seed_tree(Tree&tree,const Task&T,const std::string&source,std::vector<int>&blocks,
              bool carry_frequency){
    std::ifstream in(source);
    if(!in)throw std::runtime_error("cannot open structure "+source);
    int alphabet=0,n=0;in>>alphabet>>n;
    if(alphabet!=tree.alphabet)throw std::runtime_error("alphabet mismatch");
    std::vector<int> vis(n),acc(n);
    std::vector<std::vector<int>> adv(n),ign(n),fat(n),succ(n);
    for(int i=0;i<n;i++){
        in>>vis[i]>>acc[i];
        adv[i].resize(alphabet);ign[i].resize(alphabet);
        fat[i].resize(alphabet);succ[i].resize(alphabet);
        for(int e=0;e<alphabet;e++)in>>adv[i][e];
        for(int e=0;e<alphabet;e++)in>>ign[i][e];
        for(int e=0;e<alphabet;e++)in>>fat[i][e];
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
    tree.advanced[0]=adv[0];tree.ignored[0]=ign[0];tree.fatal[0]=fat[0];
    if(carry_frequency)tree.rank[0]=adv[0];
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
        tree.advanced[id]=adv[i];tree.ignored[id]=ign[i];tree.fatal[id]=fat[i];
        if(carry_frequency)tree.rank[id]=adv[i];
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
// A replayed row has to be interpretable as the moment it was collected, not as
// the moment it is replayed. `goal` and `node` are what the agent was actually
// pursuing and standing on; recomputing them later from current skill values and
// a null node gave a different answer, which made the no-relabel ablation
// untrue to its own name. `predead` marks a transition taken after the task had
// already failed: such a step is not evidence about the history that preceded
// the failure.
struct Raw { int s,a,ns,ev,episode,pre,post,rpre,rpost,goal,node,q;bool win,dead,predead; };

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
    // Skills indexed by (task state, goal) instead of goal alone: the same
    // reading of the automaton, with reuse across task states switched off.
    bool perstate=false;
    // The goal chosen by a learned value over goals rather than by the
    // automaton. The meta-controller sees the physical cell and the same
    // history feature the flat arm gets, never the task state.
    bool meta_route=false;
    std::vector<double> M;
    int meta_goal=-1,meta_f=-1,meta_s=-1,meta_k=0;double meta_R=0.0;
    long long options=0,option_hits=0;
    size_t gidx(int q,int g,int s,int a)const{
        return perstate?((size_t(q)*T.alphabet+size_t(g))*T.N+size_t(s))*4+size_t(a)
                       :(size_t(g)*T.N+size_t(s))*4+size_t(a);
    }
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
    // Counted, not acted on: how many decisions the ranking weights changed
    // relative to ranking the same candidates by target-side skill value alone.
    mutable long long rank_decisions=0,rank_changed=0;
    int select_goal(int q,int s,const std::vector<int>&hist,int node=-1)const{
        int cand[64];int k;
        // The same candidate set before the ranking weights narrow it, kept only
        // so the run can report how many decisions those weights changed.
        int wide[64];int kw=0;
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
                int bestn=-1;
                for(int e=0;e<T.alphabet;e++)
                    if(tree.advanced[node][e]){bestn=std::max(bestn,tree.rank[node][e]);
                                               wide[kw++]=e;}
                if(bestn>=0)for(int e=0;e<T.alphabet;e++)
                    if(tree.advanced[node][e]&&tree.rank[node][e]==bestn)cand[k++]=e;
                // Optimism about an untried event is fine; optimism about an
                // event already seen to end the task is not. Before fatal was a
                // verdict of its own it was recorded as "ignored" and excluded
                // here by accident, so separating the two silently made this
                // fallback propose lethal goals.
                if(k==0)for(int e=0;e<T.alphabet;e++)
                    if(tree.ignored[node][e]==0&&tree.fatal[node][e]==0)cand[k++]=e;
            }
            if(k==0)for(int e=0;e<T.alphabet;e++)cand[k++]=e;
        }
        else if(c.goal_select=="merged"||c.goal_select=="selective"
                ||c.goal_select=="aggressive"){
            // "aggressive" is the same use of the structure as "merged", on the
            // over-merging partition instead of the conservative one. Keeping
            // both as arms is what lets the selective arm be compared against
            // using either belief alone rather than against a strawman.
            const Partition&P=(c.goal_select=="aggressive")?aggr:part;
            // Both read the class the current history folds into, so the two
            // arms differ only in how the class's evidence is used, never in
            // what evidence they hold.
            uint64_t adv=class_bits(P,node,0),ign=class_bits(P,node,1),
                     fat=class_bits(P,node,2);
            k=0;
            if(c.goal_select=="selective"&&node>=0&&node<int(tree.advanced.size())){
                // Tier zero, and the pilot says it is the tier that matters:
                // evidence recorded at this exact history. Folding into a class
                // can only dilute it, because a class pools whatever its other
                // members saw and a wrong merge pools something false. Inference
                // is for histories the agent has no record of, not for the ones
                // it has.
                int best=-1;
                for(int e=0;e<T.alphabet;e++)
                    if(tree.advanced[node][e])best=std::max(best,tree.rank[node][e]);
                if(best>=0)for(int e=0;e<T.alphabet;e++)
                    if(tree.advanced[node][e]&&tree.rank[node][e]==best)cand[k++]=e;
            }
            if(k==0&&adv){
                // Tier one: somebody in this class saw these events advance.
                // Precision .963 in the offline audit, and the closed-world
                // reading of the same evidence says nothing else advances here.
                // Ranked by the class's pooled count for the same reason a node
                // is ranked by its own: a rarely seen frontier event has a skill
                // with little data behind it. Ranking the two differently would
                // have made these arms differ in the tie-break as well as in the
                // evidence, and the comparison would have measured the tie-break.
                int r=P.find(node),best=-1;
                for(int e=0;e<T.alphabet;e++)
                    if(adv&(1ULL<<e)){best=std::max(best,P.advn[size_t(r)][size_t(e)]);
                                      wide[kw++]=e;}
                if(best>=0)for(int e=0;e<T.alphabet;e++)
                    if((adv&(1ULL<<e))&&P.advn[size_t(r)][size_t(e)]==best)cand[k++]=e;
            }
            if(k==0){
                // Nothing positive here. Rule out what the class has seen do
                // nothing or end the task; that much is observed, not assumed.
                uint64_t mask=0;
                for(int e=0;e<T.alphabet;e++)
                    if(!((ign|fat)&(1ULL<<e)))mask|=1ULL<<e;
                if(c.goal_select=="selective"&&!any_fatal&&__builtin_popcountll(mask)>1){
                    // Tier three, and only here: the over-merging partition's
                    // extra answers, consulted when nothing fatal has ever been
                    // observed, which is the agent's own evidence that a wrong
                    // goal is recoverable.
                    uint64_t a2=class_bits(aggr,node,0);
                    if(a2&mask)mask&=a2;
                }
                for(int e=0;e<T.alphabet;e++)if(mask&(1ULL<<e))cand[k++]=e;
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
        }else if(c.goal_select=="value_ranked"){
            // The true allowed set, consumed by exactly the rule every other arm
            // uses: narrow by the ranking counts this run accumulated, then pick
            // by skill value. The earlier oracle arm skipped the narrowing, so it
            // differed from the learned arms in two ways at once and could not
            // serve as a ceiling on "what perfect rules are worth".
            int allow[64];int na=allowed(q,allow);
            int best=-1;
            for(int i=0;i<na;i++){
                int e=allow[i];wide[kw++]=e;
                int rc=(node>=0&&node<int(tree.rank.size()))?tree.rank[node][e]:0;
                best=std::max(best,rc);
            }
            k=0;
            for(int i=0;i<na;i++){
                int e=allow[i];
                int rc=(node>=0&&node<int(tree.rank.size()))?tree.rank[node][e]:0;
                if(rc==best)cand[k++]=e;
            }
        }else k=allowed(q,cand);
        if(k==0)return -1;
        auto by_value=[&](const int*set,int n){
            int b=set[0];double bv=-1e300;
            for(int i=0;i<n;i++){
                double v=*std::max_element(G.begin()+gidx(q,set[i],s,0),
                                           G.begin()+gidx(q,set[i],s,0)+4);
                if(v>bv){bv=v;b=set[i];}
            }
            return b;
        };
        if(c.goal_select=="first")return cand[0];
        int bg=(k==1)?cand[0]:by_value(cand,k);
        if(kw>k){rank_decisions++;if(by_value(wide,kw)!=bg)rank_changed++;}
        return bg;
    }
    void skill_replay(const Raw&x){
        if(c.relabel){train_skills(x.q,x.s,x.a,x.ns,x.ev);return;}
        // Without relabelling the replayed row retrains only the goal that was
        // actually being pursued when it was collected, recovered by folding its
        // episode's own event prefix through the machine.
        if(x.goal>=0)train_one_skill(x.q,x.goal,x.s,x.a,x.ns,x.ev);
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
                if(tree.ignored[i][e]||tree.fatal[i][e])block_ignored[hyp[i]]|=1ULL<<e;
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
        reward_success(c.reward_success_override>=0?c.reward_success_override
                       :c.reward_ratio*c.cost*t.maxlen),tree(t.alphabet){
        raw_history=(c.history_source=="raw")
                    ||(c.history_source=="auto"&&c.method=="rawhistory");
        verify_left=c.verify_budget;
        if(c.cache_key=="truestate")init_true_cache();
        goal_mode=c.method=="goal";
        merging=c.method=="merged"||c.method=="merged_explore"
                ||c.goal_select=="merged"||c.goal_select=="selective"
                ||c.goal_select=="aggressive"
                // The partition is also needed when the query policy or the
                // fallback consults it, even though the goal selector does not.
                ||c.verify_policy=="novel"||c.verify_policy=="aligned"
                ||c.verify_policy=="closed"||c.verify_fallback=="class"
                // ...and when a paid answer is filed against the class.
                ||c.cache_key=="mergedclass";
        // The prefix tree is also maintained without merging, so an arm can buy
        // structural evidence and still index its table by full history. That
        // control separates "the detours improved exploration" from "the merges
        // improved the representation".
        tracking=merging||c.beta>0||c.shape=="learned"||c.shape=="learnedproduct"
                 ||c.shape=="computedproduct"||c.shape=="guided"
                 ||c.goal_select=="learned"||c.goal_select=="merged"
                 ||c.goal_select=="selective"||c.goal_select=="aggressive"
                 ||c.goal_select=="value_ranked";
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
        perstate=(c.skill_key=="perstate");
        meta_route=(c.route=="meta");
        if((perstate||meta_route)&&!goal_mode)
            throw std::runtime_error("--skill-key perstate and --route meta are "
                "readings of the goal arm; use --method goal");
        G.assign(perstate?size_t(t.nstates)*t.alphabet*t.N*4
                         :size_t(t.alphabet)*t.N*4,0.0);
        if(meta_route)M.assign(size_t(MAXF)*t.N*t.alphabet,0.0);
        if(c.shape=="oracle")accept_dist=t.to_accept();
        blocks.assign(1,0);classes=1;
    }
    // The table is finite. A representation that keeps producing new states
    // eventually fills it, and everything after that collides into one row.
    // That is what happens to any bounded learner facing an unbounded feature
    // space, so it is reported rather than treated as an error. The map is
    // allowed to keep growing so the number of distinct states can be counted.
    // The absorbing failure state is a state. Post-failure transitions update
    // this row rather than the row of the healthy history the episode had
    // reached before it died.
    static const int DEADF=MAXF-2;
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
    // The hypothesis as it stood at the last repartition. JIRP's rule compares
    // two hypotheses, so the old one has to be kept rather than re-read off a
    // tree that has meanwhile grown.
    struct Hypothesis {
        bool valid=false;int classes=0;
        std::vector<int> blocks,acc;
        std::vector<std::vector<int>> adv,ign,fat,succ;
    } prev;
    long long jirp_transfers=0;
    long long verifications=0,verify_left=0,unknown_firings=0;
    long long generalised_firings=0,declined=0;
    // Measurement only, no behaviour depends on these.
    long long desync_steps=0,misattributed=0,stale_uses=0,stale_missed_advance=0;
    // The attribution decomposition. A query is "necessary" the first time this
    // (true task state, event) pair is paid for anywhere; every later query on a
    // pair already answered under some other attribution state is redundant, and
    // is exactly what a perfect abstraction would have saved. Measurement only.
    long long redundant_queries=0,necessary_queries=0;
    // The ordered log of what was actually paid for: the believed history at the
    // time, the event, and the verdict the payment bought. Enough to replay the
    // agent's information state offline and ask what was already determined.
    // history, event, verdict, and the cell the agent was standing on, which is
    // what decides how far it would have had to walk for other evidence.
    struct QLog{std::vector<int> h;int e,v,cell,node;};
    std::vector<QLog> query_log;
    std::set<std::pair<int,int>> answered_true;
    long long cache_hits=0;
    // Labels filed under the true task state: the privileged attribution arm.
    std::vector<std::vector<int>> true_adv,true_ign,true_fat;
    // Labels filed under the complete visible event prefix. That prefix costs
    // nothing — which label fired is always visible — so this changes only where
    // evidence is filed, never the goal controller and never the query budget.
    std::unordered_map<uint64_t,std::vector<int>> full_adv,full_ign,full_fat;
    std::vector<int>&full_slot(std::unordered_map<uint64_t,std::vector<int>>&m,uint64_t k){
        auto it=m.find(k);
        if(it==m.end())it=m.emplace(k,std::vector<int>(size_t(T.alphabet),0)).first;
        return it->second;
    }
    bool full_labelled(uint64_t k,int e){
        return full_slot(full_adv,k)[size_t(e)]||full_slot(full_ign,k)[size_t(e)]
               ||full_slot(full_fat,k)[size_t(e)];
    }
    void init_true_cache(){
        true_adv.assign(size_t(T.nstates),std::vector<int>(size_t(T.alphabet),0));
        true_ign=true_adv;true_fat=true_adv;
    }
    bool true_labelled(int q,int e)const{
        return true_adv[size_t(q)][size_t(e)]||true_ign[size_t(q)][size_t(e)]
               ||true_fat[size_t(q)][size_t(e)];
    }
    bool verify_metered()const{return c.verify_budget>=0;}
    bool may_verify()const{return !verify_metered()||verify_left>0;}
    // What the conservative partition says about this pair, pooled over the
    // class: 1 advance, 0 does not advance, -1 no opinion.
    const Partition&consulted()const{
        if(c.verify_partition=="aggressive")return aggr;
        if(c.verify_partition=="true")return truep;
        return part;
    }
    // What the consulted partition says about this pair: 1 advance, 0 does not
    // advance, -1 no opinion. `closed` completes the hypothesis: an event never
    // seen to advance anywhere in a class that has seen something advance is
    // asserted not to advance here. Step 19 priced that at precision .979 on the
    // conservative partition and .921 on the over-merging one, so it is a
    // separate switch rather than part of the verdict.
    int class_verdict(int node,int e,bool closed=false)const{
        const Partition&P=consulted();
        if(node<0||node>=int(P.parent.size()))return -1;
        int r=P.find(node);
        if(P.adv[size_t(r)]&(1ULL<<e))return 1;
        if(((P.ign[size_t(r)]|P.fat[size_t(r)])&(1ULL<<e))
           &&P.ignn[size_t(r)][size_t(e)]>=c.verify_support)return 0;
        if(closed&&P.adv[size_t(r)])return 0;
        return -1;
    }
    bool node_is_lost(int node)const{
        if(node<0||node>=int(tree.advanced.size()))return true;
        for(int e=0;e<T.alphabet;e++)if(tree.advanced[node][e])return false;
        return true;
    }
    // A second, deliberately over-merging partition of the same evidence. Step 19
    // priced its extra answers at precision .868 against .963, and priced its
    // characteristic error as calling a fatal event an advance. So it is kept
    // apart from the conservative one and consulted only where being wrong is
    // recoverable, rather than being mixed into a single belief.
    Partition aggr;
    std::vector<int> blocks_aggr;int classes_aggr=0;
    Partition truep;std::vector<int> blocks_true;int classes_true=0;
    bool any_fatal=false;
    void refresh_true_partition(){
        if(c.verify_partition!="true")return;
        std::vector<int> given(size_t(tree.size()));
        for(int i=0;i<tree.size();i++)given[size_t(i)]=T.minimal[tree.truth[size_t(i)]];
        truep.alphabet=tree.alphabet;truep.quota=0;truep.risk=0;
        truep.strict_children=false;
        classes_true=truep.adopt(tree,given,blocks_true);
    }
    void refresh_aggressive(){
        if(c.verify_partition=="aggressive"&&c.goal_select!="selective"
           &&c.goal_select!="aggressive"){
            aggr.alphabet=tree.alphabet;aggr.quota=c.quota;aggr.risk=c.risk;
            aggr.strict_children=c.strict_children;
            classes_aggr=aggr.rebuild_fringe(tree,c.theta,blocks_aggr,true);
            return;
        }
        if(c.goal_select!="selective"&&c.goal_select!="aggressive")return;
        aggr.alphabet=tree.alphabet;aggr.quota=c.quota;aggr.risk=c.risk;
        aggr.strict_children=c.strict_children;
        classes_aggr=aggr.rebuild_fringe(tree,c.theta,blocks_aggr,true);
    }
    uint64_t class_bits(const Partition&p,int node,int which)const{
        if(node<0||node>=int(p.parent.size()))return 0;
        int r=p.find(node);
        return which==0?p.adv[size_t(r)]:which==1?p.ign[size_t(r)]:p.fat[size_t(r)];
    }
    std::vector<uint64_t> hypothesis_signature(const std::vector<int>&part,int ncls,
            const std::vector<std::vector<int>>&adv,const std::vector<std::vector<int>>&ign,
            const std::vector<std::vector<int>>&fat,const std::vector<std::vector<int>>&succ,
            const std::vector<int>&acc)const{
        int A=tree.alphabet;
        std::vector<std::vector<int>> cs(size_t(ncls),std::vector<int>(size_t(A),-1));
        std::vector<uint64_t> am(size_t(ncls),0),im(size_t(ncls),0),fm(size_t(ncls),0);
        std::vector<char> ac(size_t(ncls),0);
        for(int nd=0;nd<int(part.size());nd++){
            int b=part[size_t(nd)];if(b<0||b>=ncls)continue;
            if(nd<int(acc.size())&&acc[size_t(nd)])ac[size_t(b)]=1;
            for(int e=0;e<A;e++){
                if(adv[size_t(nd)][size_t(e)])am[size_t(b)]|=1ULL<<e;
                if(ign[size_t(nd)][size_t(e)])im[size_t(b)]|=1ULL<<e;
                if(fat[size_t(nd)][size_t(e)])fm[size_t(b)]|=1ULL<<e;
                int t=succ[size_t(nd)][size_t(e)];
                if(t>=0&&t<int(part.size())&&cs[size_t(b)][size_t(e)]<0)
                    cs[size_t(b)][size_t(e)]=part[size_t(t)];
            }
        }
        std::vector<uint64_t> h(size_t(ncls),0ULL);
        for(int b=0;b<ncls;b++){
            uint64_t k=1469598103934665603ULL;
            hashadd(k,ac[size_t(b)]?1ULL:0ULL);
            hashadd(k,am[size_t(b)]);hashadd(k,im[size_t(b)]);hashadd(k,fm[size_t(b)]);
            h[size_t(b)]=k;
        }
        for(int it=0;it<ncls+2;it++){
            std::vector<uint64_t> nh(size_t(ncls),0ULL);
            for(int b=0;b<ncls;b++){
                uint64_t k=1469598103934665603ULL;hashadd(k,h[size_t(b)]);
                for(int e=0;e<A;e++){
                    int t=cs[size_t(b)][size_t(e)];
                    hashadd(k,t<0?0xFFFFFFFFULL:h[size_t(t)]);
                }
                nh[size_t(b)]=k;
            }
            bool same=true;
            for(int b=0;b<ncls&&same;b++)if(nh[size_t(b)]!=h[size_t(b)])same=false;
            h.swap(nh);
            if(same)break;
        }
        return h;
    }
    void snapshot_hypothesis(const std::vector<int>&part,int ncls){
        prev.valid=true;prev.classes=ncls;prev.blocks=part;
        prev.acc.assign(tree.accepting.begin(),
                        tree.accepting.begin()+long(part.size()));
        prev.adv.assign(tree.advanced.begin(),tree.advanced.begin()+long(part.size()));
        prev.ign.assign(tree.ignored.begin(),tree.ignored.begin()+long(part.size()));
        prev.fat.assign(tree.fatal.begin(),tree.fatal.begin()+long(part.size()));
        prev.succ.assign(tree.succ.begin(),tree.succ.begin()+long(part.size()));
    }
    void repartition(int step){
        std::vector<int> fresh;
        int n=c.merge_rule=="signature"?part.rebuild_signature(tree,fresh,c.sig_support)
             :c.merge_rule=="rpni"?part.rebuild_fringe(tree,c.theta,fresh,false)
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
        if(goal_mode){
            blocks=fresh;classes=n;dirty=false;refresh_aggressive();refresh_true_partition();
            if(c.revision=="jirp")snapshot_hypothesis(blocks,classes);
            return;
        }
        std::vector<double> old;old.swap(Q);Q.assign(size_t(MAXF)*T.N*4,0.0);
        if(c.revision=="jirp"){
            // JIRP transfers values between states of the OLD and NEW hypotheses
            // that behave the same on all future strings. Two things were wrong
            // before. The refinement ran a fixed three rounds, so a difference
            // first visible at length four was invisible. And both signatures
            // were computed from the *current* tree, so the old hypothesis was
            // never really the old one: adding nodes shifted its signatures and
            // the match failed for reasons that had nothing to do with
            // equivalence. The old hypothesis is now snapshotted at each
            // repartition and compared against as it stood.
            //
            // This is still an approximation: a signature fixed point over the
            // hypothesis the agent holds, not a quantification over all future
            // strings of both hypotheses. It is reported as such.
            if(prev.valid){
                std::vector<uint64_t> so=hypothesis_signature(prev.blocks,prev.classes,
                                              prev.adv,prev.ign,prev.fat,prev.succ,prev.acc);
                std::vector<uint64_t> sn=hypothesis_signature(fresh,n,
                                              tree.advanced,tree.ignored,tree.fatal,
                                              tree.succ,tree.accepting);
                std::unordered_map<uint64_t,int> once;
                for(int b=0;b<prev.classes;b++){
                    auto it=once.find(so[size_t(b)]);
                    if(it==once.end())once[so[size_t(b)]]=b;else it->second=-1;
                }
                for(int b=0;b<n;b++){
                    auto it=once.find(sn[size_t(b)]);
                    if(it==once.end()||it->second<0)continue;
                    int src=it->second;
                    if(src<MAXF&&b<MAXF){
                        std::copy(old.begin()+size_t(src)*T.N*4,old.begin()+size_t(src+1)*T.N*4,
                                  Q.begin()+size_t(b)*T.N*4);
                        jirp_transfers++;
                    }
                }
            }
        }
        if(c.revision=="transfer"){
            std::vector<int> source(size_t(n),-1);
            for(int i=int(fresh.size())-1;i>=0;i--)if(i<int(blocks.size()))source[fresh[i]]=blocks[i];
            for(int b=0;b<n;b++)if(source[b]>=0&&source[b]<MAXF)
                std::copy(old.begin()+size_t(source[b])*T.N*4,old.begin()+size_t(source[b]+1)*T.N*4,
                          Q.begin()+size_t(b)*T.N*4);
        }
        blocks=fresh;classes=n;dirty=false;refresh_aggressive();refresh_true_partition();
        if(c.revision=="jirp")snapshot_hypothesis(blocks,classes);
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
    void train_one_skill(int q,int g,int s,int a,int ns,int event){
        bool hit=event==g;
        size_t base=gidx(q,g,ns,0);
        double best_next=*std::max_element(G.begin()+base,G.begin()+base+4);
        double target=double(hit)-c.cost+(hit?0:c.gamma*best_next);
        double&v=G[gidx(q,g,s,a)];v+=c.alpha*(target-v);skill_updates++;
    }
    void train_skills(int q,int s,int a,int ns,int event){
        for(int g=0;g<T.alphabet;g++){
            bool hit=event==g;
            size_t base=gidx(q,g,ns,0);
            double best_next=*std::max_element(G.begin()+base,G.begin()+base+4);
            double target=double(hit)-c.cost+(hit?0:c.gamma*best_next);
            double&v=G[gidx(q,g,s,a)];v+=c.alpha*(target-v);skill_updates++;
        }
    }
    int skill_action(int q,int g,int s)const{
        int a=0;for(int b=1;b<4;b++)if(G[gidx(q,g,s,b)]>G[gidx(q,g,s,a)])a=b;return a;
    }
    // --- the learned router -------------------------------------------------
    // Semi-MDP Q-learning over goal skills. One decision opens an option, the
    // option runs until its goal event fires or the cap is reached, and the
    // return over that stretch is what the decision is judged by. The state is
    // the history feature plus the cell, which is exactly what the flat arm
    // indexes its own table by, so no task-state information reaches it.
    double meta_best(int f,int s)const{
        // Over the goals the router may actually take. A forbidden goal keeps
        // its initial zero, and every real option earns a negative step cost, so
        // maxing over the whole alphabet pins the bootstrap at zero everywhere
        // and the value function learns nothing at all.
        size_t b=(size_t(f)*T.N+size_t(s))*size_t(T.alphabet);
        double best=0;bool any=false;
        for(int e=0;e<T.alphabet;e++){
            if(!meta_allowed(e))continue;
            if(!any||M[b+size_t(e)]>best){best=M[b+size_t(e)];any=true;}
        }
        return any?best:*std::max_element(M.begin()+b,M.begin()+b+T.alphabet);
    }
    // Which goals the router may propose. Normally all of them: not being told
    // what is safe is part of not being told the task structure.
    bool meta_allowed(int e)const{
        if(!c.meta_safe)return true;
        for(int q=0;q<T.nstates;q++)
            if(!T.failing[q]&&T.failing[T.next_state(q,e)])return false;
        return true;
    }
    void meta_open(int f,int s,uint64_t&stream){
        stream=mix(stream);
        int ok[64],nk=0;
        for(int e=0;e<T.alphabet;e++)if(meta_allowed(e))ok[nk++]=e;
        if(!nk){for(int e=0;e<T.alphabet;e++)ok[nk++]=e;}
        int g;
        if(((stream>>11)*0x1.0p-53)<c.epsilon)g=ok[mix(stream+91)%uint64_t(nk)];
        else{
            size_t b=(size_t(f)*T.N+size_t(s))*size_t(T.alphabet);
            double v=M[b+size_t(ok[0])];
            for(int i=1;i<nk;i++)v=std::max(v,M[b+size_t(ok[i])]);
            int cand[64],k=0;
            for(int i=0;i<nk;i++)if(M[b+size_t(ok[i])]==v)cand[k++]=ok[i];
            g=cand[mix(stream+92)%uint64_t(k)];
        }
        meta_goal=g;meta_f=f;meta_s=s;meta_k=0;meta_R=0.0;options++;
    }
    void meta_close(int f2,int s2,bool terminal){
        if(meta_goal<0)return;
        double boot=terminal?0.0:std::pow(c.gamma,double(meta_k))*meta_best(f2,s2);
        size_t i=(size_t(meta_f)*T.N+size_t(meta_s))*size_t(T.alphabet)+size_t(meta_goal);
        M[i]+=c.alpha*(meta_R+boot-M[i]);
        meta_goal=-1;
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
        if(x.predead){apply(x);return;}
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
        // has some, and the evidence it has is the class's, not one member's.
        // Reading a single representative node made a class blind to events its
        // other members had seen, which mixed a missing-evidence effect into
        // what was reported as a wrong-model effect.
        //
        // The transition the agent actually took is never left to the model: for
        // its own class the real outcome is used, so switching this rule on can
        // only add updates, never replace a true one with a guess.
        // The class-level tables are rebuilt at repartitions and the tree grows
        // between them, so a node discovered since the last rebuild has no class
        // evidence yet. Those nodes are simply not available as counterfactual
        // sources; the real transition below is unaffected.
        int known=int(std::min(blocks.size(),part.parent.size()));
        if(classes<=0||known<=0){apply(x);return;}
        int actual=(x.node>=0&&x.node<int(blocks.size()))?blocks[size_t(x.node)]:-1;
        if(actual>=classes)actual=-1;
        std::vector<int> rep(size_t(classes),-1);
        for(int nd=0;nd<known;nd++){
            int b=blocks[size_t(nd)];
            if(b>=0&&b<classes&&rep[size_t(b)]<0)rep[size_t(b)]=part.find(nd);
        }
        if(actual<0)apply(x);
        for(int b=0;b<classes;b++){
            int u=rep[size_t(b)];if(u<0)continue;
            int fj=-1;bool win=false,terminal=false;
            if(b==actual){
                int child=x.node;
                if(x.post>x.pre&&x.node>=0&&x.node<int(tree.succ.size())&&ev>=0
                   &&tree.succ[size_t(x.node)][size_t(ev)]>=0)
                    child=tree.succ[size_t(x.node)][size_t(ev)];
                fj=(child>=0&&child<int(blocks.size()))?blocks[size_t(child)]:b;
                if(fj>=classes)fj=b;
                win=x.win;terminal=x.win||x.dead;
            }else if(ev<0){fj=b;}
            else if(part.adv[size_t(u)]&(1ULL<<ev)){
                int child=part.succ[size_t(u)][size_t(ev)];
                if(child<0||child>=known)continue;
                fj=blocks[size_t(child)];
                if(fj<0||fj>=classes)continue;
                win=tree.accepting[size_t(child)]>0;terminal=win;
            }else if(part.ign[size_t(u)]&(1ULL<<ev)){fj=b;}
            else if(part.fat[size_t(u)]&(1ULL<<ev)){fj=b;terminal=true;}
            else continue;
            double r=win?reward_success:-c.cost;
            double target=r+(terminal?0:c.gamma*best(fj,x.ns));
            double&v=Q[(size_t(b)*T.N+x.s)*4+x.a];v+=c.alpha*(target-v);updates++;
        }
        return;
    }
    void apply(const Raw&x){
        if(goal_mode)return;
        if(x.predead){
            // Absorbing, and it keeps costing. No shaping, no bootstrap from a
            // healthy row, and no evidence written anywhere.
            double target=-c.cost+c.gamma*best(DEADF,x.ns);
            double&dv=Q[(size_t(DEADF)*T.N+x.s)*4+x.a];dv+=c.alpha*(target-dv);updates++;
            return;
        }
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
        else if(k=="--skill-key")c.skill_key=v;
        else if(k=="--route")c.route=v;
        else if(k=="--option-cap")c.option_cap=std::stoi(v);
        else if(k=="--meta-safe")c.meta_safe=std::stoi(v)!=0;
        else if(k=="--crm")c.crm=std::stoi(v)!=0;
        else if(k=="--relabel")c.relabel=std::stoi(v)!=0;
        else if(k=="--replay")c.replay=std::stoi(v);
        else if(k=="--merge-rule")c.merge_rule=v;
        else if(k=="--sig-support")c.sig_support=std::stoi(v);
        else if(k=="--source-frequency")c.source_frequency=std::stoi(v)!=0;
        else if(k=="--verify-budget")c.verify_budget=std::stoll(v);
        else if(k=="--verify-policy")c.verify_policy=v;
        else if(k=="--verify-fallback")c.verify_fallback=v;
        else if(k=="--verify-rate")c.verify_rate=std::stod(v);
        else if(k=="--verify-partition")c.verify_partition=v;
        else if(k=="--verify-support")c.verify_support=std::stoi(v);
        else if(k=="--cache-reuse")c.cache_reuse=std::stoi(v)!=0;
        else if(k=="--cache-key")c.cache_key=v;
        else if(k=="--dump-queries")c.dump_queries=std::stoi(v)!=0;
        else if(k=="--history")c.history_source=v;
        else if(k=="--window-pure")c.window_pure=std::stoi(v)!=0;
        else if(k=="--reward-success")c.reward_success_override=std::stod(v);
        else throw std::runtime_error("unknown flag "+k);
    }
    Task T;T.load(c.task);
    Learner L(T,c);
    if(!c.seed_structure.empty()){
        L.tracking=true;
        int longest=seed_tree(L.tree,T,c.seed_structure,L.blocks,c.source_frequency);
        for(int i=0;i<L.tree.size()&&!L.any_fatal;i++)
            for(int e=0;e<T.alphabet;e++)if(L.tree.fatal[i][e]){L.any_fatal=true;break;}
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

    bool in_sink=false;
    auto reset=[&](int index){which=index%S;s=T.starts[which];qstate=0;t=0;node=0;in_sink=false;
        L.episode_path.clear();L.node_path.assign(1,0);
        episodes.emplace_back();raw_episodes.emplace_back();
        episode=int(episodes.size())-1;};

    auto rollout=[&](int start)->int{
        int cs=T.starts[start],cq=0;std::vector<int> h;
        std::vector<int> rh;int cn=0;
        // The learned router is an option policy: it commits at the moment of
        // choice and holds until the goal event fires. Re-deciding every step
        // would read value rows the training never updated, because options are
        // only ever opened where the previous one ended.
        int held=-1,held_k=0;
        for(int step=0;step<T.horizon;step++){
            int a;
            if(L.goal_mode){
                int g;
                if(L.meta_route){
                    if(held<0||held_k>=c.option_cap){
                        // The frozen policy has to respect the same goal set the
                        // training used. A goal the router was never allowed to
                        // take keeps its initial value of zero, and since every
                        // real option earns a step cost, zero ends up being the
                        // argmax nearly everywhere — so evaluating over the full
                        // alphabet sends the agent straight at the hazard.
                        int f=L.lookup(h,int(h.size()));
                        held=-1;
                        if(f>=0){
                            size_t b=(size_t(f)*L.T.N+size_t(cs))*size_t(L.T.alphabet);
                            for(int e=0;e<L.T.alphabet;e++){
                                if(!L.meta_allowed(e))continue;
                                if(held<0||L.M[b+size_t(e)]>L.M[b+size_t(held)])held=e;
                            }
                        }
                        if(held<0)for(int e=0;e<L.T.alphabet;e++)
                            if(L.meta_allowed(e)){held=e;break;}
                        if(held<0)held=0;
                        held_k=0;
                    }
                    g=held;held_k++;
                }else g=L.select_goal(cq,cs,h,cn);
                a=g<0?0:L.skill_action(cq,g,cs);
            }else{
                int f=L.raw_history?L.lookup(rh,int(rh.size())):L.lookup(h,int(h.size()));
                a=L.greedy(f,cs);
            }
            int ns=T.moves[size_t(cs)*4+a],ev=T.events[size_t(cs)*4+a];
            int nq=T.next_state(cq,ev);
            // The frozen policy has to build its history the way the agent
            // does. Under a verification budget that means the agent's own
            // labels, never the task state change it has not paid to see.
            bool adv=(nq!=cq&&!T.failing[nq]);
            if(L.verify_metered())
                adv=(ev>=0&&cn>=0&&cn<int(L.tree.advanced.size())
                     &&L.tree.advanced[cn][ev]>0);
            if(adv&&L.tracking&&cn>=0&&cn<int(L.tree.succ.size()))
                cn=L.tree.succ[cn][ev];
            if(L.verify_metered()?adv:(nq!=cq))h.push_back(ev);
            if(ev>=0)rh.push_back(ev);
            if(L.meta_route&&ev==held)held=-1;      // the option reached its goal
            cs=ns;cq=nq;
            if(T.failing[cq])return -1;
            if(T.accepting[cq])return step+1;
        }
        return -1;
    };

    int first90=-1,stable90=-1,hits=0,last_repartition=0,first_success=-1,first_failure=-1;
    // Diagnostics for the irreversibility arm. How many catastrophes it took to
    // get good is a different question from how fast it got good, and the two
    // come apart exactly when a skill carries safety across task states.
    long long fail_at_90=-1;
    std::vector<long long> goal_deaths(size_t(T.alphabet),0),goal_steps(size_t(T.alphabet),0);
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
                a=L.skill_action(qstate,L.struct_target,s);}
        }
        int active_goal;
        if(L.meta_route&&!in_sink){
            // The option in flight keeps the goal; a fresh decision is taken
            // only when the last one finished.
            if(L.meta_goal<0)L.meta_open(L.feature(hist,int(hist.size())),s,stream);
            active_goal=L.meta_goal;
        }else active_goal=(L.goal_mode&&!in_sink)?L.select_goal(qstate,s,hist,node):-1;
        if(!structural){
            if(roll<c.epsilon)a=int(mix(stream+1)%4);
            else if(L.goal_mode){
                int g=active_goal;
                if(g<0)a=int(mix(stream+2)%4);
                else{
                    size_t base=L.gidx(qstate,g,s,0);
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
        // A step taken after the task has already failed is not evidence about
        // the history that preceded the failure. Before this was separated, the
        // sink being absorbing meant every remaining step of the episode was
        // written as "this event was ignored at that healthy history", counted
        // as another failure, and charged the whole remaining horizon again.
        bool predead=in_sink;
        bool progress=!predead&&nq!=qstate&&!fell;
        if(c.shape=="computedproduct"||c.shape=="guided"){
            if(L.map_next[size_t(s)*4+a]<0)L.dirty=true;
            L.learn_edge(s,a,ns,ev);
        }
        if(L.goal_mode&&!c.relabel){if(active_goal>=0)L.train_one_skill(qstate,active_goal,s,a,ns,ev);}
        else L.train_skills(qstate,s,a,ns,ev);
        if(L.struct_target>=0&&ev==L.struct_target){L.struct_target=-1;L.struct_left=0;}
        // Whether this firing advanced the task is a separate question from
        // which label fired, and under a verification budget it costs one query.
        // A pair never paid for stays *unknown*: nothing is recorded, and the
        // agent falls back to acting as though it did not advance. That guess is
        // the agent's own, which is the point — the history may no longer be
        // filtered by a task-state change the agent cannot see, or switching the
        // feedback off would still leak the structure.
        bool believed=progress;
        if(L.verify_metered()){
            // Where the agent thinks it is, folded from the history it believes,
            // against where the task actually is. Measurement only.
            int implied=0;for(int h2:hist)implied=T.next_state(implied,h2);
            bool desynced=(implied!=qstate);
            if(desynced)L.desync_steps++;
            uint64_t fkey=L.enc.hist_key(rawhist,rpre);
            bool cached=false;
            if(c.cache_key=="truestate"){
                cached=ev>=0&&L.true_labelled(qstate,ev);
            }else if(c.cache_key=="fullprefix"){
                cached=ev>=0&&L.full_labelled(fkey,ev);
            }else if(c.cache_key=="mergedclass"){
                // Storage stays per node; only the *reading* is widened to the
                // whole class, which is precisely "who does this answer serve".
                // The node's own record has to count too: the partition is only
                // rebuilt periodically, so a pair just paid for is not yet in it
                // and the agent would buy the same answer again and again.
                cached=ev>=0&&node>=0&&node<int(L.tree.advanced.size())
                       &&(L.tree.labelled(node,ev)||L.class_verdict(node,ev)>=0);
            }else{
                cached=L.tracking&&ev>=0&&node>=0&&node<int(L.tree.advanced.size())
                       &&L.tree.labelled(node,ev);
            }
            if(cached&&!c.cache_reuse)cached=false;
            bool want=ev>=0&&!predead&&!cached;
            if(want){
                if(c.verify_policy=="random"){
                    stream=mix(stream);
                    want=((stream>>11)*0x1.0p-53)<c.verify_rate;
                }else if(c.verify_policy=="decision"){
                    // Do not spend on confirming something you already act on.
                    // Spend where you do not yet know what to do at all.
                    want=L.node_is_lost(node);
                }else if(c.verify_policy=="aligned"||c.verify_policy=="closed"){
                    // Do not pay for a question whose answer you would act on
                    // identically either way. Skipping means believing "did not
                    // advance", so skip only where the structure says the same;
                    // where it says the event advances, the skip would make the
                    // agent believe the opposite of its own model, which is the
                    // one case worth paying for.
                    want=L.class_verdict(node,ev,c.verify_policy=="closed")!=0;
                }else if(c.verify_policy=="novel"){
                    // Let the structure cover what it can and pay only for
                    // genuine gaps. This is the sharpest form of the project's
                    // original question: does building the automaton reduce how
                    // much the agent has to ask?
                    want=L.class_verdict(node,ev)<0;
                }
            }
            bool pay=want&&L.may_verify();
            if(want&&!L.may_verify())L.declined++;
            if(pay){
                L.verifications++;if(L.verify_left>0)L.verify_left--;
                if(L.answered_true.insert({qstate,ev}).second)L.necessary_queries++;
                else L.redundant_queries++;
                if(c.dump_queries){
                    int verdict=progress?0:(fell?2:1);
                    L.query_log.push_back({std::vector<int>(hist.begin(),hist.begin()+pre),
                                           ev,verdict,s,node});
                }
            }
            if(ev>=0&&!predead&&(cached||pay)){
                if(pay){
                    // What the verification buys: this firing's true verdict.
                    believed=progress;
                    // A write made while desynced is filed under a history that
                    // does not correspond to the task state it describes.
                    if(desynced)L.misattributed++;
                    if(c.cache_key=="truestate"){
                        if(progress)L.true_adv[size_t(qstate)][size_t(ev)]++;
                        else if(fell)L.true_fat[size_t(qstate)][size_t(ev)]++;
                        else L.true_ign[size_t(qstate)][size_t(ev)]++;
                    }else if(c.cache_key=="fullprefix"){
                        if(progress)L.full_slot(L.full_adv,fkey)[size_t(ev)]++;
                        else if(fell)L.full_slot(L.full_fat,fkey)[size_t(ev)]++;
                        else L.full_slot(L.full_ign,fkey)[size_t(ev)]++;
                    }
                    if(L.tracking){
                        if(progress){if(!L.tree.advanced[node][ev]++)L.dirty=true;
                                     L.tree.rank[node][ev]++;}
                        else if(fell){if(!L.tree.fatal[node][ev]++)L.dirty=true;}
                        else if(!L.tree.ignored[node][ev]++)L.dirty=true;
                    }
                }else{
                    // Already paid for at this history: re-observing the same
                    // pair costs nothing and only sharpens the counts.
                    believed=(c.cache_key=="truestate")
                             ?L.true_adv[size_t(qstate)][size_t(ev)]>0
                             :(c.cache_key=="fullprefix")
                             ?L.full_slot(L.full_adv,fkey)[size_t(ev)]>0
                             :(c.cache_key=="mergedclass")
                             ?(L.tree.labelled(node,ev)?L.tree.advanced[node][ev]>0
                                                       :L.class_verdict(node,ev)==1)
                             :L.tree.advanced[node][ev]>0;
                    L.cache_hits++;
                    // Was the cached verdict right for the situation it was used
                    // in? This is the number the audit exists to produce.
                    if(believed!=progress){L.stale_uses++;
                        if(progress&&!believed)L.stale_missed_advance++;}
                    if(L.tracking){
                        if(believed){L.tree.advanced[node][ev]++;L.tree.rank[node][ev]++;}
                        else if(L.tree.fatal[node][ev])L.tree.fatal[node][ev]++;
                        else L.tree.ignored[node][ev]++;
                    }
                }
            }else if(ev>=0&&!predead){
                int g=(c.verify_fallback=="class")?L.class_verdict(node,ev):-1;
                if(g>=0){believed=(g==1);L.generalised_firings++;}
                else{believed=false;L.unknown_firings++;}
            }
        }
        else if(L.tracking&&ev>=0&&!predead&&!progress){
            // Fatal and ignored are different verdicts and are recorded apart.
            if(fell){if(!L.tree.fatal[node][ev]++)L.dirty=true;}
            else if(!L.tree.ignored[node][ev]++)L.dirty=true;
        }
        if(believed){
            if(L.tracking){
                if(!L.verify_metered()){
                    if(!L.tree.advanced[node][ev]++)L.dirty=true;
                    L.tree.rank[node][ev]++;
                }
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
        if(believed){L.deepest=std::max(L.deepest,int(hist.size()));L.total_events++;}
        bool win=T.accepting[nq]!=0;
        bool dead=fell;
        // failures counts entries into the sink, not steps spent inside it.
        if(active_goal>=0&&active_goal<T.alphabet)goal_steps[size_t(active_goal)]++;
        if(dead&&!predead){failures++;L.any_fatal=true;
            if(active_goal>=0&&active_goal<T.alphabet)goal_deaths[size_t(active_goal)]++;
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
        Raw online{s,a,ns,ev,episode,pre,int(hist.size()),rpre,int(rawhist.size()),
                   active_goal,node,qstate,win,dead,predead};
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
        // The waste is the steps actually taken after the task failed, counted
        // once each, not the remaining horizon re-added on every one of them.
        if(predead)doomed_steps++;
        if(dead)in_sink=true;
        bool stop=win||t>=T.horizon||(dead&&!c.doom_continues);
        if(dead&&c.doom_continues)stop=(t>=T.horizon);
        if(dead&&!predead&&first_failure<0)first_failure=step;
        if(L.meta_route&&L.meta_goal>=0){
            // Same reward the flat arm learns from, accumulated over the option.
            double r=win?L.reward_success:-c.cost;
            L.meta_R+=std::pow(c.gamma,double(L.meta_k))*r;
            L.meta_k++;
            bool fired=(ev==L.meta_goal);
            if(fired)L.option_hits++;
            bool terminal=win||dead;
            if(fired||terminal||stop||L.meta_k>=c.option_cap)
                L.meta_close(L.feature(hist,int(hist.size())),s,terminal);
        }
        if(stop){L.meta_goal=-1;reset(++which);}
        if(step%c.every==0){
            int solved=0;double steps=0;
            for(int i=0;i<S;i++){int r=rollout(i);if(r>0){solved++;steps+=r;}}
            double rate=double(solved)/S;
            checks.push_back({double(step),rate,solved?steps/solved:0});
            auc+=rate*c.every;
            if(rate>=.9){if(first90<0){first90=step;fail_at_90=failures;}
                         if(last>=.9&&stable90<0)stable90=step;}
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
        for(int e=0;e<T.alphabet;e++){total++;
            if(L.tree.advanced[i][e]||L.tree.ignored[i][e]||L.tree.fatal[i][e])tried++;}
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
             <<",\"fail_at_90\":"<<fail_at_90<<",\"goal_deaths\":[";
    for(int g=0;g<T.alphabet;g++)std::cout<<(g?",":"")<<goal_deaths[size_t(g)];
    std::cout<<"],\"goal_steps\":[";
    for(int g=0;g<T.alphabet;g++)std::cout<<(g?",":"")<<goal_steps[size_t(g)];
    std::cout<<"]"
             <<",\"options\":"<<L.options<<",\"option_hits\":"<<L.option_hits
             <<",\"skill_updates\":"<<L.skill_updates
             <<",\"seeded_nodes\":"<<(c.seed_structure.empty()?0:L.tree.size())
             <<",\"quota\":"<<c.quota<<",\"risk\":"<<c.risk<<",\"beta\":"<<c.beta
             <<",\"danger_nodes\":"<<[&]{int k=0;for(int v:L.tree.danger)if(v)k++;return k;}()
             <<",\"danger_total\":"<<[&]{long long k=0;for(int v:L.tree.danger)k+=v;return k;}()<<",\"target\":\""<<c.target<<"\",\"shape\":\""<<c.shape
             <<"\",\"shape_scale\":"<<c.shape_scale
             <<",\"minimality_gap\":"<<(reached.empty()?0:double(L.classes)/double(reached.size()))
             <<",\"revision_updates\":"<<L.revision_updates
             <<",\"crm\":"<<(c.crm?1:0)<<",\"relabel\":"<<(c.relabel?1:0)
             <<",\"replay\":"<<c.replay<<",\"merge_rule\":\""<<c.merge_rule<<"\",\"jirp_transfers\":"<<L.jirp_transfers<<",\"classes_aggr\":"<<L.classes_aggr<<",\"rank_decisions\":"<<L.rank_decisions<<",\"rank_changed\":"<<L.rank_changed<<",\"source_frequency\":"<<(c.source_frequency?1:0)<<",\"verifications\":"<<L.verifications<<",\"unknown_firings\":"<<L.unknown_firings<<",\"verify_budget\":"<<c.verify_budget<<",\"generalised_firings\":"<<L.generalised_firings<<",\"declined\":"<<L.declined<<",\"verify_policy\":\""<<c.verify_policy<<"\""<<",\"verify_fallback\":\""<<c.verify_fallback<<"\""<<",\"verify_partition\":\""<<c.verify_partition<<"\""<<",\"verify_support\":"<<c.verify_support<<",\"desync_steps\":"<<L.desync_steps<<",\"misattributed\":"<<L.misattributed<<",\"cache_hits\":"<<L.cache_hits<<",\"stale_uses\":"<<L.stale_uses<<",\"stale_missed_advance\":"<<L.stale_missed_advance<<",\"necessary_queries\":"<<L.necessary_queries<<",\"redundant_queries\":"<<L.redundant_queries<<",\"cache_reuse\":"<<(c.cache_reuse?1:0)<<",\"cache_key\":\""<<c.cache_key<<"\""<<",\"history_source\":\""<<c.history_source<<"\""<<",\"window_pure\":"<<(c.window_pure?1:0)<<",\"theta\":"<<c.theta<<",\"revision\":\""<<c.revision<<"\""
             <<",\"updates\":"<<L.updates
             <<",\"episodes\":"<<episodes.size()<<",\"starts\":"<<S
             <<",\"horizon\":"<<T.horizon<<",\"reward_success\":"<<L.reward_success
             <<",\"trace_hash\":\""<<trace<<"\",\"checkpoints\":[";
    for(size_t i=0;i<checks.size();i++)std::cout<<(i?",":"")<<"["<<int(checks[i][0])<<","<<checks[i][1]<<","<<checks[i][2]<<"]";
    std::cout<<"]";
    if(c.dump_queries){
        std::cout<<",\"queries\":[";
        for(size_t i=0;i<L.query_log.size();i++){
            std::cout<<(i?",":"")<<"{\"h\":[";
            const auto&h=L.query_log[i].h;
            for(size_t j=0;j<h.size();j++)std::cout<<(j?",":"")<<h[j];
            std::cout<<"],\"e\":"<<L.query_log[i].e
                     <<",\"v\":"<<L.query_log[i].v
                     <<",\"c\":"<<L.query_log[i].cell
                     <<",\"n\":"<<L.query_log[i].node<<"}";
        }
        std::cout<<"]";
    }
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
            std::cout<<"],\"fatal\":[";
            for(int e=0;e<T.alphabet;e++)std::cout<<(e?",":"")<<L.tree.fatal[i][e];
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
