#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
uint64_t mix(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
void hashadd(uint64_t&h,uint64_t x){h^=mix(x);h*=1099511628211ULL;}
struct Config{string file,method="H";int seed=0,budget=200000,every=2000,window=1,fixed=0;double gamma=.99,ratio=8,epsilon=.02;};
struct Map{int N,n,m,S,L,G,K;vector<int> suffix,starts,labels;vector<array<int,4>> next,event;
 explicit Map(const string&file){ifstream f(file);if(!f)throw runtime_error("map missing");f>>N>>n>>m>>S>>L;G=n+3;K=n+m;suffix.resize(m);starts.resize(S);labels.resize(N);next.resize(N);event.resize(N);for(int&x:suffix)f>>x;for(int&x:starts)f>>x;for(int s=0;s<N;s++){f>>labels[s];for(int a=0;a<4;a++)f>>next[s][a]>>event[s][a];}if(!f)throw runtime_error("bad map");}
 int terminal(bool fixed)const{return fixed?K:(1<<n)-1+m;}
 int advance(int q,int e,bool fixed)const{if(fixed){if(q<K&&e==(q<n?q:suffix[q-n]))return q+1;return q;}int full=(1<<n)-1;if(q<full)return e>=0&&e<n?q|(1<<e):q;int j=q-full;return j<m&&e==suffix[j]?q+1:q;}
 int depth(int q,bool fixed)const{return fixed?q:(q<(1<<n)-1?__builtin_popcount(unsigned(q)):n+q-((1<<n)-1));}
};
struct Obs{int ns,event;bool p,win,done;double reward;};
struct Env{const Map&m;const Config&c;int s,q=0,t=0,H,context=0;
 Env(const Map&m,const Config&c):m(m),c(c),H(24*m.K){reset(0);}
 void reset(int b){context=b;s=m.starts[b];q=0;t=0;}
 Obs step(int a){int ns=m.next[s][a],e=m.event[s][a],nq=m.advance(q,e,c.fixed);bool p=nq!=q;q=nq;s=ns;t++;bool win=q==m.terminal(c.fixed);return{ns,e,p,win,win||t>=H,win?c.ratio*.01*m.L:-.01};}
};
struct Node{int parent,label,depth;bool accept=false;array<int,7> edges;long long visits=0,evidence=0;Node(int p=-1,int l=-1,int d=0):parent(p),label(l),depth(d){edges.fill(-1);}};
struct Raw{int s,a,ns,event,episode,time;bool progress,win;double reward;};
struct Agent{const Map&m;Config c;vector<Node> nodes;vector<double> Q,G;vector<Raw> buffer;vector<vector<pair<int,int>>> events{{}};map<uint64_t,int> window_ids;int cur=0,episode=0,time=0;long long updates=0,version=0;bool skills;
 Agent(const Map&m,const Config&c):m(m),c(c),G(m.G*m.N*4,0),skills(c.method=="PS"||c.method=="OS"){nodes.emplace_back();ensureQ(0);}
 void ensureQ(int f){if(f>=0&&Q.size()<size_t((f+1)*m.N*4))Q.resize((f+1)*m.N*4,0);}
 vector<int> history(int node)const{vector<int> h;while(node>0){h.push_back(nodes[node].label);node=nodes[node].parent;}reverse(h.begin(),h.end());return h;}
 int feature(int node,bool create=false){if(node<0)return -1;if(c.method=="F0")return 0;if(c.method=="F1")return nodes[node].depth;
  if(c.method=="O"||c.method=="OS"){int q=0;for(int e:history(node))q=m.advance(q,e,c.fixed);return q;}
  if(c.method=="F2"){uint64_t key=nodes[node].depth;for(int e:history(node)){(void)e;}vector<int> h=history(node);for(int i=max(0,int(h.size())-c.window);i<int(h.size());i++)key=key*8+h[i]+1;auto it=window_ids.find(key);if(it!=window_ids.end())return it->second;if(!create)return -1;int id=window_ids.size();window_ids[key]=id;return id;}
  return node;
 }
 double qval(int f,int s,int a)const{size_t i=size_t(f<0?0:f)*m.N*4+s*4+a;return f<0||i>=Q.size()?0:Q[i];}
 double best(int f,int s)const{double v=qval(f,s,0);for(int a=1;a<4;a++)v=max(v,qval(f,s,a));return v;}
 vector<int> goals(int node)const{vector<int> out;if(node<0)return out;for(int g=0;g<m.G;g++)if(nodes[node].edges[g]>=0)out.push_back(g);return out;}
 int goal(int node,int s){if(!skills)return -1;vector<int> gs;
  if(c.method=="OS"&&node>=0){int f=feature(node);for(int i=0;i<int(nodes.size());i++)if(feature(i)==f)for(int g:goals(i))gs.push_back(g);sort(gs.begin(),gs.end());gs.erase(unique(gs.begin(),gs.end()),gs.end());}
  else gs=goals(node);
  int g=-1;double bv=-1e99;for(int x:gs){int at=(x*m.N+s)*4;double v=*max_element(G.begin()+at,G.begin()+at+4);if(v>bv){bv=v;g=x;}}return g;
 }
 double value(int node,int s,int a){int g=goal(node,s);return g<0?qval(feature(node),s,a):G[(g*m.N+s)*4+a];}
 int greedy(int node,int s){int a=0;for(int b=1;b<4;b++)if(value(node,s,b)>value(node,s,a))a=b;return a;}
 int act(int s,mt19937_64&rng){double er=(rng()>>11)*0x1.0p-53;uint64_t ra=rng(),tie=rng();if(er<c.epsilon)return ra%4;double v=value(cur,s,greedy(cur,s));int as[4],n=0;for(int a=0;a<4;a++)if(value(cur,s,a)==v)as[n++]=a;return as[tie%n];}
 pair<int,int> parse(const Raw&r)const{int before=0,after=0;for(auto [t,e]:events[r.episode]){if(t<r.time)before=nodes[before].edges[e];if(t<=r.time)after=nodes[after].edges[e];else break;}assert(before>=0&&after>=0);return{before,after};}
 void task_update(const Raw&r,int oldnode,int nextnode){int f=feature(oldnode,true),nf=feature(nextnode,true);ensureQ(max(f,nf));double target=r.reward+(r.win?0:c.gamma*best(nf,r.ns));double&v=Q[(f*m.N+r.s)*4+r.a];v+=.3*(target-v);updates++;}
 void observe(int s,int a,const Obs&o,int step){int old=cur;nodes[old].visits++;nodes[old].evidence+=o.event>=0;time++;
  if(o.p){events[episode].push_back({time,o.event});if(nodes[old].edges[o.event]<0){int dest=nodes.size();nodes[old].edges[o.event]=dest;nodes.emplace_back(old,o.event,nodes[old].depth+1);version++;}cur=nodes[old].edges[o.event];if(o.win)nodes[cur].accept=true;}
  Raw raw{s,a,o.ns,o.event,episode,time,o.p,o.win,o.reward};task_update(raw,old,cur);
  if(buffer.size()<10000)buffer.push_back(raw);else buffer[(step-1)%10000]=raw;
  for(int j=0;j<7;j++){
   if(skills&&j<m.G){int at=(j*m.N+o.ns)*4;double v=*max_element(G.begin()+at,G.begin()+at+4);bool hit=o.event==j;double target=double(hit)-.01+(hit?0:c.gamma*v);double&x=G[(j*m.N+s)*4+a];x+=.3*(target-x);updates++;}
   else{auto&r=buffer[mix(uint64_t(step)*7+j+mix(c.seed+333))%buffer.size()];auto [before,after]=parse(r);task_update(r,before,after);}
  }
  if(o.done){cur=0;episode++;time=0;events.emplace_back();}
 }
};
struct Eval{double success;vector<int> wins,steps;};
Eval evaluate(Agent&a,const Map&m,const Config&c){Eval out{};Env e(m,c);for(int b=0;b<m.S;b++){e.reset(b);int node=0;bool win=false;for(int t=0;t<e.H;t++){auto o=e.step(a.greedy(node,e.s));if(o.p)node=node>=0?a.nodes[node].edges[o.event]:-1;if(o.win){win=true;break;}}out.wins.push_back(win);out.steps.push_back(e.t);out.success+=double(win)/m.S;}return out;}
template<class T>void arr(const T&xs){cout<<"[";bool first=true;for(auto x:xs){if(!first)cout<<",";first=false;cout<<x;}cout<<"]";}
int main(int argc,char**argv){Config c;for(int i=1;i<argc;i+=2){if(i+1>=argc)throw runtime_error("missing arg");string k=argv[i],v=argv[i+1];if(k=="--map")c.file=v;else if(k=="--method")c.method=v;else if(k=="--seed")c.seed=stoi(v);else if(k=="--budget")c.budget=stoi(v);else if(k=="--every")c.every=stoi(v);else if(k=="--window")c.window=stoi(v);else if(k=="--fixed")c.fixed=stoi(v);else if(k=="--gamma")c.gamma=stod(v);else if(k=="--ratio")c.ratio=stod(v);else if(k=="--epsilon")c.epsilon=stod(v);else throw runtime_error("unknown arg");}
 if(!set<string>{"F0","F1","F2","H","P","O","PS","OS"}.count(c.method)||c.budget<=0||c.every<=0)throw runtime_error("invalid config");
 Map m(c.file);Agent a(m,c);Env e(m,c);a.feature(0,true);a.ensureQ(a.feature(0));mt19937_64 rng(mix(c.seed+222));vector<array<double,3>> checks;auto ev=evaluate(a,m,c);checks.push_back({0,ev.success,0});uint64_t trace=1469598103934665603ULL;int episodes=0;long long prefix=0,frontier=0,binding=0,wrong=0;vector<int> first_actual(m.S,-1);map<int,long long> visits;double reward=0;
 for(int t=1;t<=c.budget;t++){int s=e.s;visits[a.cur]++;int g=a.goal(a.cur,s);if(a.goals(a.cur).empty())frontier++;else prefix++;if(g>=0){binding++;if(m.advance(e.q,g,c.fixed)==e.q)wrong++;}int action=a.act(s,rng);auto o=e.step(action);hashadd(trace,s+m.N*(action+4*(int(o.p)+2*m.depth(e.q,c.fixed))));reward+=o.reward;if(o.win&&first_actual[e.context]<0)first_actual[e.context]=t;a.observe(s,action,o,t);if(o.done){episodes++;e.reset(episodes%m.S);}if(t%c.every==0||t==c.budget){ev=evaluate(a,m,c);checks.push_back({double(t),ev.success,double(a.nodes.size())});}}
 uint64_t qhash=1469598103934665603ULL;for(double v:a.Q){uint64_t bits;memcpy(&bits,&v,8);hashadd(qhash,bits);}for(double v:a.G){uint64_t bits;memcpy(&bits,&v,8);hashadd(qhash,bits);}double auc=0;int first=-1;for(size_t i=1;i<checks.size();i++){auc+=(checks[i][0]-checks[i-1][0])*(checks[i][1]+checks[i-1][1])/2/c.budget;if(checks[i][1]>=.9-1e-12&&first<0)first=checks[i][0];}
 map<int,long long> feature_visits;for(auto [node,n]:visits)feature_visits[a.feature(node)]+=n;double concentration=0,entropy=0;for(auto [f,n]:feature_visits){double p=double(n)/c.budget;concentration+=p*p;entropy-=p*log(p);}
 cout<<setprecision(12)<<"{\"method\":\""<<c.method<<"\",\"seed\":"<<c.seed<<",\"n\":"<<m.n<<",\"m\":"<<m.m<<",\"window\":"<<c.window<<",\"fixed\":"<<c.fixed<<",\"budget\":"<<c.budget<<",\"gamma\":"<<c.gamma<<",\"ratio\":"<<c.ratio<<",\"epsilon\":"<<c.epsilon<<",\"horizon\":"<<e.H<<",\"starts\":"<<m.S<<",\"first90\":"<<first<<",\"auc\":"<<auc<<",\"final_success\":"<<ev.success<<",\"updates\":"<<a.updates<<",\"prefix_steps\":"<<prefix<<",\"frontier_steps\":"<<frontier<<",\"binding_steps\":"<<binding<<",\"wrong_binding_steps\":"<<wrong<<",\"cumulative_base_reward\":"<<reward<<",\"structure_versions\":"<<a.version<<",\"visited_histories\":"<<visits.size()<<",\"visited_features\":"<<feature_visits.size()<<",\"effective_states_simpson\":"<<1/concentration<<",\"effective_states_entropy\":"<<exp(entropy)<<",\"q_table_bytes\":"<<(a.Q.size()+a.G.size())*8<<",\"trace_hash\":\""<<trace<<"\",\"q_hash\":\""<<qhash<<"\",\"first_actual\":";arr(first_actual);cout<<",\"final_wins\":";arr(ev.wins);cout<<",\"final_steps\":";arr(ev.steps);cout<<",\"checkpoints\":[";for(size_t i=0;i<checks.size();i++){if(i)cout<<",";arr(checks[i]);}cout<<"],\"nodes\":[";for(size_t i=0;i<a.nodes.size();i++){if(i)cout<<",";cout<<"["<<a.nodes[i].parent<<","<<a.nodes[i].label<<","<<a.nodes[i].depth<<","<<a.nodes[i].accept<<"]";}cout<<"],\"feature_visit_counts\":[";bool firstrow=true;for(auto [f,n]:feature_visits){if(!firstrow)cout<<",";firstrow=false;cout<<"["<<f<<","<<n<<"]";}cout<<"],\"policy\":[";for(int node=0;node<int(a.nodes.size());node++){if(node)cout<<",";cout<<"[";for(int s=0;s<m.N;s++){if(s)cout<<",";cout<<a.greedy(node,s);}cout<<"]";}cout<<"]}\n";
}
