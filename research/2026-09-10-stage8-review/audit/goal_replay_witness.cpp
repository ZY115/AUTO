#define main archived_project_main
#include "../../../progressive_task_discovery/stage8_compression/src/compress.cpp"
#undef main
int main(){
 Task t;t.load("progressive_task_discovery/stage8_compression/results/depth_vs_graph/gap00.task");
 Config c;c.method="goal";c.goal_select="value";c.relabel=false;
 Learner L(t,c);std::vector<std::vector<int>> traces(1);L.trace=&traces;
 int cand[64];int n=L.allowed(0,cand);assert(n>=2);
 int s=t.starts[0];int g0=cand[0],g1=cand[1];
 L.G[(size_t(g0)*t.N+s)*4]=2;
 int collected=L.select_goal(0,s,traces[0]);assert(collected==g0);
 Raw x{s,0,s,-1,0,0,0,0,0,false,false};
 L.G[(size_t(g1)*t.N+s)*4]=3;
 auto before=L.G;L.skill_replay(x);
 int changed=-1;
 for(int g=0;g<t.alphabet;g++)if(L.G[(size_t(g)*t.N+s)*4]!=before[(size_t(g)*t.N+s)*4])changed=g;
 assert(changed==g1&&changed!=collected);
 std::cout<<"{\"collection_goal\":"<<collected<<",\"replayed_goal_after_values_change\":"<<changed<<",\"same_raw_transition\":true,\"relabel_flag\":false}\n";
}
