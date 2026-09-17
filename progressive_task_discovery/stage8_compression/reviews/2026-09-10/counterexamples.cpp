#include <array>
#define main original_main
#include "/Users/yuhang/Downloads/why TL/progressive_task_discovery/stage8_compression/src/compress.cpp"
#undef main
int main(){
 Tree t(2);t.add(1,1);t.add(2,2);for(int i=0;i<3;i++){t.visits[i]=400;t.ignored[i][1]=200;}t.advanced[1][0]=200;t.ignored[2][0]=200;
 Partition p;std::vector<int>b;p.rebuild(t,200,b);
 std::cout<<"representative_probe conflict="<<t.conflict(1,2)<<" merged="<<(b[1]==b[2])<<"\n";
 Tree u(2);u.add(1,1);u.add(2,2);u.add(3,3);
 for(int i=0;i<2;i++){u.visits[i]=2;u.advanced[i][0]=1;u.ignored[i][1]=1;}
 u.succ[0][0]=2;u.succ[1][0]=3;
 Partition p2;p2.quota=1;p2.rebuild(u,1,b);
 std::cout<<"propagation_probe children_meet_quota="<<(u.quota_met(2,1)&&u.quota_met(3,1))<<" children_merged="<<(b[2]==b[3])<<"\n";
 Task task;task.load("/Users/yuhang/Downloads/why TL/progressive_task_discovery/stage8_compression/results/family_maps/rung_1.3.task");Config c;c.method="history";c.shape="learned";c.shape_scale=.1;Learner l(task,c);std::vector<int>h={0,1};
 std::cout<<"potential_probe deepest="<<l.deepest<<" phi0="<<l.potential(h,0)<<" phi1="<<l.potential(h,1)<<" phi2="<<l.potential(h,2)<<"\n";
}
