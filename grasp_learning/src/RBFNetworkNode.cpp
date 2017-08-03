#include <grasp_learning/RBFNetwork.h>
#include <grasp_learning/power.h>
#include <vector>
int main(int argc, char **argv) {

  ros::init(argc,argv,"RBFNetworkNode");
  
  ros::NodeHandle nh;
  ros::NodeHandle nh_;

  nh_ = ros::NodeHandle("~");

  RBFNetwork network;
  std::vector<double> global_pos {0.0,0.0,0.0};
  network.buildRBFNetwork(5,1,1,1,global_pos);
  Eigen::Vector3d test(1,0,0.666667);

  network.networkOutput(test);

  // power power(5,0,2);
  // std::vector<double> noise_1 = {5,1,1,1,1};
  // std::vector<double> active_kernels_1 = {0,3};
  // double reward_1 = 2;

  // std::vector<double> noise_2 = {2,3,7,4,5};
  // std::vector<double> active_kernels_2 = {0,1,2,3,4};
  // double reward_2 = 4;

  // std::vector<double> noise_3 = {3,1,2,4,7};
  // std::vector<double> active_kernels_3 = {3,4};
  // double reward_3 = 3;

  // std::vector<double> noise_4 = {5,1,1,1,1};
  // std::vector<double> active_kernels_4 = {0,3};
  // double reward_4 = 2;

  // std::vector<double> noise_5 = {5,1,1,1,1};
  // std::vector<double> active_kernels_5 = {0,3};
  // double reward_5 = 2;


  // network.printKernelWeights();
  // power.reweight(network.getKernelWeights(),noise, reward,network.getActiveKernels());
  // power.reweight(network.getKernelWeights(),noise_1, reward_1 ,active_kernels_1);
  // network.printKernelWeights();

  // power.reweight(network.getKernelWeights(),noise_2, reward_2 ,active_kernels_2);
  // network.printKernelWeights();

  // power.reweight(network.getKernelWeights(),noise_3, reward_3 ,active_kernels_3);


  network.printKernelWeights();

  ros::AsyncSpinner spinner(1); // Use 4 threads
  spinner.start();
  ros::waitForShutdown();
  
  return 0;
}
