#include <grasp_learning/RBFNetwork.h>

GaussianKernel::GaussianKernel(Eigen::Vector3d mean, Eigen::Matrix3d covar){
	this->mean = mean;
	this->covar = covar;
}
Eigen::Vector3d GaussianKernel::residual(Eigen::Vector3d x){

	// std::cout<<"x: "<<x<<std::endl;
	// std::cout<<"mean: "<<mean<<std::endl;
	return x-mean;
}

double GaussianKernel::kernelActivation(Eigen::Vector3d x){
	Eigen::Vector3d residual = this->residual(x);
	double activation = exp(-0.5*residual.transpose()*covar.inverse()*residual);
	// std::cout<<"Activation "<<activation<<std::endl;
	// if (activation>ACTIVATION_THRESHOLD){
	// 	return activation;
	// }
	// else{
	// 	return 0;
	// }
	return activation;
}

Eigen::Vector3d GaussianKernel::getMean(){
	return mean;
}

RBFNetwork::RBFNetwork(){

	print_kernel_means_srv_ = nh.advertiseService("print_kernel_means", &RBFNetwork::printAllKernelMeans, this);
	add_weight_noise_srv_ = nh.advertiseService("add_weight_noise", &RBFNetwork::addWeightNoise, this);
	print_kernel_weights_srv_ = nh.advertiseService("print_kernel_weights", &RBFNetwork::printKernelWeights, this);
	build_RBF_network_srv_ = nh.advertiseService("build_RBFNetwork", &RBFNetwork::buildRBFNetwork, this);
	network_output_srv_ = nh.advertiseService("network_output", &RBFNetwork::networkOutput, this);
	policy_search_srv_ = nh.advertiseService("policy_search", &RBFNetwork::policySearch, this);
	get_network_weights_srv_ = nh.advertiseService("get_network_weights", &RBFNetwork::getNetworkWeights, this);
	get_running_weights_srv_ = nh.advertiseService("get_running_weights", &RBFNetwork::getRunningWeights, this);

	
	std::normal_distribution<double> d2(0,0.05);
	dist.param(d2.param());
	ROS_INFO("Set up all services");
}


bool RBFNetwork::buildRBFNetwork(grasp_learning::SetRBFN::Request& req, grasp_learning::SetRBFN::Response& res){
	ROS_INFO("Building the RBF Network");
	// std::cout<<req.numKernels<<std::endl;
	// std::cout<<req.numRows<<std::endl;
	// std::cout<<req.height<<std::endl;
	// std::cout<<req.radius<<std::endl;
	// std::cout<<"["<<req.globalPos[0]<<", "<<req.globalPos[1]<<", "<<req.globalPos[2]<<"]"<<std::endl;

	numKernels = (int)req.numKernels;
	double column_spacing = (2*PI)/numKernels;
	double row_spacing = req.height/(req.numRows+1.0);
	Eigen::Vector3d mean;
	Eigen::Matrix3d covar;
	covar<<0.1,0,0,
	0,0.1,0,
	0,0,0.1;
	for (int row = 1;row<=req.numRows;row++){
		mean(2) = req.globalPos[2]+row_spacing*row;
		for (int column=0;column<numKernels;column++){
			mean(0)= req.globalPos[0]+req.radius*cos(column*column_spacing);
			mean(1)= req.globalPos[1]+req.radius*sin(column*column_spacing);
			GaussianKernel kernel(mean, covar);
			Network.push_back(kernel);
			weights.push_back(0);
		}
	}
	runningWeights = weights;
	ROS_INFO("%d kernels were created", numKernels);
	PoWER.setParams(numKernels, 8, 5);
	return true;
}

bool RBFNetwork::networkOutput(grasp_learning::CallRBFN::Request& req, grasp_learning::CallRBFN::Response& res){
	Eigen::Vector3d x(req.pos[0], req.pos[1], req.pos[2]);
	activeKernels.clear();
	double output = 0;
	for (int i = 0; i < Network.size(); i++) {
		double activation = Network[i].kernelActivation(x);
		if (activation != 0){
			if (std::find(activeKernels.begin(), activeKernels.end(), i) == activeKernels.end())
				activeKernels.push_back(i);
		}
		output += runningWeights[i]*activation;
	}
    // std::cout<<"Network output: "<<output<<std::endl;
    // for (int i = 0; i < activeKernels.size(); i++) {
    // 	std::cout<<activeKernels[i]<<std::endl;
    // }
	res.result = output;
	return true;
}

std::vector<double> RBFNetwork::getKernelWeights(){
	return weights;
}

bool RBFNetwork::printKernelWeights(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response){
	for (auto iter:runningWeights){
		std::cout << iter << std::endl;
	}
	return true;
}

std::vector<double> RBFNetwork::getActiveKernels(){
	return activeKernels;
}

bool RBFNetwork::addWeightNoise(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response){
	resetRunningWeights();
	rollout_noise.clear();
	rollout_noise = sampleNoise();
	for(int i=0;i<rollout_noise.size();i++){
		runningWeights[i]+=rollout_noise[i];
	}
	return true;
}

std::vector<double> RBFNetwork::sampleNoise(){
	std::vector<double> noise;
	double sample = 0;
	for (int i = 0;i<numKernels;i++){
		sample = dist(generator);
		noise.push_back(sample);
	}
	return noise;
}

void RBFNetwork::resetRunningWeights(){
	runningWeights = weights;
}

void RBFNetwork::updateWeights(std::vector<double> newWeights){
	for(int i=0;i<newWeights.size();i++){
		weights[i]+=newWeights[i];
	}
}

bool RBFNetwork::printAllKernelMeans(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response){
	for(int i = 0;i<Network.size();i++){
		std::cout<<"Kernel: "<<i<<" mean "<<Network[i].getMean()<<std::endl;
	}
	return true;
}

void RBFNetwork::printKernelMean(int kernel){
	std::cout<<"Kernel: "<<kernel<<std::endl<<Network[kernel].getMean()<<std::endl;
}

bool RBFNetwork::policySearch(grasp_learning::PolicySearch::Request& req, grasp_learning::PolicySearch::Response& res){
	std::vector<double> updatedWeights = PoWER.policySearch(rollout_noise, req.reward, getActiveKernels());
	updateWeights(updatedWeights);
	// printVector(updatedWeights);
	return true;
}

bool RBFNetwork::getNetworkWeights(grasp_learning::GetNetworkWeights::Request& req, grasp_learning::GetNetworkWeights::Response& res){
	res.weights = weights;
	return true;
}

bool RBFNetwork::getRunningWeights(grasp_learning::GetNetworkWeights::Request& req, grasp_learning::GetNetworkWeights::Response& res){
	res.weights = runningWeights;
	return true;
}


void RBFNetwork::printVector(std::vector<double> vec){
	for(auto& iter: vec){
		std::cout<<iter<<std::endl;
	}
}



int main(int argc, char** argv) {
	ros::init(argc, argv, "RBFNetwork");

	RBFNetwork RBFNetwork;

	ROS_INFO("RBF network node ready");
  ros::AsyncSpinner spinner(4);  // Use 4 threads
  spinner.start();
  ros::waitForShutdown();

  return 0;
}
