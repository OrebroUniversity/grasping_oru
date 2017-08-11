#include <grasp_learning/RBFNetwork.h>
namespace demo_learning {
	namespace RBFNetwork {

		GaussianKernel::GaussianKernel(Eigen::VectorXd mean, Eigen::MatrixXd covar){
			this->mean = mean;
			this->covar = covar;
		}

		Eigen::VectorXd GaussianKernel::residual(Eigen::VectorXd x){

	// std::cout<<"x: "<<x<<std::endl;
	// std::cout<<"mean: "<<mean<<std::endl;
			return x-mean;
		}

		double GaussianKernel::kernelActivation(Eigen::VectorXd x){
			Eigen::VectorXd residual = this->residual(x);
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

		Eigen::VectorXd GaussianKernel::getMean(){
			return mean;
		}

		RBFNetwork::RBFNetwork(){

			print_kernel_means_srv_ = nh.advertiseService("RBFNetwork/print_kernel_means", &RBFNetwork::printAllKernelMeans, this);
			add_weight_noise_srv_ = nh.advertiseService("RBFNetwork/add_weight_noise", &RBFNetwork::addWeightNoise, this);
			print_kernel_weights_srv_ = nh.advertiseService("RBFNetwork/print_kernel_weights", &RBFNetwork::printKernelWeights, this);
			build_RBF_network_srv_ = nh.advertiseService("RBFNetwork/build_RBFNetwork", &RBFNetwork::buildRBFNetwork, this);
			network_output_srv_ = nh.advertiseService("RBFNetwork/network_output", &RBFNetwork::networkOutput, this);
			policy_search_srv_ = nh.advertiseService("RBFNetwork/policy_search", &RBFNetwork::policySearch, this);
			get_network_weights_srv_ = nh.advertiseService("RBFNetwork/get_network_weights", &RBFNetwork::getNetworkWeights, this);
			get_running_weights_srv_ = nh.advertiseService("RBFNetwork/get_running_weights", &RBFNetwork::getRunningWeights, this);


			std::normal_distribution<double> d2(0,0.01);
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
			Eigen::VectorXd mean(req.globalPos.size());
			Eigen::MatrixXd covar = Eigen::MatrixXd::Zero(req.globalPos.size(),req.globalPos.size());
			for(int i =0;i<req.globalPos.size();i++){
				covar(i,i)=req.variance;
			}

			if (req.globalPos.size()<3){
				for (int column=0;column<numKernels;column++){
					mean(0)= req.globalPos[0]+req.radius*cos(column*column_spacing);
					mean(1)= req.globalPos[1]+req.radius*sin(column*column_spacing);
					GaussianKernel kernel(mean, covar);
					Network.push_back(kernel);
			// weights.push_back(0);
				}

			}
			else{
				for (int row = 1;row<=req.numRows;row++){
					mean(2) = req.globalPos[2]+row_spacing*row;
					for (int column=0;column<numKernels;column++){
						mean(0)= req.globalPos[0]+req.radius*cos(column*column_spacing);
						mean(1)= req.globalPos[1]+req.radius*sin(column*column_spacing);
						GaussianKernel kernel(mean, covar);
						Network.push_back(kernel);
				// weights.push_back(0);
					}
				}
			}

			numPolicies = req.numPolicies;
			weights = Eigen::MatrixXd::Zero(req.numKernels, req.numPolicies);
			runningWeights = weights;
			ROS_INFO("%d kernels were created", numKernels);
			PoWER.setParams(numKernels, 8, 5, numPolicies);
			return true;
		}

// bool RBFNetwork::networkOutput(grasp_learning::CallRBFN::Request& req, grasp_learning::CallRBFN::Response& res){
// 	Eigen::VectorXd x(req.pos.size());
// 	for(int i =0;i<req.pos.size();i++){
// 		x(i) = req.pos[i];
// 	}
// 	// activeKernels.clear();
// 	// double output = 0;
// 	std::vector<double> num(numPolicies);
// 	double dnom = 0;	
// 	for (int i = 0; i < Network.size(); i++){
// 		double activation = Network[i].kernelActivation(x);
// 		dnom += activation;
// 		for (int j = 0; j < num_policies; j++) {
// 		// if (activation != 0){
// 		// 	if (std::find(activeKernels.begin(), activeKernels.end(), i) == activeKernels.end())
// 		// 		activeKernels.push_back(i);
// 		// }
// 			num[j] += runningWeights[j][i]*activation;
// 		}
// 	}
//     // std::cout<<"Network output: "<<output<<std::endl;
//     // for (int i = 0; i < activeKernels.size(); i++) {
//     // 	std::cout<<activeKernels[i]<<std::endl;
//     // }
// 	for (int j = 0; j < num_policies; j++) {
// 		res.result[j] = num[j]	/dnom;
// 	}
// 	return true;
// }

		bool RBFNetwork::networkOutput(grasp_learning::CallRBFN::Request& req, grasp_learning::CallRBFN::Response& res){
			Eigen::VectorXd x(req.pos.size());
			for(int i =0;i<req.pos.size();i++){
				x(i) = req.pos[i];
			}
	// activeKernels.clear();
	// double output = 0;
			std::vector<double> result(numPolicies);
			double dnom = 0;	
			for (int i = 0; i < numKernels; i++){
				double activation = Network[i].kernelActivation(x);
				dnom += activation;
				for (int j = 0; j < numPolicies; j++) {
		// if (activation != 0){
		// 	if (std::find(activeKernels.begin(), activeKernels.end(), i) == activeKernels.end())
		// 		activeKernels.push_back(i);
		// }
					result[j] += runningWeights(i,j)*activation;
				}
			}
    // std::cout<<"Network output: "<<output<<std::endl;
    // for (int i = 0; i < activeKernels.size(); i++) {
    // 	std::cout<<activeKernels[i]<<std::endl;
    // }
			for (int j = 0; j < numPolicies; j++) {
				result[j] /= dnom;
			}
			res.result = result;
			return true;
		}

// std::vector<double> RBFNetwork::getKernelWeights(){
// 	return weights;
// }

		Eigen::MatrixXd RBFNetwork::getKernelWeights(){
			return weights;
		}


		bool RBFNetwork::printKernelWeights(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response){
	// for (auto iter:runningWeights){
	// 	std::cout << iter << std::endl;
	// }
			std::cout<<runningWeights<<std::endl;
			return true;
		}

		std::vector<double> RBFNetwork::getActiveKernels(){
			return activeKernels;
		}

		bool RBFNetwork::addWeightNoise(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response){
			resetRunningWeights();
			rollout_noise = Eigen::MatrixXd::Zero(numKernels, numPolicies);
			rollout_noise = sampleNoise();
			runningWeights += rollout_noise;
			return true;
		}

// std::vector<double> RBFNetwork::sampleNoise(){
// 	std::vector<double> noise;
// 	double sample = 0;
// 	for (int i = 0;i<numKernels;i++){
// 		sample = dist(generator);
// 		noise.push_back(sample);
// 	}
// 	return noise;
// }

		Eigen::MatrixXd RBFNetwork::sampleNoise(){
			Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(numKernels, numPolicies);
			for (int j = 0; j < numPolicies; j++) {
				for (int i = 0; i < Network.size(); i++){
					noise(i,j)= dist(generator);
				}
			}
			return noise;
		}

		void RBFNetwork::resetRunningWeights(){
			runningWeights = weights;
		}

// void RBFNetwork::updateWeights(std::vector<double> newWeights){
// 	for(int i=0;i<newWeights.size();i++){
// 		weights[i]+=newWeights[i];
// 	}
// }

		void RBFNetwork::updateWeights(Eigen::MatrixXd newWeights){
			weights += newWeights;
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
	// std::vector<double> updatedWeights = PoWER.policySearch(rollout_noise, req.reward, getActiveKernels());
			Eigen::MatrixXd updatedWeights = PoWER.policySearch(rollout_noise, req.reward, getActiveKernels());
			updateWeights(updatedWeights);
	// printVector(updatedWeights);
			return true;
		}

		bool RBFNetwork::getNetworkWeights(grasp_learning::GetNetworkWeights::Request& req, grasp_learning::GetNetworkWeights::Response& res){
			std::vector<double> vec;
			for(int i = 0;i<numPolicies;i++){
				for(int j=0;j<numKernels;j++){
					vec.push_back(weights(j,i));
				}
			}
			res.weights = vec;
			return true;
		}

		bool RBFNetwork::getRunningWeights(grasp_learning::GetNetworkWeights::Request& req, grasp_learning::GetNetworkWeights::Response& res){
			std::vector<double> vec;
			for(int i = 0;i<numPolicies;i++){
				for(int j=0;j<numKernels;j++){
					vec.push_back(runningWeights(j,i));
				}
			}
			res.weights = vec;
			return true;
		}


		void RBFNetwork::printVector(std::vector<double> vec){
			for(auto& iter: vec){
				std::cout<<iter<<std::endl;
			}
		}

	}
}

int main(int argc, char** argv) {
	ros::init(argc, argv, "RBFNetwork");

	demo_learning::RBFNetwork::RBFNetwork RBFNetwork;

	ROS_INFO("RBF network node ready");
	ros::AsyncSpinner spinner(4);  // Use 4 threads
	spinner.start();
	ros::waitForShutdown();

	return 0;
}
