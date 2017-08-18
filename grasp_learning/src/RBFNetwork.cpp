#include <grasp_learning/RBFNetwork.h>
namespace demo_learning {
	namespace RBFNetwork {

		GaussianKernel::GaussianKernel(Eigen::VectorXd mean, double var){
			this->mean = mean;
			this->var = var;
		}

		double GaussianKernel::residual(Eigen::VectorXd x){
			return (x-mean).squaredNorm();
		}

		double GaussianKernel::kernelActivation(Eigen::VectorXd x){
			double residual = this->residual(x);
			double beta = 1.0/(2.0*var*var);
			double activation = exp(-beta*residual);
			// std::cout<<residual<<"         "<<activation<<std::endl;

			return activation;
		}


		Eigen::VectorXd GaussianKernel::getMean(){
			return mean;
		}

		Eigen::MatrixXd GaussianKernel::getCovar(){
			return covar;
		}

		double GaussianKernel::getVar(){
			return var;
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
			vis_kernel_mean_srv_ = nh.advertiseService("RBFNetwork/visualize_kernel_means", &RBFNetwork::visualizeKernelMeans, this);

			marker_pub = nh.advertise<visualization_msgs::MarkerArray>("RBFNetwork/kernel_markers", 1);


			nh_ = ros::NodeHandle("~");

			nh_.param<int>("num_kernels", numKernels, 10);
			nh_.param<int>("num_kernel_rows", numRows, 1);

			nh_.param<double>("variance", intialNoiceVar, 0.001);
			nh_.param<int>("num_policies", numPolicies, 1);
			nh_.param<int>("num_dim", numDim, 1);
			nh_.param<bool>("use_corr_noise", useCorrNoise, false);

			nh_.param<int>("burn_in_trials",  burnInTrials, 8);
			nh_.param<int>("max_num_samples", maxNumSamples, 5);
			nh_.param<std::string>("relative_path", relativePath, "asd");

			kernelTotalActivationPerTimeFile = relativePath + "RBFN/kernel_total_activation_per_timestep.txt";
			kernelWiseTotalActivationFile = relativePath + "RBFN/kernel_wise_total_activation.txt";
			kernelOutputFile = relativePath + "RBFN/kernel_activation.txt";
			rewardsOutputFile = relativePath + "reward/normalized_rewards.txt";
			networkOutputFile = relativePath + "RBFN/output.txt";
			runningWeightsFile = relativePath + "RBFN/running_weights.txt";
			networkWeightsFile = relativePath + "RBFN/network_weights.txt";
			noiseFile = relativePath + "RBFN/noise.txt";
			krenelMeanFile = relativePath + "RBFN/kernel_mean.txt";
			krenelCovarFile = relativePath + "RBFN/kernel_covar.txt";



			fileHandler_.createFile(kernelTotalActivationPerTimeFile);
			fileHandler_.createFile(kernelWiseTotalActivationFile);
			fileHandler_.createFile(kernelOutputFile);
			fileHandler_.createFile(rewardsOutputFile);
			fileHandler_.createFile(networkOutputFile);
			fileHandler_.createFile(runningWeightsFile);
			fileHandler_.createFile(networkWeightsFile);
			fileHandler_.createFile(noiseFile);
			fileHandler_.createFile(krenelMeanFile);
			fileHandler_.createFile(krenelCovarFile);



			PoWER.setParams(numKernels, burnInTrials, maxNumSamples, numPolicies, relativePath);
			setNoiseVariance(intialNoiceVar);

			ROS_INFO("Set up all services");
		}


		bool RBFNetwork::buildRBFNetwork(grasp_learning::SetRBFN::Request& req, grasp_learning::SetRBFN::Response& res){
			ROS_INFO("Building the RBF Network");

			global_pos = req.globalPos;
			manifold_height = req.height;

			double column_spacing = (2*PI)/numKernels;
			double row_spacing = req.height/(numRows+1.0);
			Eigen::VectorXd mean(numDim);
			Eigen::MatrixXd covar = Eigen::MatrixXd::Zero(numDim,numDim);
			double dx = req.radius*(cos(0*column_spacing)-cos(1*column_spacing));
			double dy = req.radius*(sin(0*column_spacing)-sin(1*column_spacing));
			double var = calculateVariance(dx, dy);

			// for(int i =0;i<numDim;i++){
			// 	covar(i,i)=var;
			// }

			if (numDim<3){
				for (int column=0;column<numKernels;column++){
					mean(0)= req.globalPos[0]+req.radius*cos(column*column_spacing);
					mean(1)= req.globalPos[1]+req.radius*sin(column*column_spacing);
					// GaussianKernel kernel(mean, covar);
					GaussianKernel kernel(mean, var);
					Network.push_back(kernel);
				}

			}
			else{
				for (int row = 1;row<=numRows;row++){
					mean(2) = req.globalPos[2]+req.height/2;
					for (int column=0;column<numKernels;column++){
						mean(0)= req.globalPos[0]+req.radius*cos(column*column_spacing);
						mean(1)= req.globalPos[1]+req.radius*sin(column*column_spacing);
						// GaussianKernel kernel(mean, covar);
						GaussianKernel kernel(mean, var);
						Network.push_back(kernel);
					}
				}
			}

			weights = Eigen::MatrixXd::Zero(numKernels, numPolicies);
			runningWeights = weights;
			ROS_INFO("%d kernels were created", numKernels);
			rollout_noise = Eigen::MatrixXd::Zero(numKernels,0);
			saveKernelsToFile();
			return true;
		}

		void RBFNetwork::saveKernelsToFile(){
			for (int i = 0; i < numKernels; i++){
				saveDataToFile(krenelMeanFile, Network[i].getMean().transpose(), true);
				saveDataToFile(krenelCovarFile, Network[i].getVar(), true);
			}
		}

		void RBFNetwork::setNoiseVariance(const double variance){
			if (useCorrNoise){
				multiVarGauss.setCovarAsDiffernceMatrix(numKernels, variance);
			}
			else{
				multiVarGauss.setCovarAsIndentityMatrix(numKernels, variance);
			}
		}

		double RBFNetwork::calculateVariance(double dx, double dy){
			double var = sqrt(dx*dx+dy*dy);
			// ROS_INFO("Variance is %lf", var);
			return var;
		}

		bool RBFNetwork::networkOutput(grasp_learning::CallRBFN::Request& req, grasp_learning::CallRBFN::Response& res){
			Eigen::VectorXd x(numDim);
			int num_col = kernelOutput.cols();
			kernelOutput.conservativeResize(numKernels, num_col+1);
			for(int i =0;i<numDim;i++){
				x(i) = req.pos[i];
			}

			std::vector<double> result(numPolicies,0.0);
			double dnom = 0;	
			for (int i = 0; i < numKernels; i++){
				double activation = Network[i].kernelActivation(x);
				if(activation < ACTIVATION_THRESHOLD){
					activation = 0;
				}
				kernelOutput(i,num_col) = activation;

				dnom += activation;
				for (int j = 0; j < numPolicies; j++) {
					result[j] += runningWeights(i,j)*activation;
				}
			}
			if(dnom != 0){
				kernelOutput.col(num_col) /= dnom;
				for (int j = 0; j < numPolicies; j++) {
					result[j] /= dnom;
				}
			}
			// else{
			// 	ROS_INFO("Zero total activation");
			// }

			networkOutput_.push_back(result[0]);
			res.result = result;

			return true;
		}


		Eigen::MatrixXd RBFNetwork::getKernelWeights(){
			return weights;
		}


		bool RBFNetwork::printKernelWeights(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response){
			std::cout<<runningWeights<<std::endl;
			return true;
		}

		bool RBFNetwork::addWeightNoise(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response){
			resetRollout();
			int num_cols = rollout_noise.cols();
			rollout_noise.conservativeResize(numKernels, num_cols+1);

			if (!coverged){
				rollout_noise.col(num_cols) = sampleNoise();
				runningWeights += rollout_noise.col(num_cols);
			}
			else{
				ROS_INFO("Policy has converged after %d rollouts", (int)PoWER.getNumRollouts());
			}
			return true;
		}

		bool RBFNetwork::policyConverged(){
			std::vector<double> vec = PoWER.getHighestRewards();
			if(meanOfVector(vec)>COVERGANCE_THRESHOLD){
				return true;
			}
			else{
				return false;
			}
		}

		double RBFNetwork::meanOfVector(const std::vector<double>& vec){
			double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
			return sum /vec.size();
		}

		Eigen::MatrixXd RBFNetwork::sampleNoise(){
			return multiVarGauss.sample(1);
		}

		void RBFNetwork::updateNoiseVariance(){
			double beta = PoWER.varianceSearch();
			ROS_INFO("New beta %lf and noise variance %lf", beta, beta*intialNoiceVar);
			setNoiseVariance(beta*intialNoiceVar);
		}

		void RBFNetwork::resetRunningWeights(){
			runningWeights = weights;
		}


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

		void RBFNetwork::resetRollout(){
			updateNoiseVariance();
			resetRunningWeights();
			coverged = (policyConverged() ? true:false);
			kernelOutput = Eigen::MatrixXd::Zero(numKernels, 0);
			// rollout_noise = Eigen::MatrixXd::Zero(numKernels, numPolicies);
			rollout_noise = Eigen::MatrixXd::Zero(numKernels, 0);
			networkOutput_.clear();
		}

		bool RBFNetwork::policySearch(grasp_learning::PolicySearch::Request& req, grasp_learning::PolicySearch::Response& res){
			// Eigen::MatrixXd updatedWeights = PoWER.policySearch(rollout_noise, req.reward, kernelOutput*kernelOutput.transpose());
			if(!coverged){
				kernelOutput = (kernelOutput.array() < ACTIVATION_THRESHOLD).select(0, kernelOutput);

				Eigen::MatrixXd updatedWeights = PoWER.policySearch(rollout_noise, req.rewards, kernelOutput);
				updateWeights(updatedWeights);
				std::cout<<updatedWeights.transpose()<<std::endl;
			}
			double* ptr = &req.rewards[0];
			Eigen::Map<Eigen::VectorXd> rewards(ptr, req.rewards.size());

			double* ptr2 = &networkOutput_[0];
			Eigen::Map<Eigen::VectorXd> outputs(ptr2, networkOutput_.size());


			saveDataToFile(kernelTotalActivationPerTimeFile, kernelOutput.colwise().sum(), true);
			saveDataToFile(kernelWiseTotalActivationFile, kernelOutput.rowwise().sum().transpose(), true);
			saveDataToFile(kernelOutputFile, kernelOutput, false);
			saveDataToFile(rewardsOutputFile, rewards.transpose(), true);
			saveDataToFile(networkOutputFile, outputs.transpose(), true);
			saveDataToFile(networkWeightsFile, weights.transpose(), true);
			saveDataToFile(runningWeightsFile, runningWeights.transpose(), true);
			saveDataToFile(noiseFile, rollout_noise.transpose(), true);

			return true;
		}

		template<typename T>
		void RBFNetwork::saveDataToFile(std::string file, T data, bool append){
			bool success;
			if (append){
				success = fileHandler_.appendToFile(file, data);
			}
			else{
				success = fileHandler_.writeToFile(file, data);
			}
			if (!success){
				ROS_INFO("Could not store data in file %s",file.c_str());
			}
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
				std::cout<<iter<<" ";
			}
			std::cout<<std::endl;
		}


		bool RBFNetwork::visualizeKernelMeans(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response){
			visualization_msgs::MarkerArray marker_array;

			// Eigen::MatrixXd covar = Network[0].getCovar();
			double var = Network[0].getVar();
			for(int i =0;i<numKernels;i++){
				Eigen::VectorXd mean = Network[i].getMean();

				visualization_msgs::Marker marker_mean;
				marker_mean.header.frame_id = "world";
				marker_mean.header.stamp = ros::Time();
				marker_mean.ns = "gaussian_kernels";
				marker_mean.id = i+1;
				marker_mean.type = visualization_msgs::Marker::SPHERE;
				marker_mean.action = visualization_msgs::Marker::ADD;
				marker_mean.pose.position.x = mean(0);
				marker_mean.pose.position.y = mean(1);
				marker_mean.pose.position.z = global_pos[2]+manifold_height/2;
				marker_mean.pose.orientation.x = 0.0;
				marker_mean.pose.orientation.y = 0.0;
				marker_mean.pose.orientation.z = 0.0;
				marker_mean.pose.orientation.w = 1.0;
				marker_mean.scale.x = 0.01;
				marker_mean.scale.y = 0.01;
				marker_mean.scale.z = 0.01;
				marker_mean.color.a = 0.5;
				marker_mean.color.r = 0.0;
				marker_mean.color.g = 1.0;
				marker_mean.color.b = 0.0;
				marker_mean.lifetime = ros::Duration();

				marker_array.markers.push_back(marker_mean);

				visualization_msgs::Marker marker_var;
				marker_var.header.frame_id = "world";
				marker_var.header.stamp = ros::Time();
				marker_var.ns = "gaussian_kernels";
				marker_var.id = -(i+1);
				marker_var.type = visualization_msgs::Marker::CYLINDER;
				marker_var.action = visualization_msgs::Marker::ADD;
				marker_var.pose.position.x = mean(0);
				marker_var.pose.position.y = mean(1);
				marker_var.pose.position.z = global_pos[2]+manifold_height/2;
				marker_var.pose.orientation.x = 0.0;
				marker_var.pose.orientation.y = 0.0;
				marker_var.pose.orientation.z = 0.0;
				marker_var.pose.orientation.w = 1.0;
				marker_var.scale.x = var/2.0;
				marker_var.scale.y = var/2.0;
				marker_var.scale.z = 0;
				marker_var.color.a = 0.5;
				marker_var.color.r = 1.0;
				marker_var.color.g = 0.0;
				marker_var.color.b = 0.0;
				marker_var.lifetime = ros::Duration();

				marker_array.markers.push_back(marker_var);


			}
			marker_pub.publish(marker_array);
			return true;
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
