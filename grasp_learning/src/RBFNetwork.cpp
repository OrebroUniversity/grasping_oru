#include <grasp_learning/RBFNetwork.h>
namespace demo_learning {
	namespace RBFNetwork {

		GaussianKernel::GaussianKernel(Eigen::VectorXd mean, Eigen::MatrixXd covar){
			this->mean = mean;
			this->covar = covar;
		}

		Eigen::VectorXd GaussianKernel::residual(Eigen::VectorXd x){
			// ROS_INFO_DELAYED_THROTTLE(0.5, "kernel mean [%lf, %lf]",mean(0), mean(1));

			return x-mean;
		}


		double GaussianKernel::kernelActivation(Eigen::VectorXd x){
			Eigen::VectorXd residual = this->residual(x);
			double activation = exp(-0.5*residual.transpose()*covar.inverse()*residual);
			return activation;
		}

		Eigen::VectorXd GaussianKernel::getMean(){
			return mean;
		}

		double GaussianKernel::getVar(){
			return covar(0,0);
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

			kernelOutputFileName = relativePath + "kernels.txt";

			rewardsOutputFileName = relativePath + "rewards.txt";


			PoWER.setParams(numKernels, burnInTrials, maxNumSamples, numPolicies);
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
			double var = calculateVariance(dx, dy)/100;
			for(int i =0;i<numDim;i++){
				covar(i,i)=var;
			}

			if (numDim<3){
				for (int column=0;column<numKernels;column++){
					mean(0)= req.globalPos[0]+req.radius*cos(column*column_spacing);
					mean(1)= req.globalPos[1]+req.radius*sin(column*column_spacing);
					GaussianKernel kernel(mean, covar);
					Network.push_back(kernel);
				}

			}
			else{
				for (int row = 1;row<=numRows;row++){
					mean(2) = req.globalPos[2]+row_spacing*row;
					for (int column=0;column<numKernels;column++){
						mean(0)= req.globalPos[0]+req.radius*cos(column*column_spacing);
						mean(1)= req.globalPos[1]+req.radius*sin(column*column_spacing);
						GaussianKernel kernel(mean, covar);
						Network.push_back(kernel);
					}
				}
			}

			weights = Eigen::MatrixXd::Zero(numKernels, numPolicies);
			runningWeights = weights;
			ROS_INFO("%d kernels were created", numKernels);
			rollout_noise = Eigen::MatrixXd::Zero(numKernels,0);
			return true;
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
			Eigen::VectorXd x(req.pos.size()-1);
			int num_col = kernelOutput.cols();
			kernelOutput.conservativeResize(numKernels, num_col+1);
			for(int i =0;i<2;i++){
				x(i) = req.pos[i];
			}

			std::vector<double> result(numPolicies);
			double dnom = 0;	
			for (int i = 0; i < numKernels; i++){
				double activation = Network[i].kernelActivation(x);
				kernelOutput(i,num_col) = activation;
				dnom += activation;
				for (int j = 0; j < numPolicies; j++) {
					result[j] += runningWeights(i,j)*activation;
				}
			}

			kernelOutput.col(num_col) /= dnom;

			for (int j = 0; j < numPolicies; j++) {
				result[j] /= dnom;
			}
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

		std::vector<double> RBFNetwork::getActiveKernels(){
			return activeKernels;
		}

		bool RBFNetwork::addWeightNoise(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response){
			resetRunningWeights();
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
			// Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(numKernels, numPolicies);
			// for (int j = 0; j < numPolicies; j++) {
			// 	noise.col(j) = multiVarGauss.sample(1);
			// }
			return multiVarGauss.sample(1);//noise;
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
			kernelOutput = Eigen::MatrixXd::Zero(numKernels, 0);
			// rollout_noise = Eigen::MatrixXd::Zero(numKernels, numPolicies);
			rollout_noise = Eigen::MatrixXd::Zero(numKernels, 0);

		}

		bool RBFNetwork::policySearch(grasp_learning::PolicySearch::Request& req, grasp_learning::PolicySearch::Response& res){
			// Eigen::MatrixXd updatedWeights = PoWER.policySearch(rollout_noise, req.reward, kernelOutput*kernelOutput.transpose());

			kernelOutput = (kernelOutput.array() < ACTIVATION_THRESHOLD).select(0, kernelOutput);

			Eigen::MatrixXd updatedWeights = PoWER.policySearch(rollout_noise, req.rewards, kernelOutput);
			updateWeights(updatedWeights);
			std::ofstream file(kernelOutputFileName);
			if (file.is_open())
			{
				file << kernelOutput << '\n';
			}

			double* ptr = &req.rewards[0];
			Eigen::Map<Eigen::VectorXd> rewards(ptr, req.rewards.size());

			std::ofstream file2(rewardsOutputFileName);
			if (file2.is_open())
			{
				file2 << rewards << '\n';
			}

			std::cout<<updatedWeights.transpose()<<std::endl;
			coverged = (policyConverged() ? true:false);

			updateNoiseVariance();
			resetRollout();
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


		bool RBFNetwork::visualizeKernelMeans(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response){
			visualization_msgs::MarkerArray marker_array;

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
				marker_var.scale.x = var*2;
				marker_var.scale.y = var*2;
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
