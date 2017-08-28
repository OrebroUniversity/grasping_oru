#include <grasp_learning/RBFNetwork.h>
namespace demo_learning {
namespace RBFNetwork {

GaussianKernel::GaussianKernel(Eigen::VectorXd mean, double var) {
	this->mean = mean;
	this->var = var;
}

double GaussianKernel::residual(Eigen::VectorXd x) {
	return (x - mean).squaredNorm();
}

double GaussianKernel::kernelActivation(Eigen::VectorXd x) {
	double residual = this->residual(x);
	double beta = 1.0 / (2.0 * var * var);
	double activation = exp(-beta * residual);
	// std::cout<<residual<<"         "<<activation<<std::endl;

	return activation;
}


Eigen::VectorXd GaussianKernel::getMean() {
	return mean;
}

Eigen::MatrixXd GaussianKernel::getCovar() {
	return covar;
}

double GaussianKernel::getVar() {
	return var;
}


RBFNetwork::RBFNetwork() {

	print_kernel_means_srv_ = nh.advertiseService("RBFNetwork/print_kernel_means", &RBFNetwork::printAllKernelMeans, this);
	add_weight_noise_srv_ = nh.advertiseService("RBFNetwork/add_weight_noise", &RBFNetwork::addWeightNoise, this);
	print_kernel_weights_srv_ = nh.advertiseService("RBFNetwork/print_kernel_weights", &RBFNetwork::printKernelWeights, this);
	build_RBF_network_srv_ = nh.advertiseService("RBFNetwork/build_RBFNetwork", &RBFNetwork::buildRBFNetwork, this);
	network_output_srv_ = nh.advertiseService("RBFNetwork/network_output", &RBFNetwork::networkOutput, this);
	policy_search_srv_ = nh.advertiseService("RBFNetwork/policy_search", &RBFNetwork::policySearch, this);
	get_network_weights_srv_ = nh.advertiseService("RBFNetwork/get_network_weights", &RBFNetwork::getNetworkWeights, this);
	get_running_weights_srv_ = nh.advertiseService("RBFNetwork/get_running_weights", &RBFNetwork::getRunningWeights, this);
	vis_kernel_mean_srv_ = nh.advertiseService("RBFNetwork/visualize_kernel_means", &RBFNetwork::visualizeKernelMeans, this);
	reset_RBFN_srv_ = nh.advertiseService("RBFNetwork/reset_RBFN", &RBFNetwork::resetRBFNetwork, this);

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

	nh_.param<double>("grid_x", grid_x_, 1.0);
	nh_.param<double>("grid_y", grid_y_, 1.0);
	nh_.param<double>("grid_z", grid_z_, 1.0);

	nh_.param<std::string>("spacing_policy", spacingPolicy_, "grid");

	nh_.param<std::string>("relative_path", relativePath, " ");


	createFiles(relativePath);

	PoWER.setParams(numKernels, burnInTrials, maxNumSamples, numPolicies);
	setNoiseVariance(intialNoiceVar);

	ROS_INFO("Set up all services");
}

bool RBFNetwork::createFiles(std::string relPath) {


	std::string trial_folder = relPath + "trial_" + std::to_string(numTrial_) + "/";

	std::string RBFN_folder = relPath + "trial_" + std::to_string(numTrial_) + "/RBFN/";

	std::vector<std::string> folders {trial_folder, RBFN_folder};

	if (!fileHandler_.createFolders(folders)) {
		ROS_INFO("Foalder not created");
		return false;
	}

	kernelTotalActivationPerTimeFile = RBFN_folder + "kernel_total_activation_per_timestep.txt";
	kernelWiseTotalActivationFile = RBFN_folder + "kernel_wise_total_activation.txt";
	kernelOutputFile = RBFN_folder + "kernel_activation.txt";

	networkOutputFile = RBFN_folder + "output.txt";
	runningWeightsFile = RBFN_folder + "running_weights.txt";
	networkWeightsFile = RBFN_folder + "network_weights.txt";
	noiseFile = RBFN_folder + "noise.txt";
	krenelMeanFile = RBFN_folder + "kernel_mean.txt";
	krenelCovarFile = RBFN_folder + "kernel_covar.txt";
	numKernelFile = RBFN_folder + "number_of_kernels.txt";
	noisePerTimeStepFile = RBFN_folder + "noise_per_timestep.txt";
	fileHandler_.createFile(kernelTotalActivationPerTimeFile);
	fileHandler_.createFile(kernelWiseTotalActivationFile);
	fileHandler_.createFile(kernelOutputFile);
	fileHandler_.createFile(networkOutputFile);
	fileHandler_.createFile(runningWeightsFile);
	fileHandler_.createFile(networkWeightsFile);
	fileHandler_.createFile(noiseFile);
	fileHandler_.createFile(krenelMeanFile);
	fileHandler_.createFile(krenelCovarFile);
	fileHandler_.createFile(numKernelFile);
	fileHandler_.createFile(noisePerTimeStepFile);


}


bool RBFNetwork::buildRBFNetwork(grasp_learning::SetRBFN::Request& req, grasp_learning::SetRBFN::Response& res) {
	ROS_INFO("Building the RBF Network");

	global_pos = req.globalPos;

	if (spacingPolicy_.compare("grid") == 0) {
		spaceKernelsOnGrid();
	} else if (spacingPolicy_.compare("manifold") == 0) {
		spaceKernelsOnManifold(req.height, req.radius);
	} else if (spacingPolicy_.compare("plane") == 0) {
		spaceKernelsOnPlane(req.height);
	}

	weights = Eigen::MatrixXd::Zero(numKernels, numPolicies);
	runningWeights = weights;
	ROS_INFO("%d kernels were created", numKernels);
	rollout_noise = Eigen::MatrixXd::Zero(numKernels, numPolicies);
	saveKernelsToFile();
	return true;
}

void RBFNetwork::spaceKernelsOnPlane(double height) {

	Eigen::VectorXd mean(numDim);

	double numKernPerDim = sqrt(numKernels);

	double dx = grid_x_ / numKernPerDim;
	double dy = grid_y_ / numKernPerDim;
	double var = calculateVariance(dx, dy) / 5;
	std::cout << var << std::endl;

	for (double x = 0; x < dx * numKernPerDim; x += dx) {
		for (double y = 0; y < dy * numKernPerDim; y += dy) {
			mean(0) = x + global_pos[0] / 2.0;
			mean(1) = y - grid_y_ / 2.0;
			if (numDim == 3) {
				mean(2) = global_pos[2] + height;
			}
			GaussianKernel kernel(mean, var);
			Network.push_back(kernel);
		}
	}
}

void RBFNetwork::spaceKernelsOnGrid() {

	Eigen::VectorXd mean(numDim);

	double numKernPerDim = cbrt(numKernels);


	double dx = grid_x_ / numKernPerDim;
	double dy = grid_y_ / numKernPerDim;
	double dz = grid_z_ / numKernPerDim;
	double var = calculateVariance(dx, dy, dz) / 5.0;
	std::cout << var << std::endl;

	for (double x = 0; x < dx * numKernPerDim; x += dx) {
		for (double y = 0; y < dy * numKernPerDim; y += dy) {
			for (double z = 0; z < dz * numKernPerDim; z += dz) {
				mean(0) = x + global_pos[0] / 2.0 - grid_x_ / 2.0 + 0.2;
				mean(1) = y + global_pos[1] / 2.0 - grid_y_ / 2.0;
				mean(2) = z + global_pos[2] / 2.0 - grid_z_ / 2.0 + 0.3;
				GaussianKernel kernel(mean, var);
				Network.push_back(kernel);
			}
		}
	}

}

void RBFNetwork::spaceKernelsOnManifold(double height, double radius) {

	manifold_height = height;
	double column_spacing = (2 * PI) / numKernels;
	double row_spacing = height / (numRows + 1.0);
	Eigen::VectorXd mean(numDim);
	double r = radius;
	double dx = r * (cos(0 * column_spacing) - cos(1 * column_spacing));
	double dy = r * (sin(0 * column_spacing) - sin(1 * column_spacing));
	double var = 2 * calculateVariance(dx, dy);
	std::cout << var << std::endl;

	if (numDim < 3) {
		for (int column = 0; column < numKernels; column++) {
			mean(0) = global_pos[0] + r * cos(column * column_spacing);
			mean(1) = global_pos[1] + r * sin(column * column_spacing);
			GaussianKernel kernel(mean, var);
			Network.push_back(kernel);
		}

	} else {
		for (int row = 1; row <= numRows; row++) {
			mean(2) = global_pos[2] + height / 2;
			for (int column = 0; column < numKernels; column++) {
				mean(0) = global_pos[0] + r * cos(column * column_spacing);
				mean(1) = global_pos[1] + r * sin(column * column_spacing);
				GaussianKernel kernel(mean, var);
				Network.push_back(kernel);
			}
		}
	}
}

void RBFNetwork::saveKernelsToFile() {
	for (int i = 0; i < numKernels; i++) {
		saveDataToFile(krenelMeanFile, Network[i].getMean().transpose(), true);
		saveDataToFile(krenelCovarFile, Network[i].getVar(), true);
	}
	saveDataToFile(numKernelFile, numKernels, false);

}

void RBFNetwork::setNoiseVariance(const double variance) {
	if (useCorrNoise) {
		multiVarGauss.setCovarAsDiffernceMatrix(numKernels, variance);
	} else {
		multiVarGauss.setCovarAsIndentityMatrix(numKernels, variance);
	}
}

double RBFNetwork::calculateVariance(double dx, double dy, double dz) {
	double var = sqrt(dx * dx + dy * dy + dz * dz);
	// ROS_INFO("Variance is %lf", var);
	return var;
}

bool RBFNetwork::networkOutput(grasp_learning::CallRBFN::Request& req, grasp_learning::CallRBFN::Response& res) {
	Eigen::VectorXd x(numDim);
	int num_col = kernelOutput.cols();
	kernelOutput.conservativeResize(numKernels, num_col + 1);
	for (int i = 0; i < numDim; i++) {
		x(i) = req.pos[i];
	}
	Eigen::MatrixXd kernelNoise = Eigen::MatrixXd::Zero(numKernels, numPolicies);

	std::vector<double> result(numPolicies, 0.0);
	double dnom = 0;
	double maxActivation = 0;
	int maxIdx = 0;
	for (int i = 0; i < numKernels; i++) {
		double activation = Network[i].kernelActivation(x);
		// if(activation < ACTIVATION_THRESHOLD){
		// 	activation = 0;
		// }
		if (activation > maxActivation) {
			maxActivation = activation;
			maxIdx = i;
		}
		kernelOutput(i, num_col) = activation;

		dnom += activation;
		// for (int j = 0; j < numPolicies; j++) {
		// result[j] += runningWeights(i,j)*activation;
		// }
	}
	kernelNoise.row(maxIdx) = rollout_noise.row(maxIdx);
	Eigen::MatrixXd noisyWeights = weights + kernelNoise;
	for (int j = 0; j < numPolicies; j++) {
		result[j] = noisyWeights.transpose().row(j) * kernelOutput.col(num_col);
	}
	noisePerTimestep_.push_back(kernelNoise);
	if (dnom != 0) {
		// kernelOutput.col(num_col) /= dnom;
		for (int j = 0; j < numPolicies; j++) {
			result[j] /= dnom;
		}
	}
	// std::cout<<kernelOutput<<std::endl;
	networkOutput_.push_back(result);
	res.result = result;

	return true;
}


Eigen::MatrixXd RBFNetwork::getKernelWeights() {
	return weights;
}


bool RBFNetwork::printKernelWeights(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response) {
	std::cout << runningWeights << std::endl;
	return true;
}

bool RBFNetwork::addWeightNoise(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response) {
	resetRollout();

	if (!coverged) {
		for (int i = 0; i < numPolicies; i++) {
			rollout_noise.col(i) = sampleNoise();
			runningWeights.col(i) += rollout_noise.col(i);
		}
	} else {
		ROS_INFO("Policy has converged after %d rollouts", (int)PoWER.getNumRollouts());
	}
	return true;
}

bool RBFNetwork::policyConverged() {
	std::vector<double> vec = PoWER.getHighestRewards();
	if (meanOfVector(vec) > COVERGANCE_THRESHOLD) {
		return true;
	} else {
		return false;
	}
}

double RBFNetwork::meanOfVector(const std::vector<double>& vec) {
	double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
	return sum / vec.size();
}

Eigen::MatrixXd RBFNetwork::sampleNoise() {
	return multiVarGauss.sample(1);
}

void RBFNetwork::updateNoiseVariance() {
	double beta = PoWER.varianceSearch();
	ROS_INFO("New beta %lf and noise variance %lf", beta, beta * intialNoiceVar);
	setNoiseVariance(beta * intialNoiceVar);
}

void RBFNetwork::resetRunningWeights() {
	runningWeights = weights;
}


void RBFNetwork::updateWeights(Eigen::MatrixXd newWeights) {
	weights += newWeights;
}


bool RBFNetwork::printAllKernelMeans(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response) {
	for (int i = 0; i < Network.size(); i++) {
		std::cout << "Kernel: " << i << " mean " << Network[i].getMean() << std::endl;
	}
	return true;
}

void RBFNetwork::printKernelMean(int kernel) {
	std::cout << "Kernel: " << kernel << std::endl << Network[kernel].getMean() << std::endl;
}

void RBFNetwork::resetRollout() {
	updateNoiseVariance();
	resetRunningWeights();
	kernelOutput = Eigen::MatrixXd::Zero(numKernels, 0);
	rollout_noise.setZero();
	networkOutput_.clear();
	noisePerTimestep_.clear();
}

bool RBFNetwork::policySearch(grasp_learning::PolicySearch::Request& req, grasp_learning::PolicySearch::Response& res) {

	if (!coverged) {
		// kernelOutput = (kernelOutput.array() < ACTIVATION_THRESHOLD).select(0, kernelOutput);
		// std::cout<<kernelOutput<<std::endl;

		// Eigen::MatrixXd updatedWeights = PoWER.policySearch(rollout_noise, req.rewards, kernelOutput, noisePerTimestep_);
		Eigen::MatrixXd updatedWeights = PoWER.policySearch2(noisePerTimestep_, req.rewards);

		updateWeights(updatedWeights);
		std::cout << weights.transpose() << std::endl;
		// std::cout<<updatedWeights.size()<<std::endl;

	}

	coverged = (policyConverged() ? true : false);
	res.converged = coverged;

	double* ptr = &req.rewards[0];
	Eigen::Map<Eigen::VectorXd> rewards(ptr, req.rewards.size());

	saveDataToFile(networkOutputFile, ConvertToEigenMatrix(networkOutput_).transpose(), true);
	saveDataToFile(networkWeightsFile, weights.transpose(), true);
	saveDataToFile(runningWeightsFile, runningWeights.transpose(), true);
	saveDataToFile(noiseFile, rollout_noise.transpose(), true);
	saveDataToFile(kernelWiseTotalActivationFile, kernelOutput.rowwise().sum().transpose(), true);
	saveDataToFile(kernelTotalActivationPerTimeFile, kernelOutput.colwise().sum(), true);
	saveDataToFile(kernelOutputFile, kernelOutput, false);
	saveDataToFile(noisePerTimeStepFile, concatenateMatrices(noisePerTimestep_), true);
}

template<typename T>
void RBFNetwork::saveDataToFile(std::string file, T data, bool append) {
	bool success;
	if (append) {
		success = fileHandler_.appendToFile(file, data);
	} else {
		success = fileHandler_.writeToFile(file, data);
	}
	if (!success) {
		ROS_INFO("Could not store data in file %s", file.c_str());
	}
}

bool RBFNetwork::getNetworkWeights(grasp_learning::GetNetworkWeights::Request& req, grasp_learning::GetNetworkWeights::Response& res) {
	std::vector<double> vec;
	for (int i = 0; i < numPolicies; i++) {
		for (int j = 0; j < numKernels; j++) {
			vec.push_back(weights(j, i));
		}
	}
	res.weights = vec;
	return true;
}

bool RBFNetwork::getRunningWeights(grasp_learning::GetNetworkWeights::Request& req, grasp_learning::GetNetworkWeights::Response& res) {
	std::vector<double> vec;
	for (int i = 0; i < numPolicies; i++) {
		for (int j = 0; j < numKernels; j++) {
			vec.push_back(runningWeights(j, i));
		}
	}
	res.weights = vec;
	return true;
}


void RBFNetwork::printVector(std::vector<double> vec) {
	for (auto& iter : vec) {
		std::cout << iter << " ";
	}
	std::cout << std::endl;
}

bool RBFNetwork::resetRBFNetwork(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response) {
	ROS_INFO("Resetting the RBFNetwork (including the weights) and PoWER algorithm.");
	PoWER.resetPolicySearch();
	weights.setZero();
	resetRunningWeights();
	coverged = false;
	kernelOutput = Eigen::MatrixXd::Zero(numKernels, 0);
	rollout_noise.setZero();
	networkOutput_.clear();
	setNoiseVariance(intialNoiceVar);
	numTrial_++;
	createFiles(relativePath);
	saveKernelsToFile();
	return true;
}

Eigen::MatrixXd RBFNetwork::concatenateMatrices(std::vector<Eigen::MatrixXd> data) {
	Eigen::MatrixXd eMatrix(numKernels, 0);
	for (int i = 0; i < data.size(); i++) {
		eMatrix.conservativeResize(numKernels, eMatrix.cols() + data[i].cols());
		eMatrix.col(eMatrix.cols() - 2) = data[0].col(0);
		eMatrix.col(eMatrix.cols() - 1) = data[1].col(1);
	}
	return eMatrix;
}

Eigen::MatrixXd RBFNetwork::ConvertToEigenMatrix(std::vector<std::vector<double>> data) {
	Eigen::MatrixXd eMatrix(data.size(), data[0].size());
	for (int i = 0; i < data.size(); ++i)
		eMatrix.row(i) = Eigen::VectorXd::Map(&data[i][0], data[0].size());
	return eMatrix;
}


bool RBFNetwork::visualizeKernelMeans(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response) {
	visualization_msgs::MarkerArray marker_array;

	double var = Network[0].getVar();
	for (int i = 0; i < numKernels; i++) {
		Eigen::VectorXd mean = Network[i].getMean();

		visualization_msgs::Marker marker_mean;
		marker_mean.header.frame_id = "world";
		marker_mean.header.stamp = ros::Time();
		marker_mean.ns = "gaussian_kernels";
		marker_mean.id = i + 1;
		marker_mean.type = visualization_msgs::Marker::SPHERE;
		marker_mean.action = visualization_msgs::Marker::ADD;
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

		visualization_msgs::Marker marker_num;
		marker_num.header.frame_id = "world";
		marker_num.header.stamp = ros::Time();
		marker_num.ns = "gaussian_kernels";
		marker_num.id = numKernels + 1 + i;
		marker_num.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
		marker_num.action = visualization_msgs::Marker::ADD;
		marker_num.scale.z = 0.05;
		marker_num.color.a = 0.5;
		marker_num.color.r = 0.0;
		marker_num.color.g = 0.0;
		marker_num.color.b = 1.0;
		marker_num.lifetime = ros::Duration();
		marker_num.text = std::to_string(i);


		visualization_msgs::Marker marker_var;
		marker_var.header.frame_id = "world";
		marker_var.header.stamp = ros::Time();
		marker_var.ns = "gaussian_kernels";
		marker_var.id = -(i + 1);
		marker_var.action = visualization_msgs::Marker::ADD;
		marker_var.pose.orientation.x = 0.0;
		marker_var.pose.orientation.y = 0.0;
		marker_var.pose.orientation.z = 0.0;
		marker_var.pose.orientation.w = 1.0;
		marker_var.color.a = 0.0;
		marker_var.color.r = 1.0;
		marker_var.color.g = 0.0;
		marker_var.color.b = 0.0;
		marker_var.lifetime = ros::Duration();


		if (spacingPolicy_.compare("grid") == 0) {
			marker_var.type = visualization_msgs::Marker::SPHERE;

			marker_mean.pose.position.x = mean(0);
			marker_mean.pose.position.y = mean(1);
			marker_mean.pose.position.z = mean(2);

			marker_var.pose.position.x = mean(0);
			marker_var.pose.position.y = mean(1);
			marker_var.pose.position.z = mean(2);
			marker_var.scale.x = var / 2.0;
			marker_var.scale.y = var / 2.0;
			marker_var.scale.z = var / 2.0;

		} else if (spacingPolicy_.compare("manifold") == 0) {
			marker_var.type = visualization_msgs::Marker::CYLINDER;

			marker_mean.pose.position.x = mean(0);
			marker_mean.pose.position.y = mean(1);
			marker_mean.pose.position.z = global_pos[2] + manifold_height / 2;

			marker_num.pose.position.x = mean(0);
			marker_num.pose.position.y = mean(1);
			marker_num.pose.position.z = global_pos[2] + manifold_height / 2;

			marker_var.pose.position.x = mean(0);
			marker_var.pose.position.y = mean(1);
			marker_var.pose.position.z = global_pos[2] + manifold_height / 2;
			marker_var.scale.x = var / 2.0;
			marker_var.scale.y = var / 2.0;
			marker_var.scale.z = 0;
		} else if (spacingPolicy_.compare("plane") == 0) {
			marker_var.type = visualization_msgs::Marker::CYLINDER;

			marker_mean.pose.position.x = mean(0);
			marker_mean.pose.position.y = mean(1);
			marker_mean.pose.position.z = global_pos[2] + manifold_height;

			marker_num.pose.position.x = mean(0);
			marker_num.pose.position.y = mean(1);
			marker_num.pose.position.z = global_pos[2] + manifold_height;

			marker_var.pose.position.x = mean(0);
			marker_var.pose.position.y = mean(1);
			marker_var.pose.position.z = global_pos[2] + manifold_height;
			marker_var.scale.x = var / 2.0;
			marker_var.scale.y = var / 2.0;
			marker_var.scale.z = 0;
		}


		marker_array.markers.push_back(marker_mean);
		marker_array.markers.push_back(marker_var);
		if (spacingPolicy_.compare("grid") != 0) {
			marker_array.markers.push_back(marker_num);
		}
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
