#ifndef RBFN_H
#define RBFN_H

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "ros/ros.h"
#include <math.h>
#include <grasp_learning/power.h>
#include <grasp_learning/MultiVariateGaussian.h>
#include <grasp_learning/fileHandler.h>

#include <std_srvs/Empty.h>
#include <std_msgs/Empty.h>
#include <grasp_learning/PolicySearch.h>
#include <grasp_learning/SetRBFN.h>
#include <grasp_learning/CallRBFN.h>
#include <grasp_learning/GetNetworkWeights.h>
#include <visualization_msgs/MarkerArray.h>
#include <fstream>

#define PI 3.14159265358979323846
// #define ACTIVATION_THRESHOLD 0.2
// #define COVERGANCE_THRESHOLD 0.4

namespace demo_learning {
namespace RBFNetwork {

// namespace GaussianKernel{
class GaussianKernel {
  public:
	GaussianKernel();
	GaussianKernel(Eigen::VectorXd, double);
	~GaussianKernel() {};
	double residual(Eigen::VectorXd);
	double kernelActivation(Eigen::VectorXd);
	Eigen::VectorXd getMean();
	Eigen::MatrixXd getCovar();
	double getVar();
  private:
	Eigen::VectorXd mean;
	Eigen::MatrixXd covar;
	double var;

};

class RBFNetwork {
  public:
	RBFNetwork();
	~RBFNetwork() {};

	Eigen::MatrixXd getKernelWeights();
	bool updateWeights(Eigen::MatrixXd);

	void resetRunningWeights();
	int getNumKernels();
	void printAllKernelMean();
	void printKernelMean(int kernel);
	Eigen::MatrixXd sampleNoise();
	double calculateVariance(double dx, double dy, double dz = 0);
	void updateNoiseVariance();

	bool policyConverged(Eigen::MatrixXd, Eigen::MatrixXd);

	double meanOfVector(const std::vector<double>& vec);
	void setNoiseVariance(const double variance);

	void resetRollout();

	bool createFiles(std::string relPath);

	template<typename T>
	void saveDataToFile(std::string filename, T, bool);

	void saveKernelsToFile();

	void spaceKernelsOnPlane(double height);

	void spaceKernelsOnGrid();

	void spaceKernelsOnManifold(double height, double radius);

	Eigen::MatrixXd ConvertToEigenMatrix(std::vector<std::vector<double>> data);

	Eigen::MatrixXd concatenateMatrices(std::vector<Eigen::MatrixXd> data);

	void printVector(std::vector<double> vec);
	bool printAllKernelMeans(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);
	bool addWeightNoise(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);
	bool printKernelWeights(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);
	bool buildRBFNetwork(grasp_learning::SetRBFN::Request& req, grasp_learning::SetRBFN::Response& res);
	bool networkOutput(grasp_learning::CallRBFN::Request& req, grasp_learning::CallRBFN::Response& res);
	bool policySearch(grasp_learning::PolicySearch::Request& req, grasp_learning::PolicySearch::Response& res);
	bool getNetworkWeights(grasp_learning::GetNetworkWeights::Request& req, grasp_learning::GetNetworkWeights::Response& res);
	bool getRunningWeights(grasp_learning::GetNetworkWeights::Request& req, grasp_learning::GetNetworkWeights::Response& res);
	bool visualizeKernelMeans(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);
	bool resetRBFNetwork(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);
	bool printRunningNoise(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);

  private:

	ros::NodeHandle nh;
	ros::NodeHandle nh_;

	ros::ServiceServer print_kernel_means_srv_;
	ros::ServiceServer add_weight_noise_srv_;
	ros::ServiceServer print_kernel_weights_srv_;
	ros::ServiceServer build_RBF_network_srv_;
	ros::ServiceServer network_output_srv_;
	ros::ServiceServer policy_search_srv_;
	ros::ServiceServer get_network_weights_srv_;
	ros::ServiceServer get_running_weights_srv_;
	ros::ServiceServer vis_kernel_mean_srv_;
	ros::ServiceServer reset_RBFN_srv_;
	ros::ServiceServer print_running_noise_srv_;

	ros::Publisher marker_pub;


	std::default_random_engine generator;
	std::normal_distribution<double> dist;

	std::vector<GaussianKernel> Network;
	Eigen::MatrixXd rollout_noise;
	Eigen::MatrixXd weights;
	Eigen::MatrixXd runningWeights;
	Eigen::MatrixXd kernelOutput;
	std::vector<std::vector<double> > networkOutput_;
	std::vector<Eigen::MatrixXd> noisePerTimestep_;

	std::vector<double> global_pos;
	double manifold_height = 0;
	power PoWER;
	MultiVariateGaussian multiVarGauss;
	fileHandler fileHandler_;

	int numKernels = 0;

	int numPolicies = 0;
	double intialNoiceVar = 0;
	int numTrial_ = 1;
	bool useCorrNoise;
	int numRows;
	int numDim;

	int burnInTrials;
	int maxNumSamples;

	double grid_x_;
	double grid_y_;
	double grid_z_;

	double conv_threshold_;

	std::string relativePath;
	std::string kernelTotalActivationPerTimeFile;
	std::string kernelWiseTotalActivationFile;
	std::string kernelOutputFile;
	std::string rewardsOutputFile;
	std::string networkOutputFile;
	std::string runningWeightsFile;
	std::string networkWeightsFile;
	std::string noiseFile;
	std::string krenelMeanFile;
	std::string krenelCovarFile;
	std::string numKernelFile;
	std::string spacingPolicy_;
	std::string noisePerTimeStepFile;
	bool coverged = false;
};

}
}

#endif // include guard