#ifndef RBFN_H
#define RBFN_H

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "ros/ros.h"
#include <math.h>
#include <grasp_learning/power.h>
#include <grasp_learning/MultiVariateGaussian.h>
#include <std_srvs/Empty.h>
#include <std_msgs/Empty.h>
#include <grasp_learning/PolicySearch.h>
#include <grasp_learning/SetRBFN.h>
#include <grasp_learning/CallRBFN.h>
#include <grasp_learning/GetNetworkWeights.h>
#include <visualization_msgs/MarkerArray.h>
#include <fstream>

#define PI 3.14159265358979323846
#define ACTIVATION_THRESHOLD 0.1
#define COVERGANCE_THRESHOLD 0.8
namespace demo_learning {
	namespace RBFNetwork {

// namespace GaussianKernel{
		class GaussianKernel
		{
		public:
			GaussianKernel();
			GaussianKernel(Eigen::VectorXd, Eigen::MatrixXd);
			~GaussianKernel(){};
			Eigen::VectorXd residual(Eigen::VectorXd);
			double kernelActivation(Eigen::VectorXd);
			Eigen::VectorXd getMean();
			double getVar();
		private:
	// Eigen::Vector3d mean;
	// Eigen::Matrix3d covar;
			Eigen::VectorXd mean;
			Eigen::MatrixXd covar;

		};
// }

// namespace RBFNetwork{

		class RBFNetwork
		{
		public:
			RBFNetwork();
			~RBFNetwork(){};

			std::vector<double> getActiveKernels();
			Eigen::MatrixXd getKernelWeights();
			void updateWeights(Eigen::MatrixXd);

			void resetRunningWeights();
			int getNumKernels();
			void printAllKernelMean();
			void printKernelMean(int kernel);
			Eigen::MatrixXd sampleNoise();
			double calculateVariance(double dx, double dy);
			void updateNoiseVariance();

			bool policyConverged();

			double meanOfVector(const std::vector<double>& vec);
			void setNoiseVariance(const double variance);

			void resetRollout();

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

			ros::Publisher marker_pub;


			std::default_random_engine generator;
			std::normal_distribution<double> dist;

			std::vector<GaussianKernel> Network;
			Eigen::MatrixXd rollout_noise;
			Eigen::MatrixXd weights;
			std::vector<double> activeKernels;
			Eigen::MatrixXd runningWeights;
			Eigen::MatrixXd kernelOutput;
			std::vector<double> global_pos;
			double manifold_height = 0;
			power PoWER;
			MultiVariateGaussian multiVarGauss;

			int numKernels = 0;

			int numPolicies = 0;
			double intialNoiceVar = 0;
			bool useCorrNoise;
			int numRows;
			int numDim;

			int burnInTrials;
			int maxNumSamples;

			std::string kernelOutputFileName;
			std::string rewardsOutputFileName;
			std::string relativePath;

			bool coverged = false;
		};

	}
}

#endif // include guard