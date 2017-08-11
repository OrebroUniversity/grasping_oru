#ifndef RBFN_H
#define RBFN_H

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "ros/ros.h"
#include <math.h>
#include <grasp_learning/power.h>
#include <std_srvs/Empty.h>
#include <std_msgs/Empty.h>
#include <grasp_learning/PolicySearch.h>
#include <grasp_learning/SetRBFN.h>
#include <grasp_learning/CallRBFN.h>
#include <grasp_learning/GetNetworkWeights.h>

#define PI 3.14159265358979323846
#define ACTIVATION_THRESHOLD 0.1

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
	// void buildRBFNetwork(int numKernels, int numRows, double radius, double height, std::vector<double> globalPos);
	// double networkOutput(Eigen::Vector3d);
			std::vector<double> getActiveKernels();
	// std::vector<double> getKernelWeights();
			Eigen::MatrixXd getKernelWeights();


	// void printKernelWeights();
	// void addWeightNoise(std::vector<double> noise);
	// void updateWeights(std::vector<double> newWeights);
			void updateWeights(Eigen::MatrixXd);

			void resetRunningWeights();
			int getNumKernels();
			void printAllKernelMean();
			void printKernelMean(int kernel);
	// std::vector<double> sampleNoise();
			Eigen::MatrixXd sampleNoise();


			void printVector(std::vector<double> vec);
			bool printAllKernelMeans(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);
			bool addWeightNoise(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);
			bool printKernelWeights(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);
			bool buildRBFNetwork(grasp_learning::SetRBFN::Request& req, grasp_learning::SetRBFN::Response& res);
			bool networkOutput(grasp_learning::CallRBFN::Request& req, grasp_learning::CallRBFN::Response& res);
			bool policySearch(grasp_learning::PolicySearch::Request& req, grasp_learning::PolicySearch::Response& res);
			bool getNetworkWeights(grasp_learning::GetNetworkWeights::Request& req, grasp_learning::GetNetworkWeights::Response& res);
			bool getRunningWeights(grasp_learning::GetNetworkWeights::Request& req, grasp_learning::GetNetworkWeights::Response& res);


		private:
			
			ros::NodeHandle nh;
			ros::ServiceServer print_kernel_means_srv_;
			ros::ServiceServer add_weight_noise_srv_;
			ros::ServiceServer print_kernel_weights_srv_;
			ros::ServiceServer build_RBF_network_srv_;
			ros::ServiceServer network_output_srv_;
			ros::ServiceServer policy_search_srv_;
			ros::ServiceServer get_network_weights_srv_;
			ros::ServiceServer get_running_weights_srv_;



			std::default_random_engine generator;
			std::normal_distribution<double> dist;

			int numKernels = 0;
			std::vector<GaussianKernel> Network;
			Eigen::MatrixXd rollout_noise;
	// std::vector<double> rollout_noise;
			Eigen::MatrixXd weights;
	// std::vector<double> weights;
			std::vector<double> activeKernels;
	// std::vector<double> runningWeights;
			Eigen::MatrixXd runningWeights;
			int numPolicies = 0;
			power PoWER;
		};

	}
}

#endif // include guard