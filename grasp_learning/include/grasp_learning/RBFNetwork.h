#pragma once

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "ros/ros.h"
#include <math.h>

#define PI 3.14159265358979323846
#define ACTIVATION_THRESHOLD 0.1


// namespace GaussianKernel{
class GaussianKernel
{
public:
	GaussianKernel();
	GaussianKernel(Eigen::Vector3d, Eigen::Matrix3d);
	~GaussianKernel(){};
	Eigen::Vector3d residual(Eigen::Vector3d);
	double kernelActivation(Eigen::Vector3d);
private:
	Eigen::Vector3d mean;
	Eigen::Matrix3d covar;
};
// }

// namespace RBFNetwork{

class RBFNetwork
{
public:
	RBFNetwork(){};
	~RBFNetwork(){};
	void buildRBFNetwork(int numKernels, int numRows, double radius, double height, std::vector<double> globalPos);
	double networkOutput(Eigen::Vector3d);
	std::vector<double> getActiveKernels();
	std::vector<double> getKernelWeights();
	void printKernelWeights();
	void addWeightNoise(std::vector<double> noise);
	void updateWeights(std::vector<double> newWeights);
	void resetRunningWeights();
	std::vector<double> getRunningWeights();
private:
	int numKernels = 0;
	std::vector<GaussianKernel> Network;
	std::vector<double> weights;
	std::vector<double> activeKernels;
	std::vector<double> runningWeights;
};

// }

