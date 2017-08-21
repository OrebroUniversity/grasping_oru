#pragma once

#include <vector>
#include <iostream>
#include <utility>
#include <algorithm>
#include <math.h>    
#include <Eigen/Dense>
#include "ros/ros.h"
#include <numeric>
#include <grasp_learning/fileHandler.h>

class power
{
public:
	power(){};
	power(int kernels, int initialRollouts, int samples);
	~power(){};
	bool operator()(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem);
	double varianceSearch();

	// Eigen::MatrixXd policySearch(Eigen::MatrixXd noise,const double rewards);
	Eigen::MatrixXd policySearch(const Eigen::MatrixXd noise,const std::vector<double> rewards,const  Eigen::MatrixXd kernelActivation);

	void print_imp_sampler(std::vector<std::pair<double,int> > imp_sampler);
	void clear_data();
	// void setParams(int, int, int);
	void setParams(int, int, int, int, std::string);

	std::vector<double> getHighestRewards();

	double getNumRollouts();
private:
	double curr_int = 0;
	int num_of_kernels = 0;
	int num_initial_rollouts = 0;
	int max_num_samples = 0;
	int num_policies = 0;
	std::vector<std::vector<double>> rewards;
	// std::vector<double> rewards;

	std::vector<Eigen::MatrixXd> kernelActivations;

	std::vector<std::pair<double,int> > imp_sampler;
	// Eigen::MatrixXd noises;
	std::vector<Eigen::MatrixXd> noises;
	fileHandler fileHandler_;
	std::string matrixNum;
	std::string matrixNumRed;
	std::string matrixDnom;
};