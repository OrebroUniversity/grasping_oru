#pragma once

#include <vector>
#include <iostream>
#include <utility>
#include <algorithm>
#include <math.h>    
#include <Eigen/Dense>
#include "ros/ros.h"

class power
{
public:
	power(){};
	power(int kernels, int initialRollouts, int samples);
	~power(){};
	bool operator()(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem);
	// std::vector<double> policySearch(const std::vector<double> noise,const double rewards,const std::vector<double> affectedKernels);
	Eigen::MatrixXd policySearch(const Eigen::MatrixXd noise,const double rewards,const std::vector<double> affectedKernels);

	void print_imp_sampler(std::vector<std::pair<double,int> > imp_sampler);
	void clear_data();
	// void setParams(int, int, int);
	void setParams(int, int, int, int);

private:
	double curr_int = 0;
	int num_of_kernels = 0;
	int num_initial_rollouts = 0;
	int max_num_samples = 0;
	int num_policies = 0;
	std::vector<double> rewards;
	std::vector<std::pair<double,int> > imp_sampler;
	// Eigen::MatrixXd noises;
	std::vector<Eigen::MatrixXd> noises;
};