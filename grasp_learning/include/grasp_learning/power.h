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
#include <map>

class power {
  public:
	power() {};
	power(int kernels, int initialRollouts, int samples);
	~power() {};
	bool operator()(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem);
	double varianceSearch();

	Eigen::MatrixXd policySearch(const Eigen::MatrixXd noise, const std::vector<double> rewards, const  Eigen::MatrixXd kernelActivation);
	Eigen::MatrixXd policySearch2(const std::vector<Eigen::MatrixXd> noisePerTimeStep, const std::vector<double> reward);

	void printImpSampler(std::vector<std::pair<double, int> > imp_sampler);
	void resetPolicySearch();

	void setParams(int, int, int, int);

	std::vector<double> getHighestRewards();

	double getNumRollouts();

	void populateMap(int idx);

  private:
	double curr_int = 0;
	int num_of_kernels = 0;
	int num_initial_rollouts = 0;
	int max_num_samples = 0;
	int num_policies = 0;
	std::vector<std::vector<double>> rewards;
	// std::vector<double> rewards;

	std::vector<Eigen::MatrixXd> kernelActivations;

	std::vector<std::pair<double, int> > imp_sampler;
	// Eigen::MatrixXd noises;
	std::map<int, Eigen::MatrixXd> CMat;


	std::vector<Eigen::MatrixXd> noises;
	std::vector<std::vector<Eigen::MatrixXd>> noisePerTimeStepVec;
	fileHandler fileHandler_;
	std::string matrixNum;
	std::string matrixNumRed;
	std::string matrixDnom;
};