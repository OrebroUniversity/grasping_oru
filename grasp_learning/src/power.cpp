#include <grasp_learning/power.h>

power::power(int kernels, int initialRollouts, int samples) {
	num_of_kernels = kernels;
	num_initial_rollouts = initialRollouts;
	max_num_samples = samples;

}

void power::setParams(int kernels, int initialRollouts, int samples, int numPolicies) {
	num_of_kernels = kernels;
	num_initial_rollouts = initialRollouts;
	max_num_samples = samples;
	num_policies = numPolicies;
	noises.resize(numPolicies);
	for (int i = 0; i < numPolicies; i++) {
		noises.push_back(Eigen::MatrixXd::Zero(num_of_kernels, 0));
	}
}

bool power::operator()(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem) {
	if (firstElem.first > secondElem.first)
		return 1;
	else if (firstElem.first < secondElem.first)
		return 0;
	else
		return firstElem.second > secondElem.second;
}

Eigen::MatrixXd power::policySearch(const Eigen::MatrixXd currNoise, const std::vector<double> reward, const  Eigen::MatrixXd kernelActivation) {
	// std::cout<<kernelActivation<<std::endl;
	// double sum_of_rewards = std::accumulate(reward.begin(), reward.end(), 0.0);

	imp_sampler.push_back(std::make_pair(reward[0], curr_int++));

	rewards.push_back(reward);
	kernelActivations.push_back(kernelActivation);

	for (int i = 0; i < num_policies; i++) {
		noises[i].conservativeResize(num_of_kernels, noises[i].cols() + 1);
		noises[i].col(noises[i].cols() - 1).setZero();
		noises[i].col(noises[i].cols() - 1) = currNoise.col(i);
	}
	// noises.push_back(currNoise);

	std::sort(imp_sampler.begin(), imp_sampler.end(), power());
	printImpSampler(imp_sampler);

	int num_imp_sampler_noise = (imp_sampler.size() >= max_num_samples ? max_num_samples : imp_sampler.size());

	if (curr_int <= num_initial_rollouts) {
		ROS_INFO("Burn in trial %d/%d", (int)curr_int, num_initial_rollouts);
		Eigen::MatrixXd res = Eigen::MatrixXd::Zero(num_of_kernels, num_policies);
		return res;
	}

	int idx = 0;


	Eigen::MatrixXd num = Eigen::MatrixXd::Zero(num_of_kernels, num_policies);
	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(num_of_kernels, num_of_kernels);
	Eigen::MatrixXd B = Eigen::MatrixXd::Zero(num_of_kernels, num_policies);

	// Eigen::MatrixXd W = Eigen::MatrixXd::Zero(num_of_kernels, num_of_kernels);

	for (int elem = 0; elem < num_imp_sampler_noise; elem++) {
		idx = imp_sampler[elem].second;
		Eigen::MatrixXd C = Eigen::MatrixXd::Zero(num_of_kernels, num_of_kernels);

		populateMap(idx);
		C = CMat.find(idx)->second;
		A += C;
		for (int i = 0; i < num_policies; i++) {
			B.col(i) += C * noises[i].col(idx);
		}
	}
	Eigen::MatrixXd min_matrix = Eigen::MatrixXd::Identity(num_of_kernels, num_of_kernels);

	double min = 1e-10;

	Eigen::MatrixXd invMat = (A + min * min_matrix).inverse();
	Eigen::MatrixXd new_weights = Eigen::MatrixXd::Zero(num_of_kernels, num_policies);
	// std::cout<<invMat<<std::endl;
	// std::cout<<B<<std::endl;
	for (int i = 0; i < num_policies; i++) {
		new_weights.col(i) = invMat * B.col(i);
	}

	return new_weights;
}

void power::populateMap(int idx) {

	std::map<int, Eigen::MatrixXd>::iterator it;
	it = CMat.find(idx);
	if (it == CMat.end()) {
		Eigen::MatrixXd C = Eigen::MatrixXd::Zero(num_of_kernels, num_of_kernels);
		for (unsigned int t = 0; t < rewards[idx].size(); t++) {
			C += kernelActivations[idx].col(t)*kernelActivations[idx].col(t).transpose()*rewards[idx][t];
			// C += kernelActivations[idx].col(t) * kernelActivations[idx].col(t).transpose();
		}
		CMat[idx] = C;
		// CMat[idx] = C * imp_sampler[idx].first;
		if (imp_sampler.size() > max_num_samples) {
			int idx2del = imp_sampler[max_num_samples + 1].second;
			std::cout << "Removing index " << idx2del << std::endl;
			CMat.erase(idx2del);
		}
	}
}

Eigen::MatrixXd power::policySearch2(const std::vector<Eigen::MatrixXd> noisePerTimeStep, const std::vector<double> reward) {
	// std::cout<<kernelActivation<<std::endl;
	// double sum_of_rewards = std::accumulate(reward.begin(), reward.end(), 0.0);

	imp_sampler.push_back(std::make_pair(reward[0], curr_int++));

	rewards.push_back(reward);

	noisePerTimeStepVec.push_back(noisePerTimeStep);

	std::sort(imp_sampler.begin(), imp_sampler.end(), power());
	printImpSampler(imp_sampler);

	int num_imp_sampler_noise = (imp_sampler.size() >= max_num_samples ? max_num_samples : imp_sampler.size());

	if (curr_int <= num_initial_rollouts) {
		ROS_INFO("Burn in trial %d/%d", (int)curr_int, num_initial_rollouts);
		Eigen::MatrixXd res = Eigen::MatrixXd::Zero(num_of_kernels, num_policies);
		return res;
	}

	int idx = 0;


	Eigen::MatrixXd num = Eigen::MatrixXd::Zero(num_of_kernels, num_policies);
	double dnom = 0;

	for (int elem = 0; elem < num_imp_sampler_noise; elem++) {
		idx = imp_sampler[elem].second;
		Eigen::MatrixXd C = Eigen::MatrixXd::Zero(num_of_kernels, num_policies);
		double Q = 0;
		for (unsigned int t = 0; t < rewards[idx].size(); t++) {
			Q = rewards[idx][t];
			for (int i = 0; i < num_policies; i++) {
				C.col(i) += noisePerTimeStepVec[idx][t].col(i) * Q ;
			}
			dnom += Q;
		}
		num += C;
	}
	double min = std::numeric_limits<double>::denorm_min();

	Eigen::MatrixXd new_weights = Eigen::MatrixXd::Zero(num_of_kernels, num_policies);
	// std::cout<<invMat<<std::endl;
	// std::cout<<B<<std::endl;
	for (int i = 0; i < num_policies; i++) {
		new_weights.col(i) = num.col(i) / (dnom + min);
	}

	return new_weights;
}


std::vector<double> power::getHighestRewards() {
	std::vector<double> res;
	int num_imp_sampler_noise = (imp_sampler.size() >= max_num_samples ? max_num_samples : imp_sampler.size());
	for (int elem = 0; elem < num_imp_sampler_noise; elem++) {
		res.push_back(imp_sampler[elem].first);
	}
	return res;
}

double power::varianceSearch() {
	int num_imp_sampler_noise = (imp_sampler.size() >= max_num_samples ? max_num_samples : imp_sampler.size());
	if (curr_int <= num_initial_rollouts) {
		return 1;
	} else {
		double beta = 0;
		double curr_reward = 0;
		for (int elem = 0; elem < 5; elem++) {
			curr_reward = imp_sampler[elem].first;
			beta += curr_reward;
		}
		if (beta < 1) {
			beta = 1;
		}

		return 1.0 / beta;
	}
}


void power::printImpSampler(std::vector<std::pair<double, int> > imp_sampler) {
	int idx = 0;
	std::cout << "Importance sampler: " << std::endl;
	for (const auto& elem : imp_sampler) {
		std::cout << elem.second << " " << elem.first << std::endl;
		idx++;
		if (idx >= max_num_samples)
			break;
	}
	std::cout << std::endl;
}

void power::resetPolicySearch() {

	curr_int = 0;
	rewards.clear();
	kernelActivations.clear();
	CMat.clear();
	noises.resize(num_policies);
	for (int i = 0; i < num_policies; i++) {
		noises.push_back(Eigen::MatrixXd::Zero(num_of_kernels, 0));
	}
	imp_sampler.clear();
	noisePerTimeStepVec.clear();
}

double power::getNumRollouts() {
	return curr_int;
}
