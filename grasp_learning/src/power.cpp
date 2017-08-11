#include <grasp_learning/power.h>

power::power(int kernels, int initialRollouts, int samples){
	num_of_kernels=kernels;
	num_initial_rollouts = initialRollouts;
	max_num_samples = samples;
}

void power::setParams(int kernels, int initialRollouts, int samples, int numPolicies){
	num_of_kernels=kernels;
	num_initial_rollouts = initialRollouts;
	max_num_samples = samples;
	num_policies = numPolicies;
	noises.resize(numPolicies);
	for(int i =0;i<numPolicies;i++){
		noises.push_back(Eigen::MatrixXd::Zero(num_of_kernels, 0));
	}
}

bool power::operator()(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem){
	if (firstElem.first>secondElem.first)
		return 1;
	else if(firstElem.first<secondElem.first)
		return 0;
	else
		return firstElem.second>secondElem.second;
}

// std::vector<double> power::policySearch(const std::vector<double> currNoise,const double reward,const std::vector<double> affectedKernels)
// {


// 	imp_sampler.push_back(std::make_pair(reward,curr_int++));
// 	rewards.insert(rewards.begin(),reward);
// 	noises.conservativeResize(num_of_kernels, noises.cols()+1);
// 	noises.col(noises.cols()-1).setZero();
// 	for(int i = 0; i<num_of_kernels;i++){
// 		noises(i,noises.cols()-1) = currNoise[i];
// 	}

// 	std::sort(imp_sampler.begin(),imp_sampler.end(),power());
// 	print_imp_sampler(imp_sampler);

// 	int num_imp_sampler_noise = (imp_sampler.size()>= max_num_samples ? max_num_samples:imp_sampler.size());

// 	if (curr_int<=num_initial_rollouts){
// 		ROS_INFO("Burn in trial %d/%d",(int)curr_int, num_initial_rollouts);
// 		std::vector<double> res(num_of_kernels, 0.0);
// 		return res;
// 	}

// 	double param_dnom_weights =0;
// 	double param_dnom_noise = 0;
// 	double var_dnom = 0;
// 	double curr_reward=0;
// 	int curr_idx = 0;

// 	Eigen::VectorXd param_nom(num_of_kernels);

// 	param_nom.setZero();
// 	for (int elem=0; elem<num_imp_sampler_noise;elem++){
// 		curr_reward = imp_sampler[elem].first;
// 		curr_idx = imp_sampler[elem].second;
// 		param_nom += curr_reward*noises.col(curr_idx);
// 		param_dnom_weights += curr_reward;
// 	}

// 	double min = std::numeric_limits<double>::denorm_min();

// 	param_nom/=(param_dnom_weights+min);

// 	std::vector<double> update_weights(param_nom.data(), param_nom.data() + param_nom.size());

// 	return update_weights;
// }

Eigen::MatrixXd power::policySearch(const Eigen::MatrixXd currNoise,const double reward,const std::vector<double> affectedKernels)
{


	imp_sampler.push_back(std::make_pair(reward,curr_int++));
	rewards.insert(rewards.begin(),reward);

	for(int i =0;i<num_policies;i++){
		noises[i].conservativeResize(num_of_kernels, noises[i].cols()+1);
		noises[i].col(noises[i].cols()-1).setZero();
		noises[i].col(noises[i].cols()-1) = currNoise.col(i);
	}

	std::sort(imp_sampler.begin(),imp_sampler.end(),power());
	print_imp_sampler(imp_sampler);

	int num_imp_sampler_noise = (imp_sampler.size()>= max_num_samples ? max_num_samples:imp_sampler.size());

	if (curr_int<=num_initial_rollouts){
		ROS_INFO("Burn in trial %d/%d",(int)curr_int, num_initial_rollouts);
		Eigen::MatrixXd res = Eigen::MatrixXd::Zero(num_of_kernels, num_policies);
		return res;
	}

	double param_dnom_weights =0;
	double param_dnom_noise = 0;
	double var_dnom = 0;
	double curr_reward=0;
	int curr_idx = 0;

	Eigen::MatrixXd param_nom = Eigen::MatrixXd::Zero(num_of_kernels, num_policies);
	for (int elem=0; elem<num_imp_sampler_noise;elem++){
		curr_reward = imp_sampler[elem].first;
		curr_idx = imp_sampler[elem].second;
		param_dnom_weights += curr_reward;
		for(int i =0;i<num_policies;i++){
			param_nom.col(i) += curr_reward*noises[i].col(curr_idx);
		}
	}

	double min = std::numeric_limits<double>::denorm_min();

	param_nom/=(param_dnom_weights+min);

	return param_nom;
}


void power::print_imp_sampler(std::vector<std::pair<double,int> > imp_sampler){
	int idx = 0;
	std::cout<<"Importance sampler: "<<std::endl;
	for(const auto& elem: imp_sampler){
		std::cout<<elem.second<<" "<<elem.first<<std::endl;
		idx++;
		if (idx>=max_num_samples)
			break;
	}
	std::cout<<std::endl;
}

void power::clear_data(){

	for(int i =0;i<num_policies;i++){
		noises.push_back(Eigen::MatrixXd::Zero(num_of_kernels, 0));
	}
	imp_sampler.clear();
	curr_int=0;
}
