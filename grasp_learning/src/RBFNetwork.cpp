#include <grasp_learning/RBFNetwork.h>

GaussianKernel::GaussianKernel(Eigen::Vector3d mean, Eigen::Matrix3d covar){
	this->mean = mean;
	this->covar = covar;
}
Eigen::Vector3d GaussianKernel::residual(Eigen::Vector3d x){
	return x-mean;
}

double GaussianKernel::kernelActivation(Eigen::Vector3d x){
	Eigen::Vector3d residual = this->residual(x);
	// std::cout<<"Residual norm "<<residual.norm()<<std::endl;
	double activation = exp(-0.5*residual.transpose()*covar.inverse()*residual);
	// std::cout<<"Activation "<<activation<<std::endl;
	if (activation>ACTIVATION_THRESHOLD){
		return activation;
	}
	else{
		return 0;
	}
}

void RBFNetwork::buildRBFNetwork(int numKernels, int numRows, double radius, double height, std::vector<double> globalPos){
	double column_spacing = (2*PI)/numKernels;
	double row_spacing = height/(numRows+1);
	Eigen::Vector3d mean;
	Eigen::Matrix3d covar;
	covar<<1,0,0,
		   0,1,0,
		   0,0,1;
	for (int row = 1;row<=numRows;row++){
		mean(2) = globalPos[2]+row_spacing*row;
		for (int column=0;column<numKernels;column++){
			mean(0)= globalPos[0]+radius*cos(column*column_spacing);
			mean(1)= globalPos[1]+radius*sin(column*column_spacing);
			GaussianKernel kernel(mean, covar);
			Network.push_back(kernel);
			weights.push_back(1);
			this->numKernels++;
		}
	}
	runningWeights = weights;
	std::cout<<this->numKernels<<" kernels were created"<<std::endl;
}

double RBFNetwork::networkOutput(Eigen::Vector3d x){
	activeKernels.clear();
	double output = 0;
    for (int i = 0; i < Network.size(); i++) {
    	double activation = Network[i].kernelActivation(x);
    	if (activation != 0){
    		if (std::find(activeKernels.begin(), activeKernels.end(), i) == activeKernels.end())
  				activeKernels.push_back(i);
    	}
    	output += runningWeights[i]*activation;
    }
    std::cout<<"Network output: "<<output<<std::endl;
    // for (int i = 0; i < activeKernels.size(); i++) {
    // 	std::cout<<activeKernels[i]<<std::endl;
    // }

}

std::vector<double> RBFNetwork::getKernelWeights(){
	return weights;
}

void RBFNetwork::printKernelWeights(){
	for (auto iter:weights){
		 std::cout << iter << std::endl;
	}
}

std::vector<double> RBFNetwork::getActiveKernels(){
	return activeKernels;
}

void RBFNetwork::addWeightNoise(std::vector<double> noise){
	for(int i=0;i<noise.size();i++){
		runningWeights[i]+=noise[i];
	}
}

void RBFNetwork::resetRunningWeights(){
	runningWeights = weights;
}

void RBFNetwork::updateWeights(std::vector<double> newWeights){
	for(int i=0;i<newWeights.size();i++){
		weights[i]+=newWeights[i];
	}
}

std::vector<double> RBFNetwork::getRunningWeights(){
	return runningWeights;
}