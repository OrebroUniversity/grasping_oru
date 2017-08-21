#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp> 
#include <ctime>



class MultiVariateGaussian{
public:
  MultiVariateGaussian(Eigen::MatrixXd covariance, Eigen::VectorXd mean);
  MultiVariateGaussian();

  void reset(Eigen::MatrixXd covariance, Eigen::VectorXd mean);
  Eigen::MatrixXd sample(int num_samples);
  void store_samples_to_file(Eigen::MatrixXd samples);


  ~MultiVariateGaussian();


  Eigen::MatrixXd get_cov();

  Eigen::VectorXd get_mean();

  void print_cov();

  void print_mean();

  void setCovarAsDiffernceMatrix(int size, double beta);

  void setCovarAsIndentityMatrix(int size, double beta);

private:
  Eigen::MatrixXd cov;
  Eigen::VectorXd mu;
  int rows;
  int cols;
  int num_samples;


};