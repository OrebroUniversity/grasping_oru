#include <grasp_learning/MultiVariateGaussian.h>
using namespace std;
int total_samples = 0;
namespace Eigen {
  namespace internal {
template<typename Scalar> 


    struct scalar_normal_dist_op 
    {
  static boost::mt19937 rng;    // The uniform pseudo-random algorithm
  mutable std::normal_distribution<Scalar> norm;  // The gaussian combinator

  EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

  template<typename Index>
  inline const Scalar operator() (Index, Index = 0) const {
   return norm(rng); }
 };

template<typename Scalar> boost::mt19937 scalar_normal_dist_op<Scalar>::rng(static_cast<unsigned int>(std::time(0)));

template<typename Scalar>
 struct functor_traits<scalar_normal_dist_op<Scalar> >
 { enum { Cost = 50 * NumTraits<Scalar>::MulCost, PacketAccess = false, IsRepeatable = false }; };
} // end namespace internal
} // end namespace Eigen


/*
Parametrized constructor. The function initializes a multivariate Gaussian distribution with the
given covariance matrix and mean vector as the parameters.

@param Eigen::MatrixXd covariance: A matrix representing the covariance matrix of the distribution.
@param Eigen::VectorXd mean: A vector representing the mean vector of the distribution. 

*/
MultiVariateGaussian::MultiVariateGaussian(Eigen::MatrixXd covariance, Eigen::VectorXd mean):
cov(covariance), mu(mean){
	rows = cov.rows();
	cols = cov.cols();
	num_samples = 0;
}

/*
Default contructor. This constructor intializes an empty covariance matrix and mean vector. 
*/

MultiVariateGaussian::MultiVariateGaussian()
{
	rows = 0;
	cols = 0;
	num_samples = 0;
}

/*
This method resets the covariance matrix and mean vector to a new covariance matrix and mean vector.
By calling this method the distribution can be changed without the need to implicitly creat new objects
every time the distribution is changed.

@param Eigen::MatrixXd covariance: A matrix representing the new covariance matrix of the distribution.
@param Eigen::VectorXd mean: A vector representing the new mean vector of the distribution. 
*/
void MultiVariateGaussian::reset(Eigen::MatrixXd covariance, Eigen::VectorXd mean){
  cov.setZero();
  cov = covariance;
  mu = mean.setZero();
  rows = cov.rows();
  cols = cov.cols();
}


/*
This function samples a number of samples (given as the parameter) from the multivariate Gaussian distribution.

@param int num_samples: the number of samples drawn from the distribution

returns the samples drawn from the distribution as a vector of doubles. 
*/
Eigen::MatrixXd MultiVariateGaussian::sample(int num_samples){


  Eigen::internal::scalar_normal_dist_op<double> randN; // Gaussian functor
  // Eigen::internal::scalar_normal_dist_op<double>::rng.seed(1); // Seed the rng
  Eigen::MatrixXd normTransform(rows,rows);

  Eigen::LLT<Eigen::MatrixXd> cholSolver(cov);
  // We can only use the cholesky decomposition if 
  // the covariance matrix is symmetric, pos-definite.
  // But a covariance matrix might be pos-semi-definite.
  // In that case, we'll go to an EigenSolver
  if (cholSolver.info()==Eigen::Success) {
    // Use cholesky solver
    normTransform = cholSolver.matrixL();
  } else {
    // Use eigen solver
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(cov);
    normTransform = eigenSolver.eigenvectors() 
    * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
  }
  Eigen::MatrixXd samples = (normTransform 
   * Eigen::MatrixXd::NullaryExpr(rows,num_samples,randN)).colwise() 
  + mu;

   // cout<<samples<<endl;
  // std::cout << "Mean\n" << mean << std::endl;
  // std::cout << "Covar\n" << R << std::endl;
    // std::cout << "Samples\n" << samples << std::endl;
  // store_samples_to_file(samples);
  total_samples++;
  return samples;
}

/*
With this method the samples drawn from the distribution can be stored in a file.

@param Eigen::MatrixXd samples: The samples drawn using the MultiVariateGaussian::sample method.

*/
// void MultiVariateGaussian::store_samples_to_file(Eigen::MatrixXd samples){
// 	ofstream rand_sample_file;
// 	string index_str;          // string which will contain the result
// 	ostringstream convert;   // stream used for the conversion
// 	convert << total_samples;      // insert the textual representation of 'Number' in the characters in the stream
// 	index_str = convert.str(); // set 'Result' to the contents of the stream

// 	string samples_file_name = "/home/intelligentrobotics/ws/pbd/Applications/bagfiles/Jens/samples_"+index_str+".dat";
// 	rand_sample_file.open(samples_file_name);
// 	for (size_t i = 0, nRows = samples.rows(), nCols = samples.cols(); i < nRows; ++i){
//   		for (size_t j = 0; j < nCols; ++j){
//   			rand_sample_file<< samples(i,j)<<" ";
//   		}
//   		rand_sample_file<<endl;
// 	}
// 	rand_sample_file.close();

// }

MultiVariateGaussian::~MultiVariateGaussian(){  }


Eigen::MatrixXd MultiVariateGaussian::get_cov(){
	return cov;
}

Eigen::VectorXd MultiVariateGaussian::get_mean(){
	return mu;
}

void MultiVariateGaussian::print_cov(){
	cout<<cov<<endl;
}

void MultiVariateGaussian::print_mean(){
	cout<<mu<<endl;
}

void MultiVariateGaussian::setCovarAsIndentityMatrix(int size, double beta){
  cov = Eigen::MatrixXd::Identity(size,size);
  cov *= beta;
  mu = Eigen::VectorXd::Zero(size);
  rows = cov.rows();
  cols = cov.cols();
}

void MultiVariateGaussian::setCovarAsDiffernceMatrix(int size, double beta){
  Eigen::MatrixXd A1 = Eigen::MatrixXd::Zero(size+2,size);
  for(int row = 0;row<size+2;row++){
    for(int col =0;col<size;col++){
      if(row == col){
        A1(row,col) = -0.5;
      }
      if((row-2) == col){
        A1(row,col) = 0.5;
      }

    }
  }
  cov = beta*(A1.transpose()*A1).inverse();
  mu = Eigen::VectorXd::Zero(size);
  rows = cov.rows();
  cols = cov.cols();
}