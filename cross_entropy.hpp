#ifndef __CROSS_ENTROPY
#define __CROSS_ENTROPY
#include <Eigen/Dense>
#include <Eigen/Core> // for using eigen in multithreads, run Eigen::initParallel() first.

void cross_entropy(const int n,
                   Eigen::VectorXd& theta,// initial mean parameter vector
                   Eigen::MatrixXd& cov,  // initial nxn covariance matrix
                   const int K,
                   const int E,
                   const int N,
                   double epsi,
                   double (*evalFunc)(Eigen::VectorXd, const int));

#endif
