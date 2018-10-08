#ifndef __BBO
#define __BBO
#include <Eigen/Dense>
#include <Eigen/Core> // for using eigen in multithreads, run Eigen::initParallel() first.
#include <queue>
#include <vector>
#include "conventions.hpp"
#include "multi_normal.hpp"

struct policy {
        //policy parameters
    Eigen::VectorXd param;
    double J;
};


void cross_entropy(const int n,
                   Eigen::VectorXd& theta,// initial mean parameter vector
                   Eigen::MatrixXd& cov,  // initial nxn covariance matrix
                   const int K,
                   const int E,
                   const int N,
                   double epsi,
                   double (*evalFunc)(Eigen::VectorXd, const int));

void hill_climbing(Eigen::VectorXd& theta,// Initial mean policy parameter vector
                   const double tau,      // Exploration parameter
                   const int N            // Number of episodes to evaluate   
                   );

void eval_grid_policy(struct policy& po,
                      const int num_trials);


#endif
