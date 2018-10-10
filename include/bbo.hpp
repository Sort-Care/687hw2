#ifndef __BBO
#define __BBO
#include <Eigen/Dense>
#include <Eigen/Core> // for using eigen in multithreads, run Eigen::initParallel() first.
#include <queue>
#include <vector>
#include "conventions.hpp"
#include "multi_normal.hpp"
#include "grid.hpp"

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
                   void (*evalFunc)(struct policy&, // policy
                                    const int,      // number of episodes
                                    const int       // axis
                                    ));

void hill_climbing(Eigen::VectorXd& theta,// Initial mean policy parameter vector
                   const double tau,      // Exploration parameter
                   const int N,            // Number of episodes to evaluate
                   void (*evalFunc)(struct policy&,
                                    const int,
                                    const int)
                   );

void eval_grid_policy(struct policy& po,
                      const int num_episodes,
                      const int axis);

void eval_cart_pole_policy(struct policy& po,
                           const int num_episodes,
                           const int axis);

void save_data(const double J, const int axis);


#endif
