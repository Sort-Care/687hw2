#ifndef __BBO
#define __BBO
#include <Eigen/Dense>
#include <Eigen/Core> // for using eigen in multithreads, run Eigen::initParallel() first.
#include <queue>
#include <vector>
#include <thread>
#include <future>
#include <string>
#include <fstream>
#include <mutex>


#include "conventions.hpp"
#include "eigen_multi_norm.hpp"
#include "grid.hpp"
#include "cartpole.hpp"

struct policy {
        //policy parameters
    Eigen::VectorXd param;
    double J;
};

/*
 * Class for comparing policies in the priority queue
 *
 */
class policy_compare{
public:
    bool operator() (struct policy p1, struct policy p2){
        if (p1.J < p2.J){
            return true;
        }else{
            return false;
        }
    }
};


void cross_entropy(const std::string prefix,
                   const int trial,
                   const int n,
                   Eigen::VectorXd& theta,// initial mean parameter vector
                   Eigen::MatrixXd& cov,  // initial nxn covariance matrix
                   const int K,
                   const int E,
                   const int N,
                   double epsi,
                   void (*evalFunc)(std::fstream&,
                                    struct policy&, // policy
                                    const int,      // number of episodes
                                    const int       // axis
                                    ));

void concurrent_sampling_population(std::fstream& fs,
                                    std::mutex& m,
                                    Eigen::EigenMultivariateNormal<double>& mvn,
                                    std::priority_queue<struct policy, std::vector<struct policy>, policy_compare>& poque,
                                    const int cnt,
                                    const int K,
                                    const int N,
                                    const int i,
                                    void (*evalFunc)(std::fstream&,
                                                     struct policy& ,
                                                     const int, // Number of episodes
                                                     const int // for data dumping first column
                                                     )
                                    );

void hill_climbing(const std::string prefix,
                   const int trial,
                   const int n,
                   Eigen::VectorXd& theta,// Initial mean policy parameter vector
                   const double tau,      // Exploration parameter
                   const int N,            // Number of episodes to evaluate
                   void (*evalFunc)(std::fstream&,
                                    struct policy&,
                                    const int,
                                    const int)
                   );



void eval_cart_pole_policy(std::fstream& fs,
                           struct policy& po,
                           const int num_episodes,
                           const int axis);

void save_data(std::fstream& fs,
               const double J,
               const int axis);

void eval_grid_multithread(std::fstream& fs,
                           struct policy& po,
                           const int num_episodes,
                           const int axis);



#endif
