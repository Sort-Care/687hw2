/*-----------------------------------------------------------------------------+
  | This file is for applying Cross Entropy and hill climbing to:                |
  |     1. Gridworld                                                             |
  |     2. Cart Pole                                                             |
  | The cross entropy and hill climbing functions can be considered as           |
  | interfaces for that they are fed with different evalFuncs to apply           |
  | these two methods to different problem scope.                                |
  | EvalFuncs are problem related and will use any functions by the              |
  | problem environment implementation.                                          |
  |                                                                              |
  | There is a policy structure definition in the hpp file and the               |
  | policy_compare function is for updating the priority queue in order          |
  | to extract top K elite policy parameter vectors.                             |
  |                                                                              |
  |                                                                              |
  |                                                                              |
  |                                                                              |
  |                                                                              |
  |                                                                              |
  | Author: Haoyu Ji                                                             |
  | Date: 10/09/2018                                                             |
  +-----------------------------------------------------------------------------*/

#include <Eigen/Dense>
#include <iostream>
#include "bbo.hpp"
#include "grid.hpp"

/*
 * Cross entropy implementation
 */
void cross_entropy(const std::string prefix,
                   const int trial,       // trial number   
                   const int n,
                   Eigen::VectorXd& theta,// initial mean parameter vector, shape (n,1)
                   Eigen::MatrixXd& cov,  // initial nxn covariance matrix, shape (n,n)
                   const int K,           // Population
                   const int E,           // Elite population
                   const int N,           // Number of episodes for each policy
                   double epsi,           // epsilon: stability param
                   void (*evalFunc)(std::fstream&,
                                    struct policy& ,
                                    const int, // Number of episodes
                                    const int // for data dumping first column
                                    ) // evaluating function
                   ){
    int converged = 40;// 70 for gridworld, 40 for cartpole
    int cnt = 0;        // count the outer loop
//    std::cout << "CE:" << std::endl;
    std::fstream fs;
    std::string filename = prefix + std::to_string(trial);
    fs.open(filename, std::fstream::app);

    Eigen::EigenMultivariateNormal<double> mvn(theta, cov);//initialize the multivariate normal distribution
        //note that the above structure will be changed later
    while (converged != 0){
//        std::cout << "Thread : " << trial << "\tLoop: " << cnt << std::endl;
        converged --;
        std::priority_queue <struct policy, std::vector<struct policy>, policy_compare> poque;
        std::mutex pq_mutex;

            // sample k policy parameter vectors
        Eigen::MatrixXd elite_param(n, E); // structure that holds elite population
        std::vector<std::future<void>> futures;
        REP (i, 0, K-1) {
            futures.push_back(std::async(std::launch::async,
                                         [&]{
                                             return concurrent_sampling_population(std::ref(fs),
                                                                                   pq_mutex,
                                                                                   mvn,
                                                                                   poque,
                                                                                   cnt,
                                                                                   K,
                                                                                   N,
                                                                                   i,
                                                                                   evalFunc);
                                             
                                         }));
                //sample one policy parameter vector
                // struct policy tmp_po;
            
                // tmp_po.param = mvn.sample(1000);
                //     //std::cout << "sampled" << tmp_po.param <<std::endl;
                //     //evaluate the policy
                // evalFunc(tmp_po, N, cnt*K*N + i*N, trial);
                //     //push this policy into priority queue
                // poque.push(tmp_po);
        }
            //sort according to their average discounted returns
            /*
             * May be there's no need for sorting at all!
             * Could use max heap to find top K
             */

            //summing over the first E elite and update the theta
        for(auto& e : futures){
            e.get();
        }
        
        REP (i, 0, E-1){
            elite_param.col(i) = poque.top().param;
                //std::cout<<"Elite reward: " << poque.top().J << std::endl;
                //pop out the top element
            poque.pop();
        }
        
        theta = elite_param.rowwise().mean();
            // early termination
        mvn.setMean(theta);
            //std::cout<< "New Mean: "<< theta << std::endl;
        
            //update the cov according to above results
        Eigen::MatrixXd centered = elite_param.colwise() - theta;
        cov = (Eigen::MatrixXd::Identity(n, n) * epsi + centered * centered.transpose()) / (epsi + E);
        mvn.setCovar(cov);
        cnt ++;
    }
    fs.close();

}

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
                                    ){
    struct policy tmp_po;
    tmp_po.param = mvn.samples(1);
    evalFunc(fs,
             tmp_po,
             N,
             cnt * K * N + i * N);
    std::lock_guard<std::mutex> lock(m);
    poque.push(tmp_po);
}


void hill_climbing(const std::string prefix,
                   const int trial,
                   const int n,
                   Eigen::VectorXd& theta,// Initial mean policy parameter vector
                   const double tau,      // Exploration parameter
                   const int N,           // Number of episodes to evaluate
                   void (*evalFunc)(std::fstream&,
                                    struct policy&,
                                    const int,
                                    const int) // evaluate function
                   ){
    int convergent = 0;
    
    std::fstream fs;
    auto filename = prefix + std::to_string(trial);
    fs.open(filename, std::fstream::app);
    
    policy best_policy = {theta, 0.0};

    evalFunc(fs,
             best_policy,
             N,
             0);
    Eigen::EigenMultivariateNormal<double> mvn(theta, Eigen::MatrixXd::Identity(n,n) * tau);

    while (convergent != 1){
        
        policy tmp_po;
        tmp_po.param = mvn.samples(1);
        evalFunc(fs,
                 tmp_po,
                 N,
                 0);// putting zero there just for now
        
        if (tmp_po.J > best_policy.J){
            best_policy.param = tmp_po.param;
            best_policy.J = tmp_po.J;
                //update the distribution
            mvn.setMean(best_policy.param);
        }
            //test convergence
    }

}

/*
 * Evaluate a single policy for some episodes and return average J
 * Note that in the struct data structure, the policy is represented
 * in the form of policy parameters. Thus before evaluating it, we
 * need to transfer it to another form.
 * This function run evaluation on [policy] [num_episodes] times and store the
 * average discounted return in the policy structure.
 */
// void eval_grid_policy(std::fstream& fs,
//                       struct policy& po,
//                       const int num_episodes,
//                       const int axis,
//                       const int trial){
//     po.J = 0.0;
//         // transfer policy parameter to a policy probability table

//         // loop: do N episodes
//     REP (i, 0, num_episodes-1){
//             //run the grid world policy for one episode and accumulate the reward
//         double reward = run_gridworld_on_policy(po);
//         std::cout << axis + i <<'\t' << reward << std::endl;
//         po.J += reward;
//     }
//     po.J /= num_episodes;
    
// }

void eval_grid_multithread(std::fstream& fs,
                           struct policy& po,
                           const int num_episodes,
                           const int axis){
        //maximum on my ubuntu desktop: 16GM RAM: 125599s
    
    po.J = 0.0;

    std::vector<std::future<double>> futures;
    
    REP(i, 0, num_episodes-1){
        futures.push_back(std::async(std::launch::async,
                                     [&]{ return run_gridworld_on_policy(po);}));
    }
 
        //retrive and print the value stored in the future
    int count = 0;
    for(auto &e : futures) {
        double reward = e.get();
        po.J += reward;
            //std::cout << axis + cnt << '\t' << reward << std::endl;
            // write to file
        save_data(fs,
                  reward,
                  axis + count
                  );
        count ++;
    }
    po.J /= num_episodes;
//    std::cout<< po.J << std::endl;
    
}


/*
 * Evaluate cart pole policy
 */
void eval_cart_pole_policy(std::fstream& fs,
                           struct policy& po,
                           const int num_episodes,
                           const int axis){
    po.J = 0.0;
    std::vector<std::future<double>> futures;

    REP(i, 0, num_episodes -1 ){
//        std::cout << "Episode : " << i << std::endl;
        futures.push_back(std::async(std::launch::async,
                                     [&]{
                                         return run_cartpole_on_policy(po);
                                     }));
    }

    int count = 0;
    for (auto &e : futures){
        double reward = e.get();
        po.J += reward;
        fs << axis << '\t' << reward << '\n';
            //std::cout << "suffix: " << trial << std::endl;
        
        count ++;
    }
    po.J /= num_episodes;
    std::cout <<"Average reward: "<< po.J << std::endl;

}

void save_data(std::fstream& fs,
               const double J,
               const int axis){
        //save axis, J to file
    fs << axis << '\t' << J <<'\n';
}

