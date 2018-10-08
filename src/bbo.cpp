#include <Eigen/Dense>
#include "bbo.hpp"


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


/*
 * Cross entropy implementation
 */
void cross_entropy(const int n,
                   Eigen::VectorXd& theta,// initial mean parameter vector, shape (n,1)
                   Eigen::MatrixXd& cov,  // initial nxn covariance matrix, shape (n,n)
                   const int K,           // Population
                   const int E,           // Elite population  
                   const int N,           // Number of episodes for each policy
                   double epsi,           // epsilon: stability param
                   double (*evalFunc)(Eigen::VectorXd& , const int) // evaluating function
                   ){
    int converged = 0;

    MVN mvn(theta, cov);//initialize the multivariate normal distribution
        //note that the above structure will be changed later
    while (converged != 1){
        std::priority_queue <struct policy, std::vector<struct policy>, policy_compare> poque;
        
            // sample k policy parameter vectors
        Eigen::MatrixXd policy_param(n, E); // structure that holds elite population
        REP (i, 0, K-1) {
                //sample one policy parameter vector
            struct policy tmp_po;
            tmp_po.param = mvn.sample(100);
                //evaluate the policy
            eval_grid_policy(tmp_po, N);
                //push this policy into priority queue
            poque.push(tmp_po);
        }
            //sort according to their average discounted returns
            /*
             * May be there's no need for sorting at all!
             * Could use max heap to find top K
             */
        
            //summing over the first E elite and update the theta
        
            //update the cov according to above results
        
            //update converged flag
    }
    
}

void hill_climbing(Eigen::VectorXd& theta,// Initial mean policy parameter vector
                   const double tau,      // Exploration parameter
                   const int N            // Number of episodes to evaluate   
                   ){
    
    
}

/*
 * Evaluate a single policy for some episodes and return average J
 * Note that in the struct data structure, the policy is represented 
 * in the form of policy parameters. Thus before evaluating it, we 
 * need to transfer it to another form.
 * This function run evaluation on [policy] [num_trials] times and store the
 * average discounted return in the policy structure.
 */
void eval_grid_policy(struct policy& po,
                      const int num_trials){
    
}



