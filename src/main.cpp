/*
 * Grid World: Extending the source code for hw1
 * Author: Haoyu Ji
 * Usage: 
 * > g++ -I /usr/local/include/eigen3/ gridworld.cpp 
 * > ./a.out > [out_filename]
 * Output File Structure:
 * First Line: number of states |S|
 * Next |S| lines: rewards for entering each state
 * Next |S|x|A| lines: Transition table
 * Within Transition table:
 *       {state_i transit with action |A|} * |S|
 *
 * TODO: 
 * 1. Adding absorbing state -- DONE
 * 2. Simulate the running of the gridworld -- DUE today
 * 3. Value Iteration
 */
#include <algorithm>
#include <Eigen/Dense>
#include <iostream>


#include "bbo.hpp"
#include "multi_normal.hpp"
#include "random_sampling.hpp"
#include "grid.hpp"
#include "cartpole.hpp"

//using namespace Eigen;




int main(int argc, char *argv[]){
    generateInput();
//    run_simulation_with_strategy(simulate_random);

        //options: 1. Run Random, 2. Run optimal, 3, estimate quantity
    



        // Define the covariance matrix and the mean
        // Eigen::MatrixXd sigma(2, 2);
        // sigma << 10.0, 7.0,
        //     7.0, 5;
        // Eigen::VectorXd mean(2);
        // mean << 2, 2;
        // MVN mvn(mean, sigma);

        //     // Sample a number of points
        // const unsigned int points = 1000;
        // Eigen::MatrixXd x(2, points);
        // Eigen::VectorXd vector(2);
        // for (unsigned i = 0; i < points; i++)
        // {
        //     vector = mvn.sample(200);
        //     x(0, i) = vector(0);
        //     x(1, i) = vector(1);
        // }

        // Eigen::MatrixXd test(3,4);
        // test << 0,1,2,3,
        //     2,3,4,5,
        //     3,4,5,6;

    
    

        // Eigen::VectorXd res(3);

        // res = test.col(1);
        // std::cout << res << std::endl;

        // Eigen::VectorXd r2(8);

        // r2 << 1,2,3,3,5,6,7,8;
        // struct policy po = {r2, 0.0};
        // Eigen::MatrixXd res = grid_softmax(po, 4, 2);
        // std::cout<< res << std::endl;



    

    int param_size = NUM_ACTION * STATE_NUM;
    
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(param_size);
    Eigen::MatrixXd cov = Eigen::MatrixXd::Constant(param_size, param_size,
                                                    0);
        /*
         * Seeing improvments to near optimal with:
         * K = 20, E = 2, N = 10, epsi = 0.1
         */
    int K = 20;
    int E = 1;
    int N = 10;
    double epsi = 0.1;
    cross_entropy(param_size,
                  theta,
                  cov,
                  K,
                  E,
                  N,
                  epsi,
                  eval_grid_multithread);

        //for grid world with above params, seems like it can reach near optimal
        // within 150 loop over population improvement
    
    
    
    

        //     // Calculate the mean and convariance of the produces sampled points
        // Eigen::VectorXd approx_mean(2);
        // Eigen::MatrixXd approx_sigma(2, 2);
        // approx_mean.setZero();
        // approx_sigma.setZero();

        // for (unsigned int i = 0; i < points; i++)
        // {
        //     approx_mean  = approx_mean  + x.col(i);
        //     approx_sigma = approx_sigma + x.col(i) * x.col(i).transpose();
        // }

        // approx_mean  = approx_mean  / static_cast<double>(points);
        // approx_sigma = approx_sigma / static_cast<double>(points);
        // approx_sigma = approx_sigma - approx_mean * approx_mean.transpose();

        //     // Check if the statistics of the sampled points are close to the statistics
        //     // of the given distribution.
        // printf("%d\t", approx_mean.isApprox(mean, 5e-1));
        //     //EXPECT_TRUE(approx_sigma.isApprox(sigma, 5e-1));






    
    return 0;
    
}



