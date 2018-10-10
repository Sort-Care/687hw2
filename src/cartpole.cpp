#include "cartpole.hpp"


/*
 * Environment Set up
 */
const double FAIL_ANGLE = M_PI/2; // fail angle, if exceeds (-M_PI/2, M_PI/2)
const double FORCE = 10.0;        // Motor force, Newton
const double GRAV = 9.8;          // Gravitational constant
const double CART_M = 1.0;        // Cart mass
const double POLE_M = 0.1;        // Pole mass
const double POLE_L = 0.5;        // Pole half length
const double INTERVAL = 0.02;     // time step 
const double MAXTIME = 20.2;      // Max time before end of episode

/*
 * Discretize the continuous state into buckets
 * The original state has: 
 * 1. position :  [-3.0. 3.0]
 * 2. velocity :  (-inf, inf)
 * 3. pole angle: (-M_PI/2, M_PI/2)
 * 4. angular velocity: (-inf, inf)
 */



/*
 * Transfer theta to pi: R^n --> [0,1]^n
 */
Eigen::MatrixXd cart_softmax(struct policy& po,
                             const int rows, //NUM_ACTION
                             const int cols){//STATE_NUM
        //first reshape
    Eigen::Map<Eigen::MatrixXd> reshaped(po.param.data(), rows, cols);
        // then apply softmax
    Eigen::MatrixXd soft = reshaped.array().exp();
        // then normalize
    Eigen::RowVectorXd col_mean = soft.colwise().sum();
        //replicate before division
    Eigen::MatrixXd repeat = col_mean.colwise().replicate(rows);
    return soft.array() / repeat.array();
}
