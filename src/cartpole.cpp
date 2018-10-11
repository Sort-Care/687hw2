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

const double force[2] = {-FORCE, FORCE};

/*
 * Discretize the continuous state into buckets
 * The original state has: 
 * 1. position :  [-3.0. 3.0]
 * 2. velocity :  (-inf, inf)
 * 3. pole angle: (-M_PI/2, M_PI/2)
 * 4. angular velocity: (-inf, inf)
 */

/*
 * function that take in the current real state of cart pole
 * and map that into a bucket number.
 */
int get_bucket(struct cart_state& cs)
{
        /*
         * THIS FUNCTION ACTUALLY DEFINES HOW WE SEPERATE THE BUCKETS
         * 
         */
    int bucket = 0;
    if (cs.x < -3 ||
        cs.x > 3 ||
        cs.theta < - FAIL_ANGLE ||
        cs.theta > - FAIL_ANGLE) return (-1);// end of episode


        /*
         *|--------------- (-1) ------- 0 -------- 1 -----------------|
         *    BUCKET 0             BUCKET 1             BUCKET 2
         */
    if (cs.x < -1)				bucket = 0;
    else if (cs.x < 1)			bucket = 1;
    else						bucket = 2;

    if (cs.x_dot < -0.6)		;
    else if (cs.x_dot < 0.6)	bucket += 3;
    else						bucket += 6;

    if (cs.theta < - FAIL_ANGLE / 15)		;
    else if(cs.theta < -FAIL_ANGLE / 90)	bucket += 9;
    else if(cs.theta < 0)					bucket += 18;
    else if(cs.theta < FAIL_ANGLE / 90)		bucket += 27;
    else if(cs.theta < FAIL_ANGLE / 15)		bucket += 36;
    else									bucket += 45;

    if (cs.theta_dot < - FAIL_ANGLE / 2)	;
    else if (cs.theta_dot < FAIL_ANGLE / 2) bucket += 54;
    else									bucket += 108;

    return (bucket);
}

void update_state(struct cart_state& cs){
    
}

double get_theta_ddot(struct cart_state& cs, const double& force){
    double numerator = GRAV * std::sin(cs.theta) +
        std::cos(cs.theta) * ((-force - POLE_M * POLE_L *
                               cs.theta_dot * cs.theta_dot * std::sin(cs.theta))
                              / (CART_M + POLE_M));

    double denominator = POLE_L * (4.0/3 - (POLE_L * std::cos(cs.theta)
                                            * std::cos(cs.theta))/(POLE_M + CART_M));

    return numerator / denominator;
    
}


double get_x_ddot(struct cart_state& cs, const double& force){
    return (force + POLE_M * POLE_L * (cs.theta_dot * cs.theta_dot * std::sin(cs.theta)-
                                       get_theta_ddot(cs,force) * std::cos(cs.theta)))
        / (CART_M + POLE_M);
}

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
