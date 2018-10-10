#ifndef __CART_POLE
#define __CART_POLE
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <algorithm>
#include <Eigen/Dense>

#include "bbo.hpp"

struct cart_state {
    double x;        // position
    double x_dot;    // cart velocity
    double theta;    // pole angle
    double theta_dot;// pole angular velocity
};


/*
 * function that take in the current real state of cart pole
 * and map that into a bucket number.
 */
int get_bucket(struct cart_state& cs);


Eigen::MatrixXd cart_softmax(struct policy& po,
                             const int rows, //NUM_ACTION
                             const int cols);

#endif
