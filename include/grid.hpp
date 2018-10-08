#ifndef __GRID_WORLD
#define __GRID_WORLD

#include <stdlib.h>
#include <cmath>
#include <stdio.h>
#include <algorithm>


/*** Functions ****/
void generateInput();

int get_random_action(int total_num_actions);

void print_normal();

void print_for_py();

void value_iteration();

double simulate_random();// run simulation randomly choose actions

double simulate_optimal();// run simulation using optimal policy

void run_simulation_with_strategy(double (*f)());

double estimate_quantity();




void get_array_statistics(const double array[], const int size);

void get_standard_deviation(const double array[], const int size,
                            double *amean,
                            double *adevia);

#endif
