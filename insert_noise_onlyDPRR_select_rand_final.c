#include <stdio.h>
#include <math.h>
#include <stdlib.h> 
#include <time.h> 
#include <sys/time.h>
#include "MT.h"

double make_lhaplus(double phi, double meu) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME,&ts);
  long seed = ts.tv_nsec + ts.tv_sec;
  init_genrand((unsigned int)seed);
  double U = genrand_real3();
  double x = 0.0;
  if( U < 0.5 ) {
    x = phi * log( 2*U ) + meu;
  } else {
    x = -(phi * log( 2*(1-U) ) + meu);
  }
  return x;
}

double make_lhaplus_select_seed(double phi, double meu, int seed) {
  init_genrand(seed);
  double U = genrand_real3();
  double x = 0.0;
  if( U < 0.5 ) {
    x = phi * log( 2*U ) + meu;
  } else {
    x = -(phi * log( 2*(1-U) ) + meu);
  }
  return x;
}

void shuffle(int array[], int size, int seed) {
  srand(seed);
  for(int i = 0; i < size; i++) {
      int j = rand()%size;
      int t = array[i];
      array[i] = array[j];
      array[j] = t;
  }
}

void pointer_shuffle(int *array, int size, int seed) {
  srand(seed);
  for(int i = 0; i < size; i++) {
      int j = rand()%size;
      int t = array[i];
      array[i] = array[j];
      array[j] = t;
  }
}

void pointer_shuffle_noseed(int *array, int size) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME,&ts);
  long seed = ts.tv_nsec + ts.tv_sec;
  srand((unsigned int)seed);
  for(int i = 0; i < size; i++) {
      int j = rand()%size;
      int t = array[i];
      array[i] = array[j];
      array[j] = t;
  }
}

void insert_noise_DPRR_select_rand(double **matrix, double **private, double **degrees, int n, double epsilon, double alpha, int private_count, int seed) {
  // Initialization of selection to be flipped
  int alpha_node_count = floor(n * alpha);
  // Setting epsilon
  double epsilon_one = epsilon / 10;
  double epsilon_two = (epsilon * 9) / 10;
  // Compute Probability
  double probability =  exp(epsilon_two) / (exp(epsilon_two) + 1);
  double laplace_scale = 1 / epsilon_one;
  // Calculate seed for random number generation for flip judgment for each node.
  init_genrand(seed);
  int *rand_seed_all_node;
  rand_seed_all_node = (int *)malloc(n * sizeof(int));
  for(int i = 0; i < n; i++) {
    rand_seed_all_node[i] = genrand_int32();
  }
  //
  // Start DPRR
  //
  for(int i = 0; i < private_count; i++) {
    double pre_t = private[0][i];
    int t = (int)pre_t;
    int target_degree = degrees[0][t];

    double random_lhaplace = make_lhaplus_select_seed(laplace_scale,0.0,rand_seed_all_node[i]);
    // Compute k_i
    double k_i = target_degree + random_lhaplace;
    if (k_i < 0){
      k_i = 0;
    }
    // Compute q_i
    double q_i = k_i / ((k_i * (2 * probability - 1)) + ((n - 1) * (1 - probability)));
    if (q_i > 1){
      q_i = 1;
    }
    // Calculation of the number of rows with added noise.
    int sum_adjacency = 0;
    for(int j = 0; j < n; j++) {
      if(matrix[t][j] == 1){
        sum_adjacency += 1;
      }
    }
    // get index and count
    int *one_index;
    one_index = (int *)malloc(sum_adjacency * sizeof(int));
    int one_counter = 0;
    for(int j = 0; j < n; j++) {
      if(matrix[t][j] == 1){
        one_index[one_counter] = j;
        one_counter += 1; 
      }
    }
    // Generate random numbers for the number of flip targets with the seed fixed.
    init_genrand(rand_seed_all_node[i]);
    int *rand_seed;
    rand_seed = (int *)malloc(sum_adjacency * sizeof(int));
    for(int j = 0; j < sum_adjacency; j++) {
      rand_seed[j] = genrand_int32();
    }
    for(int j = 0; j < sum_adjacency; j++) {
      int set_seed =  rand_seed[j];
      init_genrand(set_seed);
      double U = genrand_real1();
      if(U > q_i){
        int tmp = one_index[j];
        matrix[t][tmp] = 0;
      }
    }
    free(rand_seed);
    free(one_index);
  }
  free(rand_seed_all_node);
}