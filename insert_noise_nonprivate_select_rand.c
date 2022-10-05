#include <stdio.h>
#include <math.h>
#include <stdlib.h> 
#include <time.h> 
#include "MT.h"

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

void insert_noise_RR_select_rand(double **matrix, double **non_private, int n, double epsilon, double alpha, int seed) {
  // Pointer for dynamically allocation a 2D array.
  int** flip_targets = NULL;
  flip_targets = (int**)malloc(sizeof(int*) * n);
  for(int i = 0; i < n; i++){
    flip_targets[i] = (int*)malloc(sizeof(int) * n);
  }
  
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      if(i==j){
        flip_targets[i][j] = -1;
      }else{
        flip_targets[i][j] = 0;
      }  
    }
  }
  // Compute Probability
  double probability =  exp(epsilon) / (exp(epsilon) + 1);
  // Initialization of selection to be flipped.
  int alpha_node_count = floor(n * alpha);
  // Setting -1 for all non-private users.
  for(int i = 0; i < alpha_node_count; i++) {
    int tmp = non_private[0][i];
    for(int j = 0; j < n; j++) {
      flip_targets[tmp][j] = -1;
    }
  }
  int minus_count = 0;
  minus_count = (alpha_node_count * n) + (n - alpha_node_count);
  int taeget_flip_count = (n * n) - minus_count;
  // Indices storage for flip target.
  int *target_flip_index;
  target_flip_index = (int *)malloc(taeget_flip_count * sizeof(int));
  int counter = 0; 
  int allcounter = 0;
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      if(flip_targets[i][j] == 0){
        target_flip_index[counter] = allcounter;
        counter += 1;
      }  
      allcounter += 1;
    }
  }
 // Fix the seed and generate random numbers for the seed used for flip judgment as many as the number of flip targets in the state.
  init_genrand(seed);
  int *rand_seed;
  rand_seed = (int *)malloc(taeget_flip_count * sizeof(int));
  for(int i = 0; i < taeget_flip_count; i++) {
    rand_seed[i] = genrand_int32();
  }
  //Process flip.
  for(int i = 0; i < taeget_flip_count; i++) {
    int set_seed =  rand_seed[i];
    init_genrand(set_seed);
    double U = genrand_real1();
    if(U > probability){
      int select_item =  target_flip_index[i];
      int x_index = floor(select_item / n);
      int y_index = select_item % n;
      if(matrix[x_index][y_index] == 1){
        matrix[x_index][y_index] = 0;
      }else{
        matrix[x_index][y_index] = 1;
      }
    } 
  }
  free(target_flip_index);
  free(rand_seed);
  free(flip_targets);
}