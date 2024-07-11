#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void verify_vec(float *A, float *B, float *C, int n) {

  const float relativeTolerance = 1e-6;

  for(int i = 0; i < n; i++) {
    float sum = A[i] + B[i];
    float relativeError = (sum - C[i])/sum;
    if (relativeError > relativeTolerance
      || relativeError < -relativeTolerance) {
      printf("TEST FAILED\n\n");
      exit(0);
    }
  }
  printf("TEST PASSED\n\n");

}


void verify_mat_mul(float *A, float *B, float *C, int M, int K, int N) {

  
    // printf("\n");
    // for(int i = 0; i < M*; i++){
    //   printf("A[%d] = %f \n", i, A[i]);
    // }

  const float relativeTolerance = 1e-3;

  for(int row = 0; row < M; row++){
    for(int col =0; col < N; col++){

      int index = row * N + col;
      float sum = 0.0;
      for(int i = 0; i < K; i++){
        sum += A[row * K + i] * B[i * N + col];
      }

      float relativeError = (sum - C[index])/(double)sum;
      if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
                printf("TEST FAILED at position (%d, %d)\n\n", row, col);
                printf("sum = %f and C = %f\n", sum, C[index]);
                printf("Relative Error = %f\n", relativeError);
                exit(0);
            }
    }
  }
  printf("TEST PASSED\n\n");

}



void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

