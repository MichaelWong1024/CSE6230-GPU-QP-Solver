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


void verify_mat_mul_column_major(float *A, float *B, float *C, int M, int K, int N) {
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
        // sum += A[row * K + i] * B[i * N + col];
        double a = A[row * K + i];
        double b = B[i * N + col];
        sum += a * b;
      }

      float relativeError = (sum - C[index])/sum;
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


void verify_mat_mul(float *A, float *B, float *C, int M, int K, int N) {
  const float relativeTolerance = 1e-3;

  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      float sum = 0.0f;
      for(int l =0; l < K; l++){
        sum += A[K*i + l] * B[N*l + j];
      }
        int index = N*i + j;
        float relativeError = (sum - C[index]) / sum;
        if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
            printf("TEST FAILED at position (%d, %d)\n\n", i, j);
            printf("sum = %f and C = %f\n", sum, C[index]);
            printf("Relative Error = %f\n", relativeError);
            exit(0);
      }
    }
  }
    printf("TEST PASSED\n\n");
}

void verify_dmat_mul(double *A, double *B, double *C, int M, int K, int N) {
  const double relativeTolerance = 1e-3;

  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      double sum = 0.0f;
      for(int l =0; l < K; l++){
        sum += A[K*i + l] * B[N*l + j];
      }
        int index = N*i + j;
        double relativeError = (sum - C[index]) / sum;
        if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
            printf("TEST FAILED at position (%d, %d)\n\n", i, j);
            printf("sum = %f and C = %f\n", sum, C[index]);
            printf("Relative Error = %f\n", relativeError);
            exit(0);
      }
    }
  }
    printf("TEST PASSED\n\n");
}


void cpu_mat_mul(float *A, float *B, float *C, int M, int K, int N) {
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      float sum = 0.0f;
      for(int l =0; l < K; l++){
        sum += A[K*i + l] * B[N*l + j];
      }
      C[N*i + j] = sum;
    }
  }
}

void cpu_dmat_mul(double *A, double *B, double *C, int M, int K, int N) {
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      double sum = 0.0f;
      for(int l =0; l < K; l++){
        sum += A[K*i + l] * B[N*l + j];
      }
      C[N*i + j] = sum;
    }
  }
}

void cpu_dmv_mul(double *A, double *x, double *y, int M, int N){
  for(int i =0; i < M; i++){
    double sum = 0.0;
    for(int j = 0; j < N; j++){
      sum += A[ M * i + j] + x[j];
    }
    y[i] = sum;
  }
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

void print_matrix(double* A, int M, int N){

      for(int i = 0; i < min(M,6); i++){
        for(int j =0; j < min(N, 6); j++){
            printf("%5.2f", A[j+i*N]);
        }
        printf("\n");
    }

}

void print_vector(double* x, int M){
      for(int i = 0; i < min(M,6); i++){
        printf("%5.2f", x[i]);
    }
    printf("\n");
}
