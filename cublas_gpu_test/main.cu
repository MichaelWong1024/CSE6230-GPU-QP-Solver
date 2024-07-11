#include <iostream>
#include <stdio.h>
#include "support.h"
#include "kernel.cu"
#include "cublas_v2.h"

int main(int argc, char**argv) {

    Timer timer;
    // cudaError_t cuda_ret;

    // cublasStatus_t stat;     // CUBLAS functions status
    // cublasHandle_t handle;      // CUBLAS context

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem...  "); fflush(stdout);
    startTime(&timer);

    // Retreive the dimension of the matrices
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    // Calculate the size of the matrices
    int matA_size = M * K;
    int matB_size = K * N;
    int matC_size = M * N;

    // Initialize the matrices
    // int ind=11;
    double* A_h = (double*) malloc( sizeof(double)*matA_size );
    for (unsigned int i=0; i < matA_size; i++) { A_h[i] = (rand()%100)/100.00; }
        // for (unsigned int i=0; i < matA_size; i++) { A_h[i] = 1.0f; }
    
    // ind=11;
    double* B_h = (double*) malloc( sizeof(double)*matB_size );
    for (unsigned int i=0; i < matB_size; i++) { B_h[i] = (rand()%100)/100.00; }
        // for (unsigned int i=0; i < matB_size; i++) { B_h[i] = 2.0f; }
    
    // ind=11;
    double* C_h = (double*) malloc( sizeof(double)*matC_size );
    // for (unsigned int i=0; i < matC_size; i++) { C_h[i] = 0.0f; }
     for (unsigned int i=0; i < matC_size; i++) { C_h[i] = 0.0; }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    // printf("    Vector size = %u\n", n);
    printf("M = %d, K = %d, and N = %d\n", M, K, N);

    
    // CPU Matrix-Matrix multiplication ----------------------
    printf("CPU multiplication... "); fflush(stdout);
    double* C_real = (double*) malloc( sizeof(double)*matC_size );
    cpu_dmat_mul(A_h, B_h, C_real, M, K, N); // first call to have reliable result
    startTime(&timer);
    cpu_dmat_mul(A_h, B_h, C_real, M, K, N);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // GPU Matrix-Matrix multiplication ----------------------
    printf("GPU multiplication... "); fflush(stdout);
    startTime(&timer);
    gpu_dmm_mult(A_h, B_h, C_h, M, K, N);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------
    // printf("Verifying results... \n"); fflush(stdout);

    // //Print the values for verification
    // printf("Top left corner of matrix A_h: \n");fflush(stdout);
    // print_matrix(A_h, M, K);

    // printf("Top left corner of matrix B_h: \n");fflush(stdout);
    // print_matrix(B_h, K, N);

    // printf("Top left corner of matrix C_h: \n");fflush(stdout);
    // print_matrix(C_h, M, N);

    printf("Verifying results..."); fflush(stdout);
    verify_dmat_mul(A_h, B_h, C_h, M, K, N);

    free(A_h);
    free(B_h);
    free(C_h);


    return 0;

}