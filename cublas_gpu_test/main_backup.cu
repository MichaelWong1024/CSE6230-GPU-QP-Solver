#include <iostream>
#include <stdio.h>
#include "support.h"
#include "kernel.cu"
#include "cublas_v2.h"

int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;

    cublasStatus_t stat;     // CUBLAS functions status
    cublasHandle_t handle;      // CUBLAS context

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
    float* A_h = (float*) malloc( sizeof(float)*matA_size );
    for (unsigned int i=0; i < matA_size; i++) { A_h[i] = (rand()%100)/100.00; }
        // for (unsigned int i=0; i < matA_size; i++) { A_h[i] = 1.0f; }
    
    // ind=11;
    float* B_h = (float*) malloc( sizeof(float)*matB_size );
    for (unsigned int i=0; i < matB_size; i++) { B_h[i] = (rand()%100)/100.00; }
        // for (unsigned int i=0; i < matB_size; i++) { B_h[i] = 2.0f; }
    
    // ind=11;
    float* C_h = (float*) malloc( sizeof(float)*matC_size );
    // for (unsigned int i=0; i < matC_size; i++) { C_h[i] = 0.0f; }
     for (unsigned int i=0; i < matC_size; i++) { C_h[i] = 0.0f; }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    // printf("    Vector size = %u\n", n);
    printf("M = %d, K = %d, and N = %d\n", M, K, N);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables... "); fflush(stdout);
    startTime(&timer);

    float* A_d;
    cuda_ret = cudaMalloc((void**) &A_d, sizeof(float)*matA_size);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    //INSERT CODE HERE for B and C
    float* B_d;
    cuda_ret = cudaMalloc((void**) &B_d, sizeof(float)*matB_size);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    float* C_d;
    cuda_ret = cudaMalloc((void**) &C_d, sizeof(float)*matC_size);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));





    // Copy host variables to device ------------------------------------------
    printf("Copying data from host to device... "); fflush(stdout);
    startTime(&timer);

    stat = cublasCreate(&handle); // initialize CUBLAS context

    stat = cublasSetMatrix(M, K, sizeof(*A_h), A_h, M, A_d, M);
	if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to copy memory to device");

    stat = cublasSetMatrix(K, N, sizeof(*B_h), B_h, K, B_d, K);
    if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to copy memory to device");

    stat = cublasSetMatrix(M, N, sizeof(*C_h), C_h, M, C_d, M);
    if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to copy memory to device");

    float al    = 1.0f;
    float bet   = 1.0f;

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    // Perform matrix multiplication -------------------------------------------
    // C_d = al * A_d * B_d + bet * C_d
    printf("Launching kernel... "); fflush(stdout);
    startTime(&timer);
    // stat = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, M, N, K, &al, A_d, M, B_d, K, &bet, C_d, M);
    // cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&al,d_b,n,d_a,k,&bet,d_c,n)

    // To perform operation with row_major we send A' and B'
    // Effectively we are doing C' = B' @ A' and cublas by saving in column major will give us
    // the correct output C in row major.
    //reference: https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
    stat = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, N, M, K, &al, B_d, N, A_d, K, &bet, C_d, N);
    
    cudaDeviceSynchronize();
    if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    // Copy device variable from host-------------------------------------------
    printf("Copying data from device to host... "); fflush(stdout);
    startTime(&timer);
    stat = cublasGetMatrix(M, N, sizeof(*C_h), C_d, N, C_h, N); // copy C_d -> C_h
    if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to copy memory to host");
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    
    // Verify correctness -----------------------------------------------------
    printf("Verifying results... \n"); fflush(stdout);
  
    // Print the values for verification
    // printf("Top left corner of matrix A_h: \n");fflush(stdout);
    // print_matrix(A_h, M, K);

    // printf("Top left corner of matrix B_h: \n");fflush(stdout);
    // print_matrix(B_h, K, N);

    // printf("Top left corner of matrix C_h: \n");fflush(stdout);
    // print_matrix(C_h, M, N);

    // float* C_real = (float*) malloc( sizeof(float)*matC_size );
    // cpu_mat_mul(A_h, B_h, C_real, M, K, N);
    // printf("Top left corner of matrix C_real: \n");fflush(stdout);
    // print_matrix(C_real, M, N);

    // printf("Top left corner of matrix C_d: \n");fflush(stdout);
    // print_matrix(C_d, M, N);
    
    verify_mat_mul(A_h, B_h, C_h, M, K, N);



    // Free memory ------------------------------------------------------------
    cublasDestroy(handle); // destroy CUBLAS context
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A_h);
    free(B_h);
    free(C_h);


    return 0;

}