#include <iostream>
#include <stdio.h>
#include "support.h"
#include "kernel.cu"

int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    // int M;
    // int K;
    // int N;

    // if(argc == 1) {
    //     M = 1024;
    //     K = 1024;
    //     N = 1024;
    // } else if(argc == 4) {
    //     M = atoi(argv[1]);
    //     K = atoi(argv[2]);
    //     N = atoi(argv[3]);
    // } else {
    //     printf("\n    Invalid input parameters!"
    //        "\n    Usage: ./matmul               # Matrices of size 1024x1024 are used"
    //        "\n");
    //     exit(0);
    // }


    // Retreive the dimension of the matrices
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    // Calculate the size of the matrices
    int matA_size = M * K;
    int matB_size = K * N;
    int matC_size = M * N;

    // Initialize the matrices
    float* A_h = (float*) malloc( sizeof(float)*matA_size );
    for (unsigned int i=0; i < matA_size; i++) { A_h[i] = (rand()%100)/100.00; }
    //  for (unsigned int i=0; i < matA_size; i++) { A_h[i] = 1; }

    float* B_h = (float*) malloc( sizeof(float)*matB_size );
    for (unsigned int i=0; i < matB_size; i++) { B_h[i] = (rand()%100)/100.00; }

    float* C_h = (float*) malloc( sizeof(float)*matC_size );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    // printf("    Vector size = %u\n", n);



    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
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

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(A_d, A_h, sizeof(float)*matA_size, cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

    //INSERT CODE HERE for B
    cuda_ret = cudaMemcpy(B_d, B_h, sizeof(float)*matB_size, cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));



    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    const unsigned int THREADS_PER_BLOCK = 32;
    const unsigned int numBlocks_x = (N - 1)/THREADS_PER_BLOCK + 1;
    const unsigned int numBlocks_y = (M - 1)/THREADS_PER_BLOCK + 1;


    dim3 gridDim(numBlocks_x, numBlocks_y, 1), blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    //INSERT CODE HERE to call kernel
    matMulKernel_tiled<<<gridDim, blockDim>>>(A_d, B_d, C_d, M, K, N);

    
    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));




    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE to copy C
    cuda_ret = cudaMemcpy(C_h, C_d, sizeof(float)*matC_size, cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));



    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify_mat_mul(A_h, B_h, C_h, M, K, N);

    // // Print the result (optional)
    // std::cout << "Matrix C (result of A * B):\n";
    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         std::cout << C_h[i * N + j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE to free device matrices
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);




    return 0;

}

