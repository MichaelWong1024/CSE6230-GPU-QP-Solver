#include "cublas_v2.h"

#define TILE_WIDTH 32


__global__ void matMulKernel_naive(float* A, float* B, float* C, int M, int K, int N){
    /* Matric multiplication kernel. A simple matrix multitplication
        is implemeneted using CUDA on the GPU
    */

     /*
     A is dimension M x K
     B is dimension K x N
     C is dimension M x N
     */   

    // Calculate global thread index based on the block and thread indices ----
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x ;
    

    if((row < M) && (col < N)){ // Veriy first that we are withing the bound of the matrix
        float tmpC_val = 0; // Temporary value to hold the result of the dot product
        for(int i =0; i < K; i++){
            tmpC_val += A[row * K + i] * B[i * N + col ];
        }
        // store result
        C[row * N + col] = tmpC_val;
    }

}

__global__ void matMulKernel_tiled(float* A, float* B, float* C, int M, int K, int N){
    // Calculate global thread index based on the block and thread indices ----
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Shared memory for tiles
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    float sum = 0.0;

    // Loop over tiles
    for (int t = 0; t < ceil(K/ (float)TILE_WIDTH); ++t) {
        
        int tile_col = t * TILE_WIDTH + tx;
        if (row < M && tile_col < K) {
            tile_A[ty][tx] = A[row * K + tile_col];
        } else {
            tile_A[ty][tx] = 0.0;
        }

        int tile_row = t * TILE_WIDTH + ty;
        if (tile_row < K && col < N) {
            tile_B[ty][tx] = B[tile_row * N + col];
        } else {
            tile_B[ty][tx] = 0.0;
        }

        __syncthreads();

        // Compute partial sum for this tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }

        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
        // int c_idx = N * TILE_WIDTH * by + TILE_WIDTH * by;
        // C[c_idx + N * ty + tx] = sum;
    }
}



void gpu_mm_mult(float *A_h, float *B_h, float *C_h, int M, int K, int N){

    cudaError_t cuda_ret;
    cublasStatus_t stat;     // CUBLAS functions status
    cublasHandle_t handle;      // CUBLAS context
    
    const int matA_size = M * K;
    const int matB_size = K * N;
    const int matC_size = M * N;

    // Allocate device variables ----------------------------------------------
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

    // Copy host variables to device ------------------------------------------
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

    // Perform matrix multiplication -------------------------------------------

    stat = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, N, M, K, &al, B_d, N, A_d, K, &bet, C_d, N);
    
    cudaDeviceSynchronize();
    if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to launch kernel");

    // Copy device variable from host-------------------------------------------
    stat = cublasGetMatrix(M, N, sizeof(*C_h), C_d, N, C_h, N); // copy C_d -> C_h
    if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to copy memory to host");
    
    cudaDeviceSynchronize();

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    // Destroy cuBLAS handle
    cublasDestroy(handle);
}


void gpu_dmm_mult(double *A_h, double *B_h, double *C_h, int M, int K, int N){

    cudaError_t cuda_ret;
    cublasStatus_t stat;     // CUBLAS functions status
    cublasHandle_t handle;      // CUBLAS context
    
    const int matA_size = M * K;
    const int matB_size = K * N;
    const int matC_size = M * N;

    // Allocate device variables ----------------------------------------------
    double* A_d;
    cuda_ret = cudaMalloc((void**) &A_d, sizeof(double)*matA_size);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    //INSERT CODE HERE for B and C
    double* B_d;
    cuda_ret = cudaMalloc((void**) &B_d, sizeof(double)*matB_size);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    double* C_d;
    cuda_ret = cudaMalloc((void**) &C_d, sizeof(double)*matC_size);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();

    // Copy host variables to device ------------------------------------------
    stat = cublasCreate(&handle); // initialize CUBLAS context

    stat = cublasSetMatrix(M, K, sizeof(*A_h), A_h, M, A_d, M);
	if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to copy memory to device");

    stat = cublasSetMatrix(K, N, sizeof(*B_h), B_h, K, B_d, K);
    if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to copy memory to device");

    stat = cublasSetMatrix(M, N, sizeof(*C_h), C_h, M, C_d, M);
    if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to copy memory to device");

    double al    = 1.0;
    double bet   = 1.0;

    cudaDeviceSynchronize();

    // Perform matrix multiplication -------------------------------------------

    stat = cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, N, M, K, &al, B_d, N, A_d, K, &bet, C_d, N);
    
    cudaDeviceSynchronize();
    if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to launch kernel");

    // Copy device variable from host-------------------------------------------
    stat = cublasGetMatrix(M, N, sizeof(*C_h), C_d, N, C_h, N); // copy C_d -> C_h
    if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to copy memory to host");
    
    cudaDeviceSynchronize();

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    // Destroy cuBLAS handle
    cublasDestroy(handle);
}




// void gpu_dmv_mult(double *A_h, double *x_h, double *y_h, int m, int n){
//     cudaError_t cuda_ret;
//     cublasStatus_t stat;     // CUBLAS functions status
//     cublasHandle_t handle;      // CUBLAS context
    
//     const int matA_size = m * n;
//     const int vecx_size = n;
//     const int vecy_size = m;

//     // Allocate device variables ----------------------------------------------
//     double* A_d;
//     cuda_ret = cudaMalloc((void**) &A_d, sizeof(double)*matA_size);
// 	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

//     //INSERT CODE HERE for B and C
//     double* x_d;
//     cuda_ret = cudaMalloc((void**) &x_d, sizeof(double)*vecx_size);
// 	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

//     double* y_d;
//     cuda_ret = cudaMalloc((void**) &y_d, sizeof(double)*vecy_size);
// 	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

//     cudaDeviceSynchronize();

//     // Copy host variables to device ------------------------------------------
//     stat = cublasCreate(&handle); // initialize CUBLAS context

//     stat = cublasSetMatrix(M, K, sizeof(*A_h), A_h, m, A_d, n);
// 	if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to copy memory to device");

//     stat = cublasSetVector (n, sizeof (*x_h),x_h ,1, x_d ,1);
//     if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to copy memory to device");

//     stat = cublasSetVector (m, sizeof (*y_h),y_h ,1, y_d ,1);
//     if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to copy memory to device");

//     double al    = 1.0;
//     double bet   = 1.0;

//     cudaDeviceSynchronize();

//     // Perform matrix-vector multiplication -------------------------------------------

//     stat=cublasDgemv(handle,CUBLAS_OP_N, n, m, &al, A_d, n, x_d, 1, &bet, y_d, 1);
 
//     cudaDeviceSynchronize();
//     if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to launch kernel");

//     // Copy device variable from host-------------------------------------------
//     stat = cublasGetVector(m, sizeof (*y_h), y_d ,1, y_h ,1); // copy d_y ->y
//     if(stat != CUBLAS_STATUS_SUCCESS) FATAL("Unable to copy memory to host");
    
//     cudaDeviceSynchronize();

//     cudaFree(A_d);
//     cudaFree(x_d);
//     cudaFree(y_d);
//     // Destroy cuBLAS handle
//     cublasDestroy(handle);
// }