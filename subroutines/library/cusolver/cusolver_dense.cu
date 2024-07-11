#include <stdio.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#define CHECK_CUSOLVER(call, msg) { \
    cusolverStatus_t err = call; \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "%s\n", msg); \
        fprintf(stderr, "Error code: %d\n", err); \
        exit(1); \
    } \
}

int main() {
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH), "Failed to create cuSolver handle.");
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));
    CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, stream), "Failed to set stream for cuSolver.");

    // example code for solving Ax=b
    const int n = 3; // size of matrix
    const int lda = n;
    float A[lda*n] = {1, 2, 3, 4, 5, 6, 7, 8, 10};
    float b[n] = {1, 2, 3};
    int info = 0; // error message on device
    float *d_A = NULL, *d_b = NULL;
    int *d_info = NULL; // info on device
    int bufferSize = 0;
    float *buffer = NULL;
    int h_info = 0; // error message on host

    // malloc memory on device
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(float) * lda * n));
    CHECK_CUDA(cudaMalloc((void**)&d_b, sizeof(float) * n));
    CHECK_CUDA(cudaMalloc((void**)&d_info, sizeof(int)));

    // copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, A, sizeof(float) * lda * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b, sizeof(float) * n, cudaMemcpyHostToDevice));

    // check buffer size
    CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(cusolverH, n, n, d_A, lda, &bufferSize), "Failed to query buffer size.");
    CHECK_CUDA(cudaMalloc(&buffer, sizeof(float) * bufferSize));

    // solve linear system
    CHECK_CUSOLVER(cusolverDnSgetrf(cusolverH, n, n, d_A, lda, buffer, NULL, d_info), "Failed to compute LU decomposition.");
    CHECK_CUSOLVER(cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, n, 1, d_A, lda, NULL, d_b, n, d_info), "Failed to solve linear system.");

    // check error
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr, "Error: LU decomposition failed\n");
        return 1;
    }

    // copy data from device to host
    CHECK_CUDA(cudaMemcpy(b, d_b, sizeof(float) * n, cudaMemcpyDeviceToHost));

    printf("Solution: \n");
    for (int i = 0; i < n; i++) {
        printf("%f\n", b[i]);
    }

    // free memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_info));
    CHECK_CUDA(cudaFree(buffer));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH), "Failed to destroy cuSolver handle.");
    CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}
