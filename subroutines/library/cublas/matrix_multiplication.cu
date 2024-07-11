#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call, msg) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "%s\n", msg); \
        fprintf(stderr, "Error code: %d\n", status); \
        exit(1); \
    } \
}

int main() {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasH), "Failed to create cuBLAS handle.");
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream), "Failed to set stream for cuBLAS.");

    // Example cuBLAS operation: matrix-matrix multiplication (C = A * B)
    const int m = 3, k = 3, n = 3;
    float alpha = 1.0f;
    float beta = 0.0f;
    float A[m*k] = {1, 4, 7, 2, 5, 8, 3, 6, 9};
    float B[k*n] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float C[m*n];
    float *d_A, *d_B, *d_C;

    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(float) * m * k));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeof(float) * k * n));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeof(float) * m * n));
    CHECK_CUDA(cudaMemcpy(d_A, A, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, sizeof(float) * k * n, cudaMemcpyHostToDevice));

    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m), "Failed to perform matrix-matrix multiplication.");

    CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

    printf("Result of matrix multiplication (C = A * B):\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", C[i * n + j]);
        }
        printf("\n");
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(cublasH), "Failed to destroy cuBLAS handle.");
    CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}
