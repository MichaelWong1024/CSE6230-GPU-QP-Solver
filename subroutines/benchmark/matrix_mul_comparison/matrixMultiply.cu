#include <stdio.h>
#include <cuda_runtime.h>

// Kernel for data-parallel matrix multiplication
__global__ void MatrixMultiplyDataParallel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel for problem-parallel matrix multiplication
__global__ void MatrixMultiplyProblemParallel(float *A, float *B, float *C, int N, int numProblems) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int problemIndex = blockIdx.z;
    if (row < N && col < N && problemIndex < numProblems) {
        int baseIndex = problemIndex * N * N;
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[baseIndex + row * N + k] * B[baseIndex + k * N + col];
        }
        C[baseIndex + row * N + col] = sum;
    }
}

void runMultiplicationExperiments() {
    int sizes[] = {64, 128, 256, 512, 1024};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    int numProblems = 10; // Number of matrices for problem-parallel approach

    for (int i = 0; i < numSizes; i++) {
        int N = sizes[i];
        size_t size = N * N * sizeof(float);
        size_t problemSize = size * numProblems;
        float *A, *B, *C; // Host matrices
        float *d_A, *d_B, *d_C; // Device matrices

        // Allocate host memory and initialize matrices
        A = (float *)malloc(problemSize);
        B = (float *)malloc(problemSize);
        C = (float *)malloc(problemSize);
        // Initialize A and B matrices with some values
        for (int j = 0; j < N * N * numProblems; j++) {
            A[j] = 1.0f; // Example initialization
            B[j] = 2.0f; // Example initialization
        }

        // Allocate device memory
        cudaMalloc(&d_A, problemSize);
        cudaMalloc(&d_B, problemSize);
        cudaMalloc(&d_C, problemSize);

        // Copy matrices from the host to the device
        cudaMemcpy(d_A, A, problemSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, problemSize, cudaMemcpyHostToDevice);

        // Define the execution configuration
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliseconds = 0;

        // Data-parallel execution
        cudaEventRecord(start);
        MatrixMultiplyDataParallel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Data-parallel execution for N=%d: %f ms\n", N, milliseconds);

        // Problem-parallel execution (adjust grid configuration for the number of problems)
        blocksPerGrid.z = numProblems;
        cudaEventRecord(start);
        MatrixMultiplyProblemParallel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, numProblems);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Problem-parallel execution for N=%d with %d problems: %f ms\n", N, numProblems, milliseconds);

        // Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(A);
        free(B);
        free(C);
    }
}

int main() {
    runMultiplicationExperiments();
    return 0;
}
