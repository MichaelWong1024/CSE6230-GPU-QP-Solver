#include <stdio.h>
#include <cuda_runtime.h>

// Kernel for data-parallel matrix addition
__global__ void MatrixAddDataParallel(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < N && idy < N) {
        int index = idy * N + idx;
        C[index] = A[index] + B[index];
    }
}

// Kernel for problem-parallel matrix addition
__global__ void MatrixAddProblemParallel(float *A, float *B, float *C, int N, int numProblems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int problemIndex = blockIdx.z;

    if (idx < N && idy < N && problemIndex < numProblems) {
        int baseIndex = problemIndex * N * N;
        int index = baseIndex + idy * N + idx;
        C[index] = A[index] + B[index];
    }
}

void runAdditionExperiments() {
    int sizes[] = {64, 128, 256, 512, 1024};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    int numProblems = 10;

    for (int i = 0; i < numSizes; i++) {
        int N = sizes[i];
        size_t size = N * N * sizeof(float);
        size_t problemSize = size * numProblems;
        float *A, *B, *C;
        float *d_A, *d_B, *d_C;

        // Allocate and initialize host memory
        A = (float *)malloc(problemSize);
        B = (float *)malloc(problemSize);
        C = (float *)malloc(problemSize);
        for (int j = 0; j < N * N * numProblems; j++) {
            A[j] = 1.0f; // Example initialization
            B[j] = 2.0f;
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
        MatrixAddDataParallel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Data-parallel addition for N=%d: %f ms\n", N, milliseconds);

        // Problem-parallel execution
        blocksPerGrid.z = numProblems;
        cudaEventRecord(start);
        MatrixAddProblemParallel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, numProblems);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Problem-parallel addition for N=%d with %d problems: %f ms\n", N, numProblems, milliseconds);

        // Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(A);
        free(B);
        free(C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

int main() {
    runAdditionExperiments();
    return 0;
}
