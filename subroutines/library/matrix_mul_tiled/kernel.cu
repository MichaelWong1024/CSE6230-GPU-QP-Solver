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
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        
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
    }
}