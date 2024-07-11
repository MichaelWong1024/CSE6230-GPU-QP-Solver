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