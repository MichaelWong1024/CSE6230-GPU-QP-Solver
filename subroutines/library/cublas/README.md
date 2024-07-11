### matrix_multiplication.cu
General matrix multiplication using `cublas` API

Compilation:
```
module load cuda/11.7.0-7sdye3
module load gcc
module load cmake

nvcc matrix_multiplication.cu -o matrix_multiplication -lcublas -lcudart
```
