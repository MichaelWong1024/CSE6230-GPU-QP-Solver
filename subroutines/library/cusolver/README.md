### cusolver_dense.cu
Dense linear system solver using `cusolverDn` API

Compilation:
```
module load cuda/11.7.0-7sdye3
module load gcc
module load cmake

nvcc cusolver_dense.cu -o dense -lcusolver
```

### cusolver_sparse.cu
Sparse linear system solver using `cusolverSp` API

Compilation:
```
module load cuda/11.7.0-7sdye3
module load gcc
module load cmake

nvcc cusolver_sparse.cu -o sparse -lcusolver -lcusparse -lcudart
```