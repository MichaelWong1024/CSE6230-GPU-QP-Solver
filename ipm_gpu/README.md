# QP Solver with IPM using GPU
## File Structure
```
.
├── blas_actors:    Linear algebra objects and operations 
├── cpu_ipm:        Driver for CPU version of IPM QP Solver 
└── blas_test:      Test correctness of linear algebra operations
```
## Usage
### Pre-Requisite
Run
```Makefile
# on pace-ice, use module load intel
module load mkl
module load cuda
```
to load the required libraries
### Artifacts
```Makefile
make clean      # clean all artifacts
make blas_test  # build the test for cublas operations (modified to compare cpu and gpu results)
make cpu_ipm    # build the driver for ipm using only cpu
make gpu_ipm    # build the driver for ipm using gpu
make            # build both cpu_ipm and blas_test
```
### Testing
Make the `blas_test` to obtain `blas_test.out`. Executing it to test all the implemented blas operations. 
```
%prompt make test_cublas -> verify gpu implemenation of matrix-vector and matrix-matrix multiplication

```
### IPM
%prompt make cpu_ipm  # to obtain the CPU version of IPM QP solver. Output is saved at 'cpu_ipm.log'. Find below the elapse timed for each step.
% prompt make gpu_ipm # to obtain the CPU version of IPM QP solver. Output is saved at 'gpu_ipm.log'.
```

The driver currently performs IPM on a `5x5` cost function and `10x5` cosntraint. The error in each iteration should decrease.
