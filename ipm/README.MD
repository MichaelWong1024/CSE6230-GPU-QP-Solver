# QP Solver with IPM
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
```
module load gcc
module load mkl
```
to load the required libraries
### Artifacts
```Makefile
make clean      # clean all artifacts
make blas_test  # build the test for blas operations
make cpu_ipm    # build the driver for ipm
make            # build both cpu_ipm and blas_test
```
### Testing
Make the `blas_test` to obtain `blas_test.out`. Executing it to test all the implemented blas operations. The expected result should be
```
[swang980@login-ice-3 ipm]$ ./blas_test.out
Testing functionalities of blas wrappers.
Vector norm correct
Vector addition correct
Vector dot product correct
Vector pointwise product correct
Vector scalar product correct
Vector inverse product correct
Matrix norm correct
Matrix-Vector product correct
Vector-Matrix product correct
Matrix-Matrix product correct
```
### IPM
Make the `cpu_ipm` to obtain the CPU version of IPM QP solver. The expected result should be
```
Matrix size [10, 5]
0.985386
-0.00683172 -0.0115425 0.00784131 -0.00617549 0.00833539

0.742601
-0.146656 -0.313695 0.18057 -0.151094 0.190682

0.559854
-0.249277 -0.534464 0.323012 -0.275874 0.361196
...

0.136539
-0.543496 -0.743134 0.817743 -0.955524 1.06377

0.102951
-0.59148 -0.657991 0.906627 -1.13853 1.16709

0.0776151
-0.634275 -0.568132 0.987159 -1.31026 1.25217
```
The driver currently performs IPM on a `5x5` cost function and `10x5` cosntraint. The error in each iteration should decrease.