#include <stdio.h>
#include <cusolverSp.h>
#include <cuda_runtime_api.h>

int main() {
    cusolverSpHandle_t cusolverHandle = NULL;
    cusparseMatDescr_t descrA = NULL;
    int rowsA = 3;
    int colsA = 3;
    int nnzA = 5;

    // CSR representation of matrix A
    int h_csrRowPtrA[4] = {0, 2, 4, 5};
    int h_csrColIndA[5] = {0, 1, 1, 2, 2};
    double h_csrValA[5] = {4.0, -1.0, -1.0, 4.0, -2.0};
    double h_b[3] = {1.0, 2.0, 3.0}; // vector b

    // allocate memory
    int *d_csrRowPtrA, *d_csrColIndA;
    double *d_csrValA, *d_b, *d_x;
    cudaMalloc((void **)&d_csrRowPtrA, sizeof(int) * (rowsA + 1));
    cudaMalloc((void **)&d_csrColIndA, sizeof(int) * nnzA);
    cudaMalloc((void **)&d_csrValA, sizeof(double) * nnzA);
    cudaMalloc((void **)&d_b, sizeof(double) * rowsA);
    cudaMalloc((void **)&d_x, sizeof(double) * colsA);
    cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int) * (rowsA + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrValA, h_csrValA, sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(double) * rowsA, cudaMemcpyHostToDevice);

    // create cuSolver and cuSparse handle
    cusolverSpCreate(&cusolverHandle);
    cusparseCreateMatDescr(&descrA);

    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    int singularity = 0; // check if matrix is singular

    // solve linear system
    cusolverSpDcsrlsvqr(cusolverHandle, rowsA, nnzA, descrA,
                        d_csrValA, d_csrRowPtrA, d_csrColIndA, d_b,
                        0.0,
                        0,
                        d_x,
                        &singularity);

    if (singularity >= 0) {
        printf("Matrix is singular at %d\n", singularity);
    } else {
        double h_x[3];
        cudaMemcpy(h_x, d_x, sizeof(double) * colsA, cudaMemcpyDeviceToHost);
        printf("Solution: \n");
        for (int i = 0; i < colsA; i++) {
            printf("%f\n", h_x[i]);
        }
    }

    // free memory
    cusolverSpDestroy(cusolverHandle);
    cusparseDestroyMatDescr(descrA);
    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIndA);
    cudaFree(d_csrValA);
    cudaFree(d_b);
    cudaFree(d_x);

    return 0;
}
