#ifndef BLAS_ACTOR_CPP
#define BLAS_ACTOR_CPP

#include <sstream>
#include <cmath>
#include <iostream>
#include "mkl.h"
#include "cublas_v2.h"

#include "blas_actors.h"
#include "support.h"
#include <stdio.h>


std::string matrix::info() {
    std::ostringstream result;
    result << "Matrix size [" << this->m << ", " << this->n << "]";
    return result.str();
}

std::string vector::info() {
    std::ostringstream result;
    result << "Vector size " << this->n;
    return result.str();
}

// Vector-Vector addition
    /*    
        Computes a constant times a vector plus a vector (double-precision).
        the contents of vector b are replaced with the result. 
        The value computed is (alpha * a[i]) + b[i].
    */   
void computation::vv_add(vector& a, vector& b, double scalar) {
    cblas_daxpy(a.n, scalar, a.entris, DEFAULT_INC, b.entris, DEFAULT_INC);
}

// Vector dotproduct
    /*
        Computes the dot product of two vectors (double-precision).
        The dot product return a vector of double-precision
    */
double computation::vv_dot(vector& a, vector& b) {
    return cblas_ddot(a.n, a.entris, DEFAULT_INC, b.entris, DEFAULT_INC);
}

// Vector-Vector point-wise multiplication
    /*
        Point wise multiplication between two vectors.
        b[i] = a[i] * b[i]
        Vector b is replaced by the result.
    */
void computation::vv_pw_mult(vector& a, vector& b) {
    for (int i=0; i<a.n; i++) {
        b.entris[i] *= a.entris[i];
    }
}

// Matrix-Vector multplication
    /*
    Multiplies a matrix by a vector (double precision).
    This function multiplies A * b (after transposing A, if needed)
    and multiplies the resulting matrix by alpha.
    It then multiplies vector res by  beta. It stores the sum of 
    these two products in vector res.

    res = alpha * (A @ b) + beta * res
    */
void computation::mv_mult(matrix& A, vector& b, vector& res, bool tp) {
    cblas_dgemv(CblasRowMajor, tp ? CblasTrans : CblasNoTrans,
        A.m, A.n, 1, A.entries, A.n, b.entris, DEFAULT_INC, 1, res.entris, DEFAULT_INC);
}

// Scales a symmetric matrix
/*
Scales a symmetric matrix, multiplies by a vector, 
then scales and adds another vector (single precision).
Computes alpha*A*b + beta*res and stores the results in res.
in our case -> res = A @ b + res
*/
void computation::smv_mult(matrix& A, vector& b, vector& res) {
    cblas_dsymv(CblasRowMajor, CblasUpper, A.m, 1, A.entries, A.n, b.entris, DEFAULT_INC, 1, res.entris, DEFAULT_INC);
}

// Vector-Matrix multiplication
    /*
    Use cblas_dscal which Multiplies each element of a vector by
     a constant (double-precision) to do vector-matrix multiplication
     Result is saved in vector a.
    */
void computation::vm_mult(vector& a, matrix& A) {
    for (int i=0; i<a.n; i++) {
        cblas_dscal(A.n, a.entris[i], &(A.entries[i * A.n]), DEFAULT_INC);
    }
}

// Matrix-Matrix multiplication
    /*
        Multiplies two matrices (double-precision).
        This function multiplies A * B and multiplies the resulting matrix by alpha. 
        It then multiplies matrix res by beta. 
        It stores the sum of these two products in matrix res.
        Thus, it calculates either

        res <- alpha * (A @ B) + beta * res

        or

        res <- alpha * (B @ A) +  beta * res

        with optional use of transposed forms of A, B, or both.
    */
void computation::mm_mult(matrix& A, matrix& B, matrix& res, bool tpa, bool tpb) {
    cblas_dgemm(CblasRowMajor, tpa ? CblasTrans : CblasNoTrans, tpb ? CblasTrans : CblasNoTrans, tpa ? A.n : A.m, tpb ? B.m : B.n, tpa ? A.m : A.n, 1, A.entries, A.n, B.entries, B.n, 1, res.entries, res.n);
}


// Gen_matrix - Sym_matrix mulitplication
    /*
    Multiplies a matrix by a symmetric matrix (double-precision).
    This function multiplies A * B or B * A (depending on the value of Side) 
    and multiplies the resulting matrix by alpha. It then multiplies matrix res by beta.
     It stores the sum of these two products in matrix res.

    res = alpha * (A @ B) + beta * res
    or
    res = alpha * (B @ A) + beta * res
    */
void computation::smm_mult(matrix& A, matrix& B, matrix& res) {
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, res.m, res.n, 1, A.entries, A.n, B.entries, B.n, 1, res.entries, res.n);  
}


// Matrix-Matrix Addition
/*
    cblas_daxpy is resude for matrix-matrix addrition.
    With the for loop each time two rows of the matrix are added and saved in
    matrix B 
    B = scalar * A + B ?
*/
void computation::mm_add(matrix& A, matrix& B, double scalar) {
    for (int i=0; i<A.m; i++) {
        cblas_daxpy(A.n, scalar, &(A.entries[i * A.n]), DEFAULT_INC, &(B.entries[i * A.n]), DEFAULT_INC);
    }
}

// Residual norm of matrix 
    /*
        Compute the L2 norm of each row and add them to compute  a final norm
        res = sqrt (sum (|Ai|_2)) where |Ai|_2 is the L2 norm of row Ai.
        the resulta is return.
        Can be used to check the result of matrix operations. Norm of calculated and 
        expected values should be close enough
    */
double computation::m_norm(matrix& A) {
    double sum = 0.0;
    for (int i=0; i<A.m; i++) {
        double part = cblas_dnrm2(A.n, &(A.entries[i * A.n]), DEFAULT_INC);
        sum += part * part;
    }
    return sqrt(sum);
}

// Vector scaling 
    /*
        Multiplies each element of a vector by a constant (double-precision).
        b[i] = a * b[i]
    */
void computation::sv_mult(double a, vector& b) {
    cblas_dscal(b.n, a, b.entris, DEFAULT_INC);
}


// Vector norm
/*
     Compute and return the L2 norm of a vector a
*/
double computation::v_norm(vector& a) {
    return cblas_dnrm2(a.n, a.entris, DEFAULT_INC);
}

// Vector inversion
    /*
        Compute the inverse of a vector by inverting each entry.
        The result are saved back into the vector.
    */
void computation::v_inv(vector& a) {
    for (int i=0; i<a.n; i++) {
        a.entris[i] = 1.0 / a.entris[i];
    }
}

// CUBLAS CODE ----------------------------------------------------------------------
void computation::gpu_mv_mult(matrix& A_h, double * A_d, vector& x_h, double* x_d, vector& y_h, double* y_d, bool tp, cublasHandle_t& handle){
    // printf("Input of the module \n");
    // A_h.display();
    // B_h.display();
    cudaError_t cuda_ret;
    cublasStatus_t stat;     // CUBLAS functions status
    
    const int M = A_h.m;
    const int N = A_h.n;

    // Copy host variables to device ------------------------------------------
    CHECK_CUBLAS(cublasCreate(&handle), "Failed handle creation"); // initialize CUBLAS context

    CHECK_CUBLAS(cublasSetMatrix(M, N, sizeof(*A_h.entries), A_h.entries, M, A_d, M), "Failed set matrix");

    CHECK_CUBLAS(cublasSetVector(tp ? M : N, sizeof(*x_h.entris), x_h.entris, 1, x_d, 1), "Failed set vector");

    CHECK_CUBLAS(cublasSetVector(tp ? N : M, sizeof(*y_h.entris), y_h.entris, 1, y_d, 1), "Failed set vector");

    double al    = 1.0;
    double bet   = 1.0;

    cudaDeviceSynchronize();

    // Perform matrix-vector multiplication -------------------------------------------
    stat = cublasDgemv(handle,
                        tp ? CUBLAS_OP_N : CUBLAS_OP_T, 
                        tp ? A_h.n : A_h.m, 
                        tp ? A_h.m : A_h.n, 
                        &al, A_d, N, x_d, 1, &bet, y_d, 1);
    
    
    cudaDeviceSynchronize();
    CHECK_CUBLAS(stat, "Failed mat-vec");

    // Copy device variable from host-------------------------------------------
    CHECK_CUBLAS(cublasGetVector(tp ? N : M, sizeof(*y_h.entris), y_d, 1, y_h.entris, 1), "Failed get vector"); // copy C_d -> C_h
    CHECK_CUBLAS(cublasDestroy(handle), "Failed destruction.");
}



void computation::gpu_mm_mult(matrix& A_h, double* A_d, matrix& B_h, double* B_d, matrix& C_h, double* C_d, bool tpa, bool tpb, cublasHandle_t& handle){

    // printf("Input of the module \n");
    // A_h.display();
    // B_h.display();
    
    const int M = tpa ? A_h.n : A_h.m;
    const int K = tpa ? A_h.m : A_h.n;
    const int N = tpb ? B_h.m : B_h.n;

    // Allocate device variables ----------------------------------------------

    // Copy host variables to device ------------------------------------------
    CHECK_CUBLAS(cublasCreate(&handle), "Failed creation.");

    CHECK_CUBLAS(cublasSetMatrix(M, K, sizeof(*A_h.entries), A_h.entries, M, A_d, M), "Unable to copy memory to device");

    CHECK_CUBLAS(cublasSetMatrix(K, N, sizeof(*B_h.entries), B_h.entries, K, B_d, K), "Unable to copy memory to device");

    CHECK_CUBLAS(cublasSetMatrix(M, N, sizeof(*C_h.entries), C_h.entries, M, C_d, M), "Unable to copy memory to device");

    double al    = 1.0;
    double bet   = 1.0;

    cudaDeviceSynchronize();

    // Perform matrix multiplication -------------------------------------------
    /*Since cublas uses column major we will do C' = B' @ A' to get backe the
    correct row major code*/
    // stat = cublasDgemm(handle,
    //                           tpb ? CUBLAS_OP_T : CUBLAS_OP_N, 
    //                           tpa ? CUBLAS_OP_T : CUBLAS_OP_N, 
    //                           tpb ? K : N, 
    //                           tpa ? K : M,
    //                           tpb ? N : K, 
    //                           &al, B_d, N, A_d, K, &bet, C_d, N);
    //                           // modify this to look like cpu cblas

    CHECK_CUBLAS(cublasDgemm(handle,
                        tpa ? CUBLAS_OP_T : CUBLAS_OP_N, 
                        tpb ? CUBLAS_OP_T : CUBLAS_OP_N, 
                        M, N, K, 
                        &al, A_d, A_h.m, B_d, B_h.m, &bet, C_d, C_h.m), "matmul failed");
    
    cudaDeviceSynchronize();
    // Copy device variable from host-------------------------------------------
    CHECK_CUBLAS(cublasGetMatrix(N, M, sizeof(double), C_d, N, C_h.entries, N), "Failed to copy to host"); // copy C_d -> C_h

    cudaDeviceSynchronize();
    CHECK_CUBLAS(cublasDestroy(handle), "Failed destruction.");
}

#endif