#ifndef BLAS_ACTORS_H
#define BLAS_ACTORS_H

#include <stdlib.h>
#include <cstring>
#include <string>
#include <iostream>

#include "cublas_v2.h"

#define DEFAULT_INC 1

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#define CHECK_CUSOLVER(call, msg) { \
    cusolverStatus_t err = call; \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "%s\n", msg); \
        fprintf(stderr, "Error code: %d\n", err); \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call, msg) { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "%s\n", msg); \
        fprintf(stderr, "Error: %s\n", cublasGetStatusString(err)); \
        exit(1); \
    } \
}


class matrix {
public:
    int m, n;
    double * entries;

    matrix(int m, int n) {
        this->m = m;
        this->n = n;
        this->entries = (double *)calloc(m*n, sizeof(double));
    }

    matrix(int m, int n, double * entris) {
        this->m = m;
        this->n = n;
        this->entries = (double *)calloc(m*n, sizeof(double));
        memcpy(this->entries, entris, m*n*sizeof(double));
    }

    matrix(matrix& original) {
        this->m = original.m;
        this->n = original.n;
        this->entries = (double *)calloc(m*n, sizeof(double));
        memcpy(this->entries, original.entries, m*n*sizeof(double));
    }

    void display() {
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                std::cout << this->entries[i*j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    ~matrix() {
        delete(entries);
    }

    std::string info();
};

class vector {
public:
    int n;
    double * entris;

    // Constructor to create a vector of dimension n with all components initialized to 0
    vector(int n) {
        this->n = n;
        this->entris = (double *) calloc(n, sizeof(double));
    }

    // Constructor to create a vector of dimension n with components copied from an existing array
    vector(int n, double * entris) {
        this->n = n;
        this->entris = (double *) calloc(n, sizeof(double));
        memcpy(this->entris, entris, n * sizeof(double));
    }

    // Copy constructor to create a copy of an existing vector
    vector(vector& original) {
        this->n = original.n;
        this->entris = (double *) calloc(this->n, sizeof(double));
        memcpy(this->entris, original.entris, n * sizeof(double));
    }

    void display() {
            for (int j=0; j<n; j++) {
                std::cout << this->entris[j] << " ";
            }
            std::cout << std::endl << std::endl;;
    }
    std::string info();

    ~vector() {
        delete(entris);
    }
};

class computation {
public:
    // vector-vector operations
    static void vv_add(vector& a, vector& b, double scalar);
    static double vv_dot(vector& a, vector& b);
    static void vv_pw_mult(vector& a, vector& b);

    // matrix-vector operations
    static void mv_mult(matrix& A, vector& b, vector& res, bool tp);
    static void smv_mult(matrix& A, vector& b, vector& res);
    static void vm_mult(vector& a, matrix& A);

    // matrix-matrix operations
    static void mm_mult(matrix& A, matrix& B, matrix& res, bool tpa, bool tpb);
    static void smm_mult(matrix& A, matrix& B, matrix& res);
    static void mm_add(matrix& A, matrix& B, double scalar);
    static double m_norm(matrix& A);

    // vector operations
    static void sv_mult(double a, vector& b);
    static double v_norm(vector& a);
    static void v_inv(vector& a);

    // GPU code
   static void gpu_mm_mult(matrix& A_h, double* A_d, matrix& B_h, double* B_d,
        matrix& C_h, double* C_d, bool tpa, bool tpb, cublasHandle_t& handle);
   static void gpu_mv_mult(matrix& A_h, double* A_d, vector& x_h, double* x_d,
        vector& y_h, double* y_d, bool tp, cublasHandle_t& handle);
};

#endif