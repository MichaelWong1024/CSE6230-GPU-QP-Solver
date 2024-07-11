#ifndef BLAS_ACTORS_H
#define BLAS_ACTORS_H

#include <stdlib.h>
#include <cstring>
#include <string>
#include <iostream>

#define DEFAULT_INC 1

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

    std::string info();
};

class vector {
public:
    int n;
    double * entris;

    vector(int n) {
        this->n = n;
        this->entris = (double *) calloc(n, sizeof(double));
    }

    vector(int n, double * entris) {
        this->n = n;
        this->entris = (double *) calloc(n, sizeof(double));
        memcpy(this->entris, entris, n * sizeof(double));
    }

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
};
#endif