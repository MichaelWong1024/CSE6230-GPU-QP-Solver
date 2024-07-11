#ifndef BLAS_ACTOR_CPP
#define BLAS_ACTOR_CPP

#include <sstream>
#include <cmath>
#include <iostream>
#include "mkl.h"

#include "blas_actors.h"

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

void computation::vv_add(vector& a, vector& b, double scalar) {
    cblas_daxpy(a.n, scalar, a.entris, DEFAULT_INC, b.entris, DEFAULT_INC);
}

double computation::vv_dot(vector& a, vector& b) {
    return cblas_ddot(a.n, a.entris, DEFAULT_INC, b.entris, DEFAULT_INC);
}

void computation::vv_pw_mult(vector& a, vector& b) {
    for (int i=0; i<a.n; i++) {
        b.entris[i] *= a.entris[i];
    }
}

void computation::mv_mult(matrix& A, vector& b, vector& res, bool tp) {
    cblas_dgemv(CblasRowMajor, tp ? CblasTrans : CblasNoTrans,
        A.m, A.n, 1, A.entries, A.n, b.entris, DEFAULT_INC, 1, res.entris, DEFAULT_INC);
}

void computation::smv_mult(matrix& A, vector& b, vector& res) {
    cblas_dsymv(CblasRowMajor, CblasUpper, A.m, 1, A.entries, A.n, b.entris, DEFAULT_INC, 1, res.entris, DEFAULT_INC);
}

void computation::vm_mult(vector& a, matrix& A) {
    for (int i=0; i<a.n; i++) {
        cblas_dscal(A.n, a.entris[i], &(A.entries[i * A.n]), DEFAULT_INC);
    }
}

void computation::mm_mult(matrix& A, matrix& B, matrix& res, bool tpa, bool tpb) {
    cblas_dgemm(CblasRowMajor, tpa ? CblasTrans : CblasNoTrans, tpb ? CblasTrans : CblasNoTrans, tpa ? A.n : A.m, tpb ? B.m : B.n, tpa ? A.m : A.n, 1, A.entries, A.n, B.entries, B.n, 1, res.entries, res.n);
}

void computation::smm_mult(matrix& A, matrix& B, matrix& res) {
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, res.m, res.n, 1, A.entries, A.n, B.entries, B.n, 1, res.entries, res.n);  
}

void computation::mm_add(matrix& A, matrix& B, double scalar) {
    for (int i=0; i<A.m; i++) {
        cblas_daxpy(A.n, scalar, &(A.entries[i * A.n]), DEFAULT_INC, &(B.entries[i * A.n]), DEFAULT_INC);
    }
}

double computation::m_norm(matrix& A) {
    double sum = 0.0;
    for (int i=0; i<A.m; i++) {
        double part = cblas_dnrm2(A.n, &(A.entries[i * A.n]), DEFAULT_INC);
        sum += part * part;
    }
    return sqrt(sum);
}

void computation::sv_mult(double a, vector& b) {
    cblas_dscal(b.n, a, b.entris, DEFAULT_INC);
}

double computation::v_norm(vector& a) {
    return cblas_dnrm2(a.n, a.entris, DEFAULT_INC);
}

void computation::v_inv(vector& a) {
    for (int i=0; i<a.n; i++) {
        a.entris[i] = 1.0 / a.entris[i];
    }
}

#endif