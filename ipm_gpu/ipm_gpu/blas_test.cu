#include "blas_actors.h"

#include <iostream>
#include <cmath>
#include <assert.h>

#define CUTOFF 0.0001

using namespace std;

int main() {
    cout << "Testing functionalities of blas wrappers." << endl;
    double a_arr[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    vector a(5, a_arr);
    double b_arr[5] = {-3.0, 1.1, 17.2, 0.3, 2.0};
    vector b(5, b_arr);
    double A_arr[25] = {
        0.8147,    0.0975,    0.1576,    0.1419,    0.6557,
        0.9058,    0.2785,    0.9706,    0.4218,    0.0357,
        0.1270,    0.5469,    0.9572,    0.9157,    0.8491,
        0.9134,    0.9575,    0.4854,    0.7922,    0.9340,
        0.6324,    0.9649,    0.8003,    0.9595,    0.6787
    };
    matrix A(5, 5, A_arr);
    double B_arr[30] = {
        0.7577,    0.7060,    0.8235,    0.4387,    0.4898,    0.2760,
        0.7431,    0.0318,    0.6948,    0.3816,    0.4456,    0.6797,
        0.3922,    0.2769,    0.3171,    0.7655,    0.6463,    0.6551,
        0.6555,    0.0462,    0.9502,    0.7952,    0.7094,    0.1626,
        0.1712,    0.0971,    0.0344,    0.1869,    0.7547,    0.1190
    };
    matrix B(5, 6, B_arr);

    // // test vector norm
    // double real_norm = 7.4162;    
    // assert(abs(real_norm - computation::v_norm(a)) <= CUTOFF);
    // cout << "Vector norm correct" << endl;

    // // test vv-add
    // double ans_arr[5] = {-5.0, -2.9, 11.2, -7.7, -8.0};
    // vector ans(5, ans_arr);
    // vector b_copy(b);
    // computation::vv_add(a, b_copy, -2);
    // assert(abs(computation::v_norm(b_copy) - computation::v_norm(ans)) <= CUTOFF);
    // cout << "Vector addition correct" << endl;

    // // test vv-dot
    // double real_dot = 62.0;
    // assert(abs(real_dot - computation::vv_dot(a, b)) <= CUTOFF);
    // cout << "Vector dot product correct" << endl;

    // // test vv_pw_mult
    // double real_pw_arr[5] = {-3, 2.2, 51.6, 1.2, 10.0};
    // vector real_pw(5, real_pw_arr);
    // vector b_pw_copy(b);
    // computation::vv_pw_mult(a, b_pw_copy);
    // computation::vv_add(real_pw, b_pw_copy, -1);
    // assert(computation::v_norm(b_pw_copy) <= CUTOFF);
    // cout << "Vector pointwise product correct" << endl;

    // // test sv_mult
    // double real_sv_arr[5] = {-9.42, 3.4540, 54.008, 0.942, 6.28};
    // vector real_sv(5, real_sv_arr);
    // vector b_sv_copy(b);
    // computation::sv_mult(3.14, b_sv_copy);
    // computation::vv_add(real_sv, b_sv_copy, -1);
    // assert(computation::v_norm(b_sv_copy) <= CUTOFF);
    // cout << "Vector scalar product correct" << endl;

    // // test v_inv
    // double real_inv_arr[5] = {-0.3333, 0.9091, 0.0581, 3.3333, 0.5000};
    // vector real_inv(5, real_inv_arr);
    // vector b_inv_copy(b);
    // computation::v_inv(b_inv_copy);
    // computation::vv_add(real_inv, b_inv_copy, -1);
    // assert(computation::v_norm(b_inv_copy) <= CUTOFF);
    // cout << "Vector inverse product correct" << endl;

    // // test m_norm
    // double my_norm = computation::m_norm(A);
    // assert(abs(my_norm - 3.5774) <= CUTOFF);
    // cout << "Matrix norm correct" << endl;

    // test mv_mult
    printf("CPU Matrix-Vector multiplication\n");
    double real_mv_arr[5] = {2.2755, 11.6376, 18.8049, 17.9450, 14.3143};
    vector real_mv(5, real_mv_arr);
    vector mv_res(5);
    computation::mv_mult(A, b, mv_res, true);
    // printf("Expected Output\n");
    // mv_res.display();
    computation::vv_add(real_mv, mv_res, -1);
    assert(computation::v_norm(mv_res) <= CUTOFF);
    cout << "CPU Matrix-Vector product correct" << endl;

    printf("GPU Matrix_Vector multiplication\n");
    vector real_mv2(5, real_mv_arr);
    vector mv_res2(5);
    computation::gpu_mv_mult(A, b, mv_res2, true);
    // printf("Calculated Output\n");
    // mv_res2.display();
    computation::vv_add(real_mv2, mv_res2, -1);
    assert(computation::v_norm(mv_res2) <= CUTOFF);
    cout << "GPU Matrix-Vector product correct" << endl;

    // // test vm_mult
    // double real_vm_arr[30] = {
    //     -2.2731,   -2.1180,   -2.4705,   -1.3161,   -1.4694,   -0.8280,
    //     0.8174,    0.0350,    0.7643,    0.4198,    0.4902,    0.7477,
    //     6.7458,    4.7627,    5.4541,   13.1666,   11.1164,   11.2677,
    //     0.1966,    0.0139,    0.2851,    0.2386,    0.2128,    0.0488,
    //     0.3424,    0.1942,    0.0688,    0.3738,    1.5094,    0.2380
    // };
    // matrix real_vm(5, 6, real_vm_arr);
    // matrix B_vm_copy(B);
    // computation::vm_mult(b, B_vm_copy);
    // computation::mm_add(real_vm, B_vm_copy, -1);
    // assert(computation::m_norm(B_vm_copy)/5.0 <= CUTOFF);
    // cout << "Vector-Matrix product correct" << endl;

    // test mm_mult
    double real_mm_arr[30] = {
        2.0472,    0.7428,    2.2302,    1.6448,    2.0100,    1.1475,
        1.2882,    0.3671,    1.3902,    1.5094,    1.9328,    0.8450,
        1.6713,    0.5073,    1.5964,    1.7078,    2.0767,    1.5044,
        1.4636,    0.4969,    1.4860,    1.7335,    2.1354,    1.1687,
        1.5848,    0.8082,    1.7449,    1.8208,    2.0606,    0.9941
    };
    matrix real_mm(5, 6, real_mm_arr);
    matrix my_mm(5, 6);
    printf("CPU Matrix_Matrix multiplication\n");
    computation::mm_mult(A, B, my_mm, true, false);
    computation::mm_add(real_mm, my_mm, -1);
    // printf("Correct output\n");
    // my_mm.display();
    assert(computation::m_norm(my_mm)/5.0 <= CUTOFF);
    cout << "CPU Matrix-Matrix product correct" << endl;


    // test gpu_mm_mult
    matrix real_mm2(5, 6, real_mm_arr);
    matrix my_mm2(5,6);
    printf("GPU Matrix_Matrix multiplication\n");
    computation::gpu_mm_mult(A, B, my_mm2, true, false);
    computation::mm_add(real_mm2, my_mm2, -1);
    // printf("Calculated output\n");
    // my_mm2.display();
    assert(computation::m_norm(my_mm2)/5.0 <= CUTOFF);
    cout << "GPU Matrix-Matrix product correct" << endl;

}