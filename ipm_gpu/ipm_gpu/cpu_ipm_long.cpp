#include <iostream>
#include <cassert>
#include <iostream>
#include <algorithm>
#include "mkl.h"
#include "blas_actors.h"

#define STEP 0.25
#define IPM_ERR 1
#define IPM_SUCCESS 0

int ipm(matrix& Q, vector& c, matrix& J, vector& g, int itr) {
  std::cout << J.info() << std::endl;
  assert(Q.n == Q.m);
  assert(Q.n == J.n);
  assert(g.n == J.m);
  assert(Q.n == c.n);

  int num_var = Q.n;
  int num_ineq = g.n;

  double posi_init[num_ineq];
  std::fill(posi_init, posi_init+num_ineq, 1.0);
  double mu = 1.0/num_ineq;

  vector z(num_var), lambda(num_ineq, posi_init), t(num_ineq, posi_init);

  for (int i=0; i<itr; i++) {
    // r_d = Q*z_k + J' * lambda_k + c
    vector rd(num_var);
    vector rd_temp(num_var);
    computation::mv_mult(Q, z, rd, false);          // rd = Q @ z + rd
    computation::mv_mult(J, lambda, rd_temp, true); // rd_temp  = J' @ lambda  + rd_temp
    computation::vv_add(rd_temp, rd, 1.0);          // rd = (1 * rd) + rd_temp = (Q@z_k) + (J' @ lambda_k)
    computation::vv_add(c, rd, 1.0);                // rd = (rd * 1) + c 
    // rd.display();

    // r_b = -J*z_k + g - t_k
    vector rb(num_ineq);
    vector rb_temp(num_ineq);
    computation::mv_mult(J, z, rb_temp, false);     // rb_temp = J @ z + rb_temp
    computation::vv_add(rb_temp, rb, -1.0);         // rb = -rb_temp + rb
    computation::vv_add(g, rb, 1.0);                // rb = g + rb
    computation::vv_add(t, rb, -1.0);               // rb = -t + rb 
    // rb.display();

    // LHS_4 = Q + J'(lambda_k * t_k * J)
    matrix LHS_4(num_var, num_var);
    matrix LHS_4_temp(J);
    computation::vm_mult(t, LHS_4_temp);                      // LHS_4_temp = t_k @ LHS_4_temp = tk @ J
    computation::vm_mult(lambda, LHS_4_temp);                 // LHS_4_temp = lambda_k @ LHS_4_temp
    computation::mm_mult(J, LHS_4_temp, LHS_4, true, false);  // LHS_4 = J' @ LHS_4_temp + LHS_4
    computation::mm_add(Q, LHS_4, 1.0);                       // LHS_4 = 1*Q + LHS_4 
    // LHS_4.display();

    // RHS_4 = -r_d - J' * inv(t_k) * lambda_k * (-r_b - t_k + step * mu_k * inv(lambda_k))
    vector inv_t(t);
    vector inv_lambda(lambda);

    computation::v_inv(inv_t);                        // inv_t = 1/t
    computation::v_inv(inv_lambda);                   // inv_lambda = 1/ lambda
    //inv_t.display();
    //inv_lambda.display();

    vector RHS_4_temp(inv_lambda);
    vector RHS_4(num_var);
    computation::sv_mult(STEP*mu, RHS_4_temp);        // RHS_4_temp = RHS_4_temp * STEP*mu = step * mu_k * inv(lambda_k)
    computation::vv_add(t, RHS_4_temp, -1.0);         // RHS_4_temp = -t + RHS4_temp
    computation::vv_add(rb, RHS_4_temp, -1.0);        // RHS_4_temp = -rb + RHS4_temp
    computation::vv_pw_mult(lambda, RHS_4_temp);      // RHS_4_temp = lambda .* RHS_4_temp : This is point wise multiplication 
    computation::vv_pw_mult(inv_t, RHS_4_temp);       // RHS_4_temp = inv(t) .* RHS_4_temp : This is point wise multiplication
    computation::mv_mult(J, RHS_4_temp, RHS_4, true); // RHS_4 = J' @ RHS_4_temp + RHS_4
    computation::sv_mult(-1.0, RHS_4);                // RHS_4 = -1 * RHS_4
    computation::vv_add(rd, RHS_4, -1.0);             // RHS_4 = -r_d - RHS_4
    //RHS_4.display();

    // LHS_4 * delta_z = RHS_4
    // result is save dinto RHS_4 ??
    int ipiv[num_var];
    int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, num_var, 1, LHS_4.entries, num_var, ipiv, RHS_4.entris, 1);
    if (info > 0) {
      std::cerr << "The given system cannot be solved." << std::endl;
      return IPM_ERR;
    }
    //RHS_4.display();

    vector delta_z_l(RHS_4);

    // delta_L = lambda_k * inv(t_k) * (J * delta_z - r_b) - lambda_k + step * mu_k * inv(lambda_k)
    vector delta_L(inv_lambda);
    computation::sv_mult(STEP * mu, delta_L);                 // delta_L = (mu_k * step) * delta_L
    computation::vv_add(lambda, delta_L, -1);                 // delta_L = -1 * lambda + delta_L
    vector delta_L_temp(num_ineq);                                
    computation::mv_mult(J, delta_z_l, delta_L_temp, false);  // delta_L_temp = J @ delta_z_l + delta_L_temp
    computation::vv_add(rb, delta_L_temp, -1);                // delta_L_temp = -1 * rb + delta_L_temp
    computation::vv_pw_mult(inv_t, delta_L_temp);             // delta_L_temp = inv(t) .* delta_L_temp
    computation::vv_pw_mult(lambda, delta_L_temp);            // delta_L_temp = lambda .* delta_L_temp
    computation::vv_add(delta_L_temp, delta_L, 1.0);          // delata_L = 1* delat_L_temp + delta_L
    //std::cout << "Check" << std::endl;

    // delta_t = -t_k + inv(lambda_k) * (step * mu_k * e - T * delta_lambda);
    vector yee(num_ineq, posi_init); // create vector yee and initialize it with the value fo posi_init
    vector delta_t(delta_L);
    computation::vv_pw_mult(t, delta_t);          // delta_t = t .* delta_t
    computation::sv_mult(-1.0, delta_t);          // delta_t = -1 * delta_t
    computation::vv_add(yee, delta_t, STEP * mu); // delta_t = (step * mu) * yee + delta_t
    computation::vv_pw_mult(inv_lambda, delta_t); // delta_t = inv(lambda) .* delta-t
    computation::vv_add(t, delta_t, -1.0);        // delta_t = -1 * t + delta_t 
    //std::cout << "Check" << std::endl;

    // alpha
    double alpha = 1.5E-2;

    // z, lambda, t += alpha (delta_z, delta_lambda, delta_t)
    computation::vv_add(RHS_4, z, alpha);         // z = alpha * RHS_4 + z
    computation::vv_add(delta_L, lambda, alpha);  // lambda = alpha * delta_l + lambda
    computation::vv_add(delta_t, t, alpha);       // t = alpha * delta_t + t
    //std::cout << "Check" << std::endl;

    // mu_k = t_k * lambda_k / num_ineq
    mu = computation::vv_dot(t, lambda) / num_ineq; 
    if (i % 25 == 0) {
      std::cout << mu << std::endl;
      z.display();
    }
  }

  return IPM_SUCCESS;
}

int main() {
  double Q_arr[100] = {
    2.7345,    1.8859,    2.0785,    1.9442,    1.9567,
    1.8859,    2.2340,    2.0461,    2.3164,    2.0875,
    2.0785,    2.0461,    2.7591,    2.4606,    1.9473,
    1.9442,    2.3164,    2.4606,    2.5848,    2.2768,
    1.9567,    2.0875,    1.9473,    2.2768,    2.4853,
    2.7345,    1.8859,    2.0785,    1.9442,    1.9567,
    1.8859,    2.2340,    2.0461,    2.3164,    2.0875,
    2.0785,    2.0461,    2.7591,    2.4606,    1.9473,
    1.9442,    2.3164,    2.4606,    2.5848,    2.2768,
    1.9567,    2.0875,    1.9473,    2.2768,    2.4853,
    2.7345,    1.8859,    2.0785,    1.9442,    1.9567,
    1.8859,    2.2340,    2.0461,    2.3164,    2.0875,
    2.0785,    2.0461,    2.7591,    2.4606,    1.9473,
    1.9442,    2.3164,    2.4606,    2.5848,    2.2768,
    1.9567,    2.0875,    1.9473,    2.2768,    2.4853,
    2.7345,    1.8859,    2.0785,    1.9442,    1.9567,
    1.8859,    2.2340,    2.0461,    2.3164,    2.0875,
    2.0785,    2.0461,    2.7591,    2.4606,    1.9473,
    1.9442,    2.3164,    2.4606,    2.5848,    2.2768,
    1.9567,    2.0875,    1.9473,    2.2768,    2.4853
  };
  matrix Q(10, 10, Q_arr);

  double c_arr[10] = {
    0.7577,
    0.7431,
    0.3922,
    0.6555,
    0.1712,
    0.7577,
    0.7431,
    0.3922,
    0.6555,
    0.1712
  };
  vector c(10, c_arr);

  double J_arr[100] = {
    0.7060,    0.4387,    0.2760,    0.7513,    0.8407,
    0.0318,    0.3816,    0.6797,    0.2551,    0.2543,
    0.2769,    0.7655,    0.6551,    0.5060,    0.8143,
    0.0462,    0.7952,    0.1626,    0.6991,    0.2435,
    0.0971,    0.1869,    0.1190,    0.8909,    0.9293,
    0.8235,    0.4898,    0.4984,    0.9593,    0.3500,
    0.6948,    0.4456,    0.9597,    0.5472,    0.1966,
    0.3171,    0.6463,    0.3404,    0.1386,    0.2511,
    0.9502,    0.7094,    0.5853,    0.1493,    0.6160,
    0.0344,    0.7547,    0.2238,    0.2575,    0.4733,
    0.7060,    0.4387,    0.2760,    0.7513,    0.8407,
    0.0318,    0.3816,    0.6797,    0.2551,    0.2543,
    0.2769,    0.7655,    0.6551,    0.5060,    0.8143,
    0.0462,    0.7952,    0.1626,    0.6991,    0.2435,
    0.0971,    0.1869,    0.1190,    0.8909,    0.9293,
    0.8235,    0.4898,    0.4984,    0.9593,    0.3500,
    0.6948,    0.4456,    0.9597,    0.5472,    0.1966,
    0.3171,    0.6463,    0.3404,    0.1386,    0.2511,
    0.9502,    0.7094,    0.5853,    0.1493,    0.6160,
    0.0344,    0.7547,    0.2238,    0.2575,    0.4733     
  };
  matrix J(10, 10, J_arr);

  double g_arr[10] = {
    0.3517,
    0.8308,
    0.5853,
    0.5497,
    0.9172,
    0.2858,
    0.7572,
    0.7537,
    0.3804,
    0.5678
  };
  vector g(10, g_arr);

  ipm(Q, c, J, g, 1000);
}