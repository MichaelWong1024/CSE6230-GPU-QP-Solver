#include <iostream>
#include <cassert>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <vector>
#include <sstream>
#include <array>
#include <memory>
#include "mkl.h"
#include "blas_actors.h"
#include "support.h" 

#define STEP 0.25
#define IPM_ERR 1
#define IPM_SUCCESS 0

int ipm(matrix& Q, vector& c, matrix& J, vector& g, int itr) {
  
  std::cout << J.info() << std::endl;
  assert(Q.n == Q.m);
  assert(Q.n == J.n);
  assert(g.n == J.m);
  assert(Q.n == c.n);
  Timer timer_rhs, timer_lhs, timer_dz, timer_total, timer_rd, timer_rb;
  Timer timer_dl, timer_dt;
  float timer_rhs_total     = 0.0f;
  float timer_lhs_total     = 0.0f;
  float timer_dz_total      = 0.0f;
  float timer_rd_total      = 0.0f;
  float timer_rb_total      = 0.0f;
  float timer_dl_total       = 0.0f;
  float timer_dt_total       = 0.0f;

  int num_var = Q.n;
  int num_ineq = g.n;

  double posi_init[num_ineq];
  std::fill(posi_init, posi_init+num_ineq, 1.0);
  double mu = 1.0/num_ineq;

  vector z(num_var), lambda(num_ineq, posi_init), t(num_ineq, posi_init);

  startTime(&timer_total);
  for (int i=0; i<itr; i++) {
    
    // r_d = Q*z_k + J' * lambda_k + c
    startTime(&timer_rd);
    vector rd(num_var);
    vector rd_temp(num_var);
    computation::mv_mult(Q, z, rd, false);          // rd = Q @ z + rd
    computation::mv_mult(J, lambda, rd_temp, true); // rd_temp  = J' @ lambda  + rd_temp
    computation::vv_add(rd_temp, rd, 1.0);          // rd = (1 * rd) + rd_temp = (Q@z_k) + (J' @ lambda_k)
    computation::vv_add(c, rd, 1.0);                // rd = (rd * 1) + c 
    // rd.display();
    stopTime(&timer_rd);
    

    startTime(&timer_rb);
    // r_b = -J*z_k + g - t_k
    vector rb(num_ineq);
    vector rb_temp(num_ineq);
    computation::mv_mult(J, z, rb_temp, false);     // rb_temp = J @ z + rb_temp
    computation::vv_add(rb_temp, rb, -1.0);         // rb = -rb_temp + rb
    computation::vv_add(g, rb, 1.0);                // rb = g + rb
    computation::vv_add(t, rb, -1.0);               // rb = -t + rb 
    // rb.display();
    stopTime(&timer_rb);

    startTime(&timer_lhs);
    // LHS_4 = Q + J'(lambda_k * t_k * J)
    matrix LHS_4(num_var, num_var);
    matrix LHS_4_temp(J);
    computation::vm_mult(t, LHS_4_temp);                      // LHS_4_temp = t_k @ LHS_4_temp = tk @ J
    computation::vm_mult(lambda, LHS_4_temp);                 // LHS_4_temp = lambda_k @ LHS_4_temp
    computation::mm_mult(J, LHS_4_temp, LHS_4, true, false);  // LHS_4 = J' @ LHS_4_temp + LHS_4
    computation::mm_add(Q, LHS_4, 1.0);                       // LHS_4 = 1*Q + LHS_4 
    // LHS_4.display();
    stopTime(&timer_lhs);
  
    startTime(&timer_rhs);
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
    stopTime(&timer_rhs);

    // LHS_4 * delta_z = RHS_4
    // result is save dinto RHS_4 ??
    startTime(&timer_dz);
    int ipiv[num_var];
    int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, num_var, 1, LHS_4.entries, num_var, ipiv, RHS_4.entris, 1);
    if (info > 0) {
      std::cerr << "The given system cannot be solved." << std::endl;
      return IPM_ERR;
    }
    //RHS_4.display();
    stopTime(&timer_dz);

    startTime(&timer_dl);
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
    stopTime(&timer_dl);

    startTime(&timer_dt);
    // delta_t = -t_k + inv(lambda_k) * (step * mu_k * e - T * delta_lambda);
    vector yee(num_ineq, posi_init); // create vector yee and initialize it with the value fo posi_init
    vector delta_t(delta_L);
    computation::vv_pw_mult(t, delta_t);          // delta_t = t .* delta_t
    computation::sv_mult(-1.0, delta_t);          // delta_t = -1 * delta_t
    computation::vv_add(yee, delta_t, STEP * mu); // delta_t = (step * mu) * yee + delta_t
    computation::vv_pw_mult(inv_lambda, delta_t); // delta_t = inv(lambda) .* delta-t
    computation::vv_add(t, delta_t, -1.0);        // delta_t = -1 * t + delta_t 
    //std::cout << "Check" << std::endl;
    stopTime(&timer_dt);
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
      // z.display();
      // printf("lhs %f s\n", elapsedTime(timer_lhs));
      // printf("rhs %f s\n", elapsedTime(timer_rhs));
      // printf("lapack %f s\n", elapsedTime(timer_dz));
    }

    timer_lhs_total += elapsedTime(timer_lhs);
    timer_rhs_total += elapsedTime(timer_rhs);
    timer_dz_total += elapsedTime(timer_dz);
    timer_rd_total += elapsedTime(timer_rd);
    timer_rb_total += elapsedTime(timer_rb);
    timer_dl_total += elapsedTime(timer_dl);
    timer_dt_total += elapsedTime(timer_dt);


  }

  stopTime(&timer_total);
    // timer_lhs_total += timer_lhs_total;
    // timer_rhs_total += timer_rhs_total;
    // timer_dz_total += timer_dz_total;

      printf("rd_total %f s\n", timer_rd_total);
      printf("rb_total %f s\n", timer_rb_total);
      printf("lhs_total %f s\n", timer_lhs_total);
      printf("rhs_total %f s\n", timer_rhs_total);
      printf("dz_total %f s\n", timer_dz_total);
      printf("dl_total %f s\n", timer_dl_total);
      printf("dt_total %f s\n", timer_dt_total);
      printf("Total_time %f s\n", elapsedTime(timer_total));
  return IPM_SUCCESS;
}

double* parse_doubles_from_string(const std::string& str, size_t& out_size) {
    std::vector<double> intermediate;
    std::stringstream ss(str);
    std::string item;
    while (getline(ss, item, ',')) {
        intermediate.push_back(stod(item));
    }

    double* result = new double[intermediate.size()];
    out_size = intermediate.size();

    for (size_t i = 0; i < intermediate.size(); i++) {
        result[i] = intermediate[i];
    }

    return result;
}

int main(int argc, char ** argv) {
     int n = 100;
    if (argc == 2) {
      n = atoi(argv[1]);
      printf("%d\n", n);
    }
    std::string command = "python ./generateRandomQP.py " + std::to_string(n);
    std::array<char, 128> buffer;
    std::string output;
    std::shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        output += buffer.data();
    }

    // parse the output
    std::stringstream ss(output);
    std::string part;
    std::vector<double*> all_parts;
    std::vector<size_t> sizes;
    size_t size;

    while (getline(ss, part, ';')) {
        all_parts.push_back(parse_doubles_from_string(part, size));
        sizes.push_back(size);
    }

    double* Q_arr = all_parts[0];
    std::cout << "Q size: " << sizes[0] << std::endl;
    matrix Q(n, n, Q_arr);
    
    double* c_arr = all_parts[1];
    std::cout << "c size: " << sizes[1] << std::endl;
    vector c(n, c_arr);

    double* J_arr = all_parts[2];
    std::cout << "J size: " << sizes[2] << std::endl;
    matrix J(2 * n, n, J_arr);

    double* g_arr = all_parts[3];
    std::cout << "g size: " << sizes[3] << std::endl;
    vector g(2 * n, g_arr);

    ipm(Q, c, J, g, 100);

    for (double* ptr : all_parts) {
        delete[] ptr;
    }
}
