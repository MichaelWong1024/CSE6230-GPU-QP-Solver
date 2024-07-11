
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <vector>
#include <sstream>
#include <array>
#include <memory>
#include "cublas_v2.h"
#include "mkl.h"
#include "blas_actors.h"
#include "support.h" 

#define STEP 0.25
#define IPM_ERR 1
#define IPM_SUCCESS 0


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
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, SIZE_MAX);
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

    for (double* ptr : all_parts) {
        delete[] ptr;
    }
}
