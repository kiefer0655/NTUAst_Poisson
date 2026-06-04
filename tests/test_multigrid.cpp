#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../include/utils.h"
#include "../include/multigrid.h"

int main() {
    int N = 65; // Multigrid usually uses N = 2^k + 1
    double h2 = 1.0 / ((N - 1) * (N - 1));
    double w = 1.0; 
    int max_iters = 10; // Multigrid converges extremely fast compared to SOR!

    std::vector<double> u_v(N * N, 0.0);
    std::vector<double> u_w(N * N, 0.0);
    std::vector<double> phi = get_phi_exact_solution(N);
    std::vector<double> u_exact = get_exact_solution(N);

    std::cout << "Testing V-Cycle Integration..." << std::endl;
    for(int iter = 0; iter < max_iters; iter++) {
        v_cycle(u_v, phi, N, 2, 2, w);
    }
    double err_v = get_error(u_v, u_exact, N);
    std::cout << "L1 Error after " << max_iters << " V-cycles: " << err_v << std::endl;
    assert(err_v < 1e-2 && "V-Cycle did not converge.");

    std::cout << "\nTesting W-Cycle Integration..." << std::endl;
    for(int iter = 0; iter < max_iters; iter++) {
        w_cycle(u_w, phi, N, 2, 2, w);
    }
    double err_w = get_error(u_w, u_exact, N);
    std::cout << "L1 Error after " << max_iters << " W-cycles: " << err_w << std::endl;
    assert(err_w < 1e-2 && "W-Cycle did not converge.");

    std::cout << "\nMultigrid Integration Phase 2 Complete & Successful!" << std::endl;
    return 0;
}
