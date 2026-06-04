#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include "utils.h"
#include "multigrid.h"
#include "smoother.h"

// Standalone wrapper to test convergence speed of different solvers
void benchmark_solver(int N, std::string name, int mode) {
    // mode 0: Baseline SOR
    // mode 1: V-cycle MG
    // mode 2: W-cycle MG
    double h2 = 1.0 / ((N - 1) * (N - 1));
    double w = 1.0; 
    std::vector<double> u(N * N, 0.0);
    std::vector<double> phi = get_phi_exact_solution(N);
    std::vector<double> u_exact = get_exact_solution(N);

    double start_time = omp_get_wtime();
    int iters = 0;
    
    // Iterate until the L1 error is sufficiently small
    while(true) {
        if (mode == 0) {
            SOR_smooth(N, u, phi, h2, w, 100);
            iters += 100;
        } else if (mode == 1) {
            v_cycle(u, phi, N, 2, 2, w);
            iters += 1;
        } else {
            w_cycle(u, phi, N, 2, 2, w);
            iters += 1;
        }
        
        double err = get_error(u, u_exact, N);
        if (err < 1e-4) break;
        if (iters > 50000) {
            std::cout << "Did not converge..." << std::endl;
            break; 
        }
    }
    double end_time = omp_get_wtime();
    
    std::cout << std::left << std::setw(15) << name 
              << " | N=" << std::setw(4) << N 
              << " | Iters: " << std::setw(6) << iters 
              << " | Time: " << std::fixed << std::setprecision(4) << (end_time - start_time) << "s"
              << " | Final Err: " << get_error(u, u_exact, N) << std::endl;
}

int main(int argc, char** argv) {
    int threads = 1;
    if (argc > 1) threads = std::atoi(argv[1]);
    omp_set_num_threads(threads);
    std::cout << "========================================\n";
    std::cout << "Running Benchmark with " << threads << " Threads\n";
    std::cout << "========================================\n";

    std::vector<int> Ns = {33, 65, 129, 257};
    for (int N : Ns) {
        benchmark_solver(N, "Baseline SOR", 0);
        benchmark_solver(N, "V-Cycle MG", 1);
        benchmark_solver(N, "W-Cycle MG", 2);
        std::cout << "---------------------------------------------------------\n";
    }
    return 0;
}
