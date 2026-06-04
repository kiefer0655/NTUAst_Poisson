#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <fstream>
#include "../include/utils.h"
#include "../include/multigrid_cuda.cuh"

struct BenchmarkResult {
    int N;
    std::string method;
    std::string hardware;
    int iters;
    double time_s;
    double err;
};

int main() {
    std::cout << "========================================\n";
    std::cout << "Running GPU Benchmark Suite\n";
    std::cout << "========================================\n";

    std::vector<int> Ns = {33, 65, 129, 257, 513, 1025, 2049, 4097, 8193, 16385, 32769};
    std::vector<BenchmarkResult> results;
    
    double w = 1.0;
    
    for (int run = 0; run < 3; run++) {
        for (int N : Ns) {
        std::cout << "Benchmarking Grid N=" << N << "...\n";
        std::vector<double> u_exact = get_exact_solution(N);
        std::vector<double> phi = get_phi_exact_solution(N);

        // ========================== GPU V-Cycle ==========================
        int v_iters_gpu = (N <= 129) ? 10 : 5;


        std::vector<double> u_vg(N * N, 0.0);
        auto t1 = std::chrono::high_resolution_clock::now();
        v_cycle_cuda_run(u_vg, phi, N, 2, 2, w, v_iters_gpu);
        auto t2 = std::chrono::high_resolution_clock::now();
        double time_vg = std::chrono::duration<double>(t2 - t1).count();
        results.push_back({N, "V-Cycle", "GPU", v_iters_gpu, time_vg, get_error(u_vg, u_exact, N)});

        // ========================== GPU FMG ==========================
        std::vector<double> u_fg(N * N, 0.0);
        t1 = std::chrono::high_resolution_clock::now();
        fmg_cycle_cuda_run(u_fg, phi, N, 2, 2, w); // FMG 1 iteration
        t2 = std::chrono::high_resolution_clock::now();
        double time_fg = std::chrono::duration<double>(t2 - t1).count();
        results.push_back({N, "FMG", "GPU", 1, time_fg, get_error(u_fg, u_exact, N)});
    }
    }

    std::ofstream file("results/results_gpu.csv");
    file << "N,Method,Hardware,Iterations,Time_s,Error\n";
    for (const auto& r : results) {
        file << r.N << "," << r.method << "," << r.hardware << "," << r.iters << "," 
             << std::fixed << std::setprecision(6) << r.time_s << "," 
             << std::scientific << std::setprecision(4) << r.err << "\n";
    }
    file.close();

    std::cout << "Benchmarking Complete! Data saved to results/results_gpu.csv\n";
    return 0;
}
