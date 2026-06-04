#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <omp.h>
#include "../include/utils.h"
#include "../include/multigrid.h"
#include "../include/smoother.h"

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
    std::cout << "Running CPU Benchmark Suite\n";
    std::cout << "========================================\n";

    std::vector<int> Ns = {33, 65, 129, 257, 513, 1025, 2049, 4097, 8193, 16385, 32769};
    std::vector<BenchmarkResult> results;
    
    double w = 1.0;
    
    for (int run = 0; run < 3; run++) {
        for (int N : Ns) {
        std::cout << "Benchmarking Grid N=" << N << "...\n";
        std::vector<double> u_exact = get_exact_solution(N);
        std::vector<double> phi = get_phi_exact_solution(N);

        // ========================== CPU V-Cycle ==========================
        int v_iters_cpu = (N <= 129) ? 10 : 5;
        std::vector<double> u_vc(N * N, 0.0);
        auto t1 = std::chrono::high_resolution_clock::now();
        for(int i=0; i<v_iters_cpu; i++) v_cycle(u_vc, phi, N, 2, 2, w);
        auto t2 = std::chrono::high_resolution_clock::now();
        double time_vc = std::chrono::duration<double>(t2 - t1).count();
        results.push_back({N, "V-Cycle", "CPU", v_iters_cpu, time_vc, get_error(u_vc, u_exact, N)});

        // ========================== CPU W-Cycle ==========================
        if (N <= 8193) {
            int w_iters = (N <= 129) ? 5 : 2;
            std::vector<double> u_wc(N * N, 0.0);
            t1 = std::chrono::high_resolution_clock::now();
            for(int i=0; i<w_iters; i++) w_cycle(u_wc, phi, N, 2, 2, w);
            t2 = std::chrono::high_resolution_clock::now();
            double time_wc = std::chrono::duration<double>(t2 - t1).count();
            results.push_back({N, "W-Cycle", "CPU", w_iters, time_wc, get_error(u_wc, u_exact, N)});
        }

        // ========================== CPU FMG ==========================
        std::vector<double> u_fc(N * N, 0.0);
        t1 = std::chrono::high_resolution_clock::now();
        fmg_cycle(u_fc, phi, N, 2, 2, w); 
        t2 = std::chrono::high_resolution_clock::now();
        double time_fc = std::chrono::duration<double>(t2 - t1).count();
        results.push_back({N, "FMG", "CPU", 1, time_fc, get_error(u_fc, u_exact, N)});

        // ========================== CPU SOR (Baseline) ==========================
        if (N <= 2049) {
            std::vector<double> u_sor(N * N, 0.0);
            int sor_iters = (N == 33) ? 1000 : (N == 65 ? 5000 : 20000);
            double h2 = 1.0 / ((N - 1) * (N - 1));
            t1 = std::chrono::high_resolution_clock::now();
            SOR_smooth(N, u_sor, phi, h2, w, sor_iters);
            t2 = std::chrono::high_resolution_clock::now();
            double time_sor = std::chrono::duration<double>(t2 - t1).count();
            results.push_back({N, "SOR", "CPU", sor_iters, time_sor, get_error(u_sor, u_exact, N)});
        }
        }
    }

    std::ofstream file("results/results_cpu.csv");
    file << "N,Method,Hardware,Iterations,Time_s,Error\n";
    for (const auto& r : results) {
        file << r.N << "," << r.method << "," << r.hardware << "," << r.iters << "," 
             << std::fixed << std::setprecision(6) << r.time_s << "," 
             << std::scientific << std::setprecision(4) << r.err << "\n";
    }
    file.close();

    std::cout << "Benchmarking Complete! Data saved to results/results_cpu.csv\n";

    // =========================================================================
    // Strong Scaling Benchmark
    // =========================================================================
    std::cout << "\n========================================\n";
    std::cout << "Running Strong Scaling Benchmark (N=2049)\n";
    std::cout << "========================================\n";
    
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 32, 64, 144};
    int scale_N = 2049;
    std::vector<double> u_exact_scale = get_exact_solution(scale_N);
    std::vector<double> phi_scale = get_phi_exact_solution(scale_N);
    std::vector<BenchmarkResult> scale_results;

    for (int threads : thread_counts) {
        std::cout << "Benchmarking with " << threads << " Threads...\n";
        omp_set_num_threads(threads);
        
        // Measure V-Cycle
        std::vector<double> u_vc(scale_N * scale_N, 0.0);
        auto t1 = std::chrono::high_resolution_clock::now();
        for(int i=0; i<5; i++) v_cycle(u_vc, phi_scale, scale_N, 2, 2, w);
        auto t2 = std::chrono::high_resolution_clock::now();
        double time_vc = std::chrono::duration<double>(t2 - t1).count();
        scale_results.push_back({scale_N, "V-Cycle", "CPU-" + std::to_string(threads) + "T", 5, time_vc, get_error(u_vc, u_exact_scale, scale_N)});

        // Measure FMG
        std::vector<double> u_fc(scale_N * scale_N, 0.0);
        t1 = std::chrono::high_resolution_clock::now();
        fmg_cycle(u_fc, phi_scale, scale_N, 2, 2, w);
        t2 = std::chrono::high_resolution_clock::now();
        double time_fc = std::chrono::duration<double>(t2 - t1).count();
        scale_results.push_back({scale_N, "FMG", "CPU-" + std::to_string(threads) + "T", 1, time_fc, get_error(u_fc, u_exact_scale, scale_N)});
    }

    std::ofstream scale_file("results/scaling_cpu.csv");
    scale_file << "N,Method,Hardware,Iterations,Time_s,Error\n";
    for (const auto& r : scale_results) {
        scale_file << r.N << "," << r.method << "," << r.hardware << "," << r.iters << "," 
             << std::fixed << std::setprecision(6) << r.time_s << "," 
             << std::scientific << std::setprecision(4) << r.err << "\n";
    }
    scale_file.close();
    std::cout << "Scaling Benchmark Complete! Data saved to results/scaling_cpu.csv\n";

    return 0;
}
