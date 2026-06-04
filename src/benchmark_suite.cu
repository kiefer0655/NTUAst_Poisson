#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <fstream>
#include "../include/utils.h"
#include "../include/multigrid.h"
#include "../include/multigrid_cuda.cuh"
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
    std::cout << "Running Comprehensive Final Benchmark\n";
    std::cout << "========================================\n";

    std::vector<int> Ns = {33, 65, 129, 257, 513, 1025, 2049};
    std::vector<BenchmarkResult> results;
    
    double w = 1.0;
    
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

        // ========================== GPU V-Cycle ==========================
        int v_iters_gpu = (N <= 129) ? 10 : 5;
        std::vector<double> u_vg(N * N, 0.0);
        t1 = std::chrono::high_resolution_clock::now();
        v_cycle_cuda_run(u_vg, phi, N, 2, 2, w, v_iters_gpu);
        t2 = std::chrono::high_resolution_clock::now();
        double time_vg = std::chrono::duration<double>(t2 - t1).count();
        results.push_back({N, "V-Cycle", "GPU", v_iters_gpu, time_vg, get_error(u_vg, u_exact, N)});

        // ========================== CPU W-Cycle ==========================
        if (N <= 513) {
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
        fmg_cycle(u_fc, phi, N, 2, 2, w); // FMG only needs 1 iteration usually
        t2 = std::chrono::high_resolution_clock::now();
        double time_fc = std::chrono::duration<double>(t2 - t1).count();
        results.push_back({N, "FMG", "CPU", 1, time_fc, get_error(u_fc, u_exact, N)});

        // ========================== GPU FMG ==========================
        std::vector<double> u_fg(N * N, 0.0);
        t1 = std::chrono::high_resolution_clock::now();
        fmg_cycle_cuda_run(u_fg, phi, N, 2, 2, w); // FMG 1 iteration
        t2 = std::chrono::high_resolution_clock::now();
        double time_fg = std::chrono::duration<double>(t2 - t1).count();
        results.push_back({N, "FMG", "GPU", 1, time_fg, get_error(u_fg, u_exact, N)});

        // ========================== CPU SOR (Baseline) ==========================
        if (N <= 129) {
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

    std::ofstream file("results.csv");
    file << "N,Method,Hardware,Iterations,Time_s,Error\n";
    for (const auto& r : results) {
        file << r.N << "," << r.method << "," << r.hardware << "," << r.iters << "," 
             << std::fixed << std::setprecision(6) << r.time_s << "," 
             << std::scientific << std::setprecision(4) << r.err << "\n";
    }
    file.close();

    std::cout << "Benchmarking Complete! Data saved to results.csv\n";
    return 0;
}
