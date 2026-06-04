#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include "../include/utils.h"
#include "../include/multigrid.h"
#include "../include/multigrid_cuda.cuh"

int main() {
    std::cout << "========================================\n";
    std::cout << "Running CPU vs GPU Benchmark\n";
    std::cout << "========================================\n";

    std::vector<int> Ns = {129, 257, 513, 1025}; // Push the grid size up for GPU
    int iters = 10;
    double w = 1.0;

    for (int N : Ns) {
        std::vector<double> u_cpu(N * N, 0.0);
        std::vector<double> u_gpu(N * N, 0.0);
        std::vector<double> phi = get_phi_exact_solution(N);
        std::vector<double> u_exact = get_exact_solution(N);

        // CPU Benchmark
        auto start_cpu = std::chrono::high_resolution_clock::now();
        for(int i=0; i<iters; i++) v_cycle(u_cpu, phi, N, 2, 2, w);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;

        // GPU Benchmark
        auto start_gpu = std::chrono::high_resolution_clock::now();
        v_cycle_cuda_run(u_gpu, phi, N, 2, 2, w, iters);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_gpu = end_gpu - start_gpu;

        double err_cpu = get_error(u_cpu, u_exact, N);
        double err_gpu = get_error(u_gpu, u_exact, N);

        std::cout << "Grid N=" << std::setw(4) << N << " | Iterations: " << iters << "\n";
        std::cout << "  CPU V-Cycle Time: " << std::fixed << std::setprecision(4) << diff_cpu.count() << "s | Err: " << err_cpu << "\n";
        std::cout << "  GPU V-Cycle Time: " << std::fixed << std::setprecision(4) << diff_gpu.count() << "s | Err: " << err_gpu << "\n";
        std::cout << "  GPU Speedup:      " << std::fixed << std::setprecision(2) << (diff_cpu.count() / diff_gpu.count()) << "x\n";
        std::cout << "---------------------------------------------------------\n";
    }

    return 0;
}
