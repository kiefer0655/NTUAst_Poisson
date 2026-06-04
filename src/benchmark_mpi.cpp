#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <mpi.h>
#include "../include/utils.h"
#include "../include/multigrid.h"
#include "../include/multigrid_mpi.h"
#include "../include/smoother.h"

struct BenchmarkResult {
    int N;
    std::string method;
    std::string hardware;
    int iters;
    double time_s;
    double err;
};

void run_benchmark() {
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (rank == 0) {
        std::cout << "========================================\n";
        std::cout << "Running MPI Benchmark Suite (P=" << P << ")\n";
        std::cout << "========================================\n";
    }

    std::vector<int> Ns = {33, 65, 129, 257, 513, 1025, 2049, 4097, 8193, 16385, 32769};
    std::vector<BenchmarkResult> results;
    double w = 1.0;
    
    for (int N : Ns) {
        if (rank == 0) std::cout << "Benchmarking Grid N=" << N << "...\n";
        
        int K = (N - 1) / P;
        int local_rows = (rank == P - 1) ? (N - rank * K) : K;
        
        std::vector<int> sendcounts(P);
        std::vector<int> displs(P);
        for (int p = 0; p < P; p++) {
            int lr = (p == P - 1) ? (N - p * K) : K;
            sendcounts[p] = lr * N;
            displs[p] = (p * K) * N;
        }

        std::vector<double> u_exact_global;
        std::vector<double> phi_global;
        if (rank == 0) {
            u_exact_global = get_exact_solution(N);
            phi_global = get_phi_exact_solution(N);
        }

        std::vector<double> phi_local((local_rows + 2) * N, 0.0);
        MPI_Scatterv(rank == 0 ? phi_global.data() : nullptr, sendcounts.data(), displs.data(), 
                     MPI_DOUBLE, &phi_local[1 * N], local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                     
        exchange_ghost_cells(phi_local, N, local_rows, rank, P);

        auto benchmark_cycle = [&](std::string method, int iters, int cycle_type) {
            std::vector<double> u_local((local_rows + 2) * N, 0.0);
            std::vector<double> u_global;
            if (rank == 0) u_global.resize(N * N, 0.0);

            MPI_Barrier(MPI_COMM_WORLD);
            auto t1 = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < iters; i++) {
                if (cycle_type == 1) v_cycle_mpi(u_local, phi_local, N, rank, P, 2, 2, w);
                else if (cycle_type == 2) w_cycle_mpi(u_local, phi_local, N, rank, P, 2, 2, w);
                else if (cycle_type == 3) fmg_cycle_mpi(u_local, phi_local, N, rank, P, 2, 2, w);
                else {
                    double h2 = 1.0 / ((N - 1) * (N - 1));
                    SOR_smooth_mpi(N, local_rows, u_local, phi_local, h2, w, 1, rank, P);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto t2 = std::chrono::high_resolution_clock::now();
            double time_s = std::chrono::duration<double>(t2 - t1).count();

            MPI_Gatherv(&u_local[1 * N], local_rows * N, MPI_DOUBLE, 
                        rank == 0 ? u_global.data() : nullptr, sendcounts.data(), displs.data(), 
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                double err = get_error(u_global, u_exact_global, N);
                std::string hw = "MPI-" + std::to_string(P) + "P";
                results.push_back({N, method, hw, iters, time_s, err});
            }
        };

        // V-Cycle
        benchmark_cycle("V-Cycle", (N <= 129) ? 10 : 5, 1);
        
        // W-Cycle
        if (N <= 513) benchmark_cycle("W-Cycle", (N <= 129) ? 5 : 2, 2);
        
        // FMG
        benchmark_cycle("FMG", 1, 3);
        
        // SOR
        if (N <= 129) {
            int sor_iters = (N == 33) ? 1000 : (N == 65 ? 5000 : 20000);
            benchmark_cycle("SOR", sor_iters, 4);
        }
    }

    if (rank == 0) {
        // Use append mode so run_benchmark_mpi.sh can loop through P
        bool file_exists = false;
        std::ifstream infile("results/results_mpi.csv");
        if (infile.good()) file_exists = true;
        infile.close();

        std::ofstream file("results/results_mpi.csv", std::ios_base::app);
        if (!file_exists) file << "N,Method,Hardware,Iterations,Time_s,Error\n";
        
        for (const auto& r : results) {
            file << r.N << "," << r.method << "," << r.hardware << "," << r.iters << "," 
                 << std::fixed << std::setprecision(6) << r.time_s << "," 
                 << std::scientific << std::setprecision(4) << r.err << "\n";
        }
        file.close();
        std::cout << "Data appended to results/results_mpi.csv\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    run_benchmark();
    MPI_Finalize();
    return 0;
}
