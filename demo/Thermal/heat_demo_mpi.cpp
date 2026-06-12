// heat_demo_mpi.cpp
// Usage: mpirun -np 4 ./heat_demo_mpi

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <chrono>
#include <functional>
#include <mpi.h>

#include "../../include/utils.h"
#include "../../include/multigrid_mpi.h"

using Clock = std::chrono::high_resolution_clock;
double elapsed_ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// ============================================================
// Domain decomposition helper
// ============================================================

// Returns how many rows this rank owns (excluding ghost rows)
// MPI solver expects u_local to include 1 ghost row on each side,
// so u_local has (local_rows + 2) * N entries.
int get_local_rows(int N, int rank, int P) {
    int interior = N - 2;           // rows 1..N-2
    int base = interior / P;
    int rem  = interior % P;
    return base + (rank < rem ? 1 : 0);
}

int get_row_start(int N, int rank, int P) {
    int interior = N - 2;
    int base = interior / P;
    int rem  = interior % P;
    // sum of local_rows for ranks 0..rank-1
    return 1 + rank * base + std::min(rank, rem);
}

// ============================================================
// CSV output (rank 0 only, receives full grid via Gatherv)
// ============================================================

void write_field_csv(const std::string& filename,
                     const std::vector<double>& u, int N) {
    std::ofstream f(filename);
    if (!f.is_open()) { std::cerr << "Cannot open " << filename << "\n"; return; }
    f << "i,j,x,y,u\n";
    double h = 1.0 / (N - 1);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            f << i << "," << j << ","
              << i*h << "," << j*h << ","
              << u[i*N+j] << "\n";
    std::cout << "  Saved " << filename << "\n";
}

// Gather u_local (interior rows only) back to rank 0 as full N*N array
std::vector<double> gather_full(const std::vector<double>& u_local,
                                int N, int local_rows, int rank, int P) {
    // u_local layout: (local_rows + 2) rows x N cols, rows 0 and local_rows+1 are ghosts
    // Extract interior rows
    std::vector<double> my_interior(local_rows * N);
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < N; j++)
            my_interior[i*N+j] = u_local[(i+1)*N+j];  // skip ghost row 0

    // Gatherv
    std::vector<int> recvcounts(P), displs(P);
    for (int r = 0; r < P; r++) {
        recvcounts[r] = get_local_rows(N, r, P) * N;
        displs[r] = (r == 0) ? 0 : displs[r-1] + recvcounts[r-1];
    }

    std::vector<double> full_interior;
    if (rank == 0) full_interior.resize((N-2) * N);

    MPI_Gatherv(my_interior.data(), local_rows * N, MPI_DOUBLE,
                full_interior.data(), recvcounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rank 0 reconstructs full N*N array with boundaries
    std::vector<double> full;
    if (rank == 0) {
        full.resize(N * N, 0.0);
        // Copy interior rows
        for (int i = 1; i < N-1; i++)
            for (int j = 0; j < N; j++)
                full[i*N+j] = full_interior[(i-1)*N+j];
        // Boundaries remain 0 (Dirichlet)
    }
    return full;
}

// ============================================================
// Generic scenario runner
// ============================================================

void run_scenario(const std::string& name,
                  const std::string& outfile,
                  int N, int rank, int P,
                  std::function<double(double,double)> source_fn) {

    int local_rows = get_local_rows(N, rank, P);
    int row_start  = get_row_start(N, rank, P);
    double h = 1.0 / (N - 1);

    // u_local and rho_local include ghost rows: size = (local_rows+2)*N
    std::vector<double> u_local((local_rows + 2) * N, 0.0);
    std::vector<double> rho_local((local_rows + 2) * N, 0.0);

    // Fill rho for interior rows (index 1..local_rows in local array)
    auto t0 = Clock::now();
    for (int li = 0; li < local_rows; li++) {
        int gi = row_start + li;          // global row index
        for (int j = 1; j < N-1; j++) {
            double x = gi * h, y = j * h;
            rho_local[(li+1)*N + j] = source_fn(x, y);
        }
    }
    double t_deposit = elapsed_ms(t0);

    // Solve
    auto t1 = Clock::now();
    fmg_cycle_mpi(u_local, rho_local, N, rank, P, 2, 2, 1.0);
    double t_solve = elapsed_ms(t1);

    if (rank == 0)
        std::cout << "  [" << name << "]"
                  << "  deposit=" << t_deposit << "ms"
                  << "  solve="   << t_solve   << "ms\n";

    // Gather and write
    auto full = gather_full(u_local, N, local_rows, rank, P);
    if (rank == 0) write_field_csv(outfile, full, N);
}

// ============================================================
// main
// ============================================================

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    const int N = 129;

    if (rank == 0) {
        std::cout << "=== Heat Conduction Demo (MPI) ===\n";
        std::cout << "Grid: " << N << " x " << N
                  << "  |  Processes: " << P << "\n\n";
    }

    // Scenario 1: Single central source
    if (rank == 0) std::cout << "[1/4] Single central source\n";
    run_scenario("Single Source", "results/heat_single_source.csv", N, rank, P,
        [](double x, double y) {
            double cx=0.5, cy=0.5, sigma=0.05;
            double r2 = (x-cx)*(x-cx)+(y-cy)*(y-cy);
            return -10.0 * std::exp(-r2/(2.0*sigma*sigma));
        });

    // Scenario 2: Multi source/sink
    if (rank == 0) std::cout << "[2/4] Multiple sources and sinks\n";
    run_scenario("Multi Source", "results/heat_multi_source.csv", N, rank, P,
        [](double x, double y) {
            double sx[]={0.25,0.75,0.50,0.75,0.25};
            double sy[]={0.25,0.75,0.50,0.25,0.75};
            double ss[]={ 8.0, 8.0, 6.0,-6.0,-6.0};
            double sigma=0.04, val=0.0;
            for (int k=0;k<5;k++){
                double r2=(x-sx[k])*(x-sx[k])+(y-sy[k])*(y-sy[k]);
                val += -ss[k]*std::exp(-r2/(2.0*sigma*sigma));
            }
            return val;
        });

    // Scenario 3: Boundary gradient (pure Laplace, rho=0)
    if (rank == 0) std::cout << "[3/4] Boundary temperature gradient\n";
    {
        int local_rows = get_local_rows(N, rank, P);
        int row_start  = get_row_start(N, rank, P);
        double h = 1.0/(N-1);

        std::vector<double> u_local((local_rows+2)*N, 0.0);
        std::vector<double> rho_local((local_rows+2)*N, 0.0);

        // Set boundary values in u_local for rows that touch boundary
        for (int li = 0; li < local_rows; li++) {
            int gi = row_start + li;
            double frac = (double)gi/(N-1);
            u_local[(li+1)*N + 0]   = 1.0 - frac;   // left wall
            u_local[(li+1)*N + N-1] = 1.0 - frac;   // right wall
        }
        // Top/bottom boundary rows only exist on rank 0 and rank P-1
        if (rank == 0)
            for (int j=0;j<N;j++) u_local[0*N+j]           = 1.0; // ghost=boundary
        if (rank == P-1)
            for (int j=0;j<N;j++) u_local[(local_rows+1)*N+j] = 0.0;

        auto t1 = Clock::now();
        for (int k = 0; k < 20; k++) {
            v_cycle_mpi(u_local, rho_local, N, rank, P, 3, 3, 1.0);
            // Re-enforce side walls
            for (int li = 0; li < local_rows; li++) {
                int gi = row_start + li;
                double frac = (double)gi/(N-1);
                u_local[(li+1)*N + 0]   = 1.0 - frac;
                u_local[(li+1)*N + N-1] = 1.0 - frac;
            }
        }
        if (rank == 0)
            std::cout << "  solve=" << elapsed_ms(t1) << "ms (20 V-cycles)\n";

        auto full = gather_full(u_local, N, local_rows, rank, P);
        if (rank == 0) write_field_csv("results/heat_boundary_gradient.csv", full, N);
    }

    // Scenario 4: Ring source
    if (rank == 0) std::cout << "[4/4] Ring-shaped source\n";
    run_scenario("Ring Source", "results/heat_ring_source.csv", N, rank, P,
        [](double x, double y) {
            double cx=0.5, cy=0.5, ring_r=0.25, sigma=0.03;
            double dist=std::abs(std::sqrt((x-cx)*(x-cx)+(y-cy)*(y-cy))-ring_r);
            return -12.0 * std::exp(-(dist*dist)/(2.0*sigma*sigma));
        });

    if (rank == 0)
        std::cout << "\nAll done. Run plot_heat.py to visualize.\n";

    MPI_Finalize();
    return 0;
}