#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <omp.h>

#include "../../include/utils.h"
#include "../../include/multigrid.h"

// ============================================================
// Grid helpers
// ============================================================

void zero_boundary(std::vector<double>& u, int N) {
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < N; k++) {
        u[idx(0,     k, N)] = 0.0;
        u[idx(N-1,   k, N)] = 0.0;
        u[idx(k,     0, N)] = 0.0;
        u[idx(k,   N-1, N)] = 0.0;
    }
}

void set_boundary_gradient(std::vector<double>& u, int N, double T_left, double T_right) {
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < N; k++) {
        double frac = (double)k / (N - 1);
        u[idx(k,   0, N)] = T_left + (T_right - T_left) * frac;
        u[idx(k, N-1, N)] = T_left + (T_right - T_left) * frac;
        u[idx(0,   k, N)] = T_left;
        u[idx(N-1, k, N)] = T_right;
    }
}

// ============================================================
// Write CSV
// ============================================================

void write_field_csv(const std::string& filename,
                     const std::vector<double>& u,
                     int N) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << filename << "\n";
        return;
    }
    f << "i,j,x,y,u\n";
    double h = 1.0 / (N - 1);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            f << i << "," << j << ","
              << i * h << "," << j * h << ","
              << u[idx(i, j, N)] << "\n";
        }
    }
    std::cout << "  Saved " << filename << "\n";
}

// ============================================================
// Timing helper
// ============================================================

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// ============================================================
// Scenario 1: Single central heat source
// ============================================================

void scenario_single_source(int N) {
    double h = 1.0 / (N - 1);
    std::vector<double> rho(N * N, 0.0);
    std::vector<double> u(N * N, 0.0);

    double cx = 0.5, cy = 0.5, sigma = 0.05;

    // --- OpenMP: parallel rho deposit ---
    auto t0 = Clock::now();
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            double x = i * h, y = j * h;
            double r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy);
            rho[idx(i, j, N)] = -10.0 * std::exp(-r2 / (2.0*sigma*sigma));
        }
    }
    double t_deposit = elapsed_ms(t0);

    zero_boundary(u, N);

    // --- fmg_cycle already uses OpenMP internally ---
    auto t1 = Clock::now();
    fmg_cycle(u, rho, N, 2, 2, 1.0);
    double t_solve = elapsed_ms(t1);

    std::cout << "  deposit=" << t_deposit << "ms  solve=" << t_solve << "ms\n";
    write_field_csv("results/heat_single_source.csv", u, N);
}

// ============================================================
// Scenario 2: Multiple heat sources and cold sinks
// ============================================================

void scenario_multi_source(int N) {
    double h = 1.0 / (N - 1);
    std::vector<double> rho(N * N, 0.0);
    std::vector<double> u(N * N, 0.0);

    struct SourcePoint { double x, y, strength; };
    std::vector<SourcePoint> sources = {
        {0.25, 0.25,  8.0},
        {0.75, 0.75,  8.0},
        {0.50, 0.50,  6.0},
        {0.75, 0.25, -6.0},
        {0.25, 0.75, -6.0},
    };
    double sigma = 0.04;

    // --- OpenMP: each (i,j) is independent, safe to parallelize ---
    auto t0 = Clock::now();
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            double x = i * h, y = j * h;
            double val = 0.0;
            for (const auto& s : sources) {
                double r2 = (x-s.x)*(x-s.x) + (y-s.y)*(y-s.y);
                val += -s.strength * std::exp(-r2 / (2.0*sigma*sigma));
            }
            rho[idx(i, j, N)] = val;
        }
    }
    double t_deposit = elapsed_ms(t0);

    zero_boundary(u, N);

    auto t1 = Clock::now();
    fmg_cycle(u, rho, N, 2, 2, 1.0);
    double t_solve = elapsed_ms(t1);

    std::cout << "  deposit=" << t_deposit << "ms  solve=" << t_solve << "ms\n";
    write_field_csv("results/heat_multi_source.csv", u, N);
}

// ============================================================
// Scenario 3: Boundary temperature gradient (pure Laplace)
// ============================================================

void scenario_boundary_gradient(int N) {
    std::vector<double> rho(N * N, 0.0);
    std::vector<double> u(N * N, 0.0);

    set_boundary_gradient(u, N, 1.0, 0.0);

    // V-cycles with boundary re-enforcement
    // (fmg zeros boundaries internally, so we iterate manually)
    auto t0 = Clock::now();
    for (int k = 0; k < 20; k++) {
        v_cycle(u, rho, N, 3, 3, 1.0);
        set_boundary_gradient(u, N, 1.0, 0.0);
    }
    double t_solve = elapsed_ms(t0);

    std::cout << "  solve=" << t_solve << "ms (20 V-cycles)\n";
    write_field_csv("results/heat_boundary_gradient.csv", u, N);
}

// ============================================================
// Scenario 4: Ring-shaped heat source
// ============================================================

void scenario_ring_source(int N) {
    double h = 1.0 / (N - 1);
    std::vector<double> rho(N * N, 0.0);
    std::vector<double> u(N * N, 0.0);

    double cx = 0.5, cy = 0.5;
    double ring_r = 0.25, sigma = 0.03;

    // --- OpenMP: embarrassingly parallel, no dependencies ---
    auto t0 = Clock::now();
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            double x = i * h, y = j * h;
            double dist = std::abs(
                std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)) - ring_r
            );
            rho[idx(i, j, N)] = -12.0 * std::exp(-(dist*dist) / (2.0*sigma*sigma));
        }
    }
    double t_deposit = elapsed_ms(t0);

    zero_boundary(u, N);

    auto t1 = Clock::now();
    fmg_cycle(u, rho, N, 2, 2, 1.0);
    double t_solve = elapsed_ms(t1);

    std::cout << "  deposit=" << t_deposit << "ms  solve=" << t_solve << "ms\n";
    write_field_csv("results/heat_ring_source.csv", u, N);
}

// ============================================================
// main
// ============================================================

int main(int argc, char* argv[]) {
    const int N = 129;

    // Allow thread count override: ./heat_demo 8
    int nthreads = omp_get_max_threads();
    if (argc > 1) nthreads = std::atoi(argv[1]);
    omp_set_num_threads(nthreads);

    std::cout << "=== Heat Conduction Demo (OpenMP) ===\n";
    std::cout << "Grid: " << N << " x " << N
              << "  |  Threads: " << nthreads << "\n\n";

    std::cout << "[1/4] Single central source\n";
    scenario_single_source(N);

    std::cout << "[2/4] Multiple sources and sinks\n";
    scenario_multi_source(N);

    std::cout << "[3/4] Boundary temperature gradient\n";
    scenario_boundary_gradient(N);

    std::cout << "[4/4] Ring-shaped source\n";
    scenario_ring_source(N);

    std::cout << "\nAll done. Run plot_heat.py to visualize.\n";
    return 0;
}