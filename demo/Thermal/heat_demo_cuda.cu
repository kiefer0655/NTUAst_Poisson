// heat_demo_cuda.cu
// Usage: ./heat_demo_cuda

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

#include "../../include/utils.h"
#include "../../include/multigrid_cuda.cuh"

using Clock = std::chrono::high_resolution_clock;
double elapsed_ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// ============================================================
// CUDA kernels: rho deposit
// ============================================================

__global__ void kernel_single_source(double* rho, int N, double h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= N-1 || j >= N-1) return;
    double cx=0.5, cy=0.5, sigma=0.05;
    double x=i*h, y=j*h;
    double r2=(x-cx)*(x-cx)+(y-cy)*(y-cy);
    rho[i*N+j] = -10.0 * exp(-r2/(2.0*sigma*sigma));
}

__global__ void kernel_multi_source(double* rho, int N, double h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= N-1 || j >= N-1) return;
    double x=i*h, y=j*h, sigma=0.04;
    double sx[]={0.25,0.75,0.50,0.75,0.25};
    double sy[]={0.25,0.75,0.50,0.25,0.75};
    double ss[]={ 8.0, 8.0, 6.0,-6.0,-6.0};
    double val=0.0;
    for (int k=0;k<5;k++){
        double r2=(x-sx[k])*(x-sx[k])+(y-sy[k])*(y-sy[k]);
        val += -ss[k]*exp(-r2/(2.0*sigma*sigma));
    }
    rho[i*N+j] = val;
}

__global__ void kernel_ring_source(double* rho, int N, double h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= N-1 || j >= N-1) return;
    double cx=0.5, cy=0.5, ring_r=0.25, sigma=0.03;
    double x=i*h, y=j*h;
    double dist=fabs(sqrt((x-cx)*(x-cx)+(y-cy)*(y-cy))-ring_r);
    rho[i*N+j] = -12.0 * exp(-(dist*dist)/(2.0*sigma*sigma));
}

// ============================================================
// CSV output
// ============================================================

void write_field_csv(const std::string& filename,
                     const std::vector<double>& u, int N) {
    std::ofstream f(filename);
    if (!f.is_open()) { std::cerr << "Cannot open " << filename << "\n"; return; }
    f << "i,j,x,y,u\n";
    double h = 1.0/(N-1);
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            f << i << "," << j << ","
              << i*h << "," << j*h << ","
              << u[i*N+j] << "\n";
    std::cout << "  Saved " << filename << "\n";
}

// ============================================================
// Generic GPU scenario runner
// Uses host wrapper fmg_cycle_cuda_run (manages GPU memory internally)
// ============================================================

void run_gpu_scenario(const std::string& name,
                      const std::string& outfile,
                      int N,
                      void(*kernel)(double*, int, double)) {
    double h = 1.0/(N-1);
    int total = N*N;

    // Deposit rho on GPU via kernel
    double* d_rho;
    cudaMalloc(&d_rho, total*sizeof(double));
    cudaMemset(d_rho, 0, total*sizeof(double));

    dim3 block(16,16);
    dim3 grid((N+15)/16, (N+15)/16);

    auto t0 = Clock::now();
    kernel<<<grid, block>>>(d_rho, N, h);
    cudaDeviceSynchronize();
    double t_deposit = elapsed_ms(t0);

    // Copy rho back to host for the host wrapper
    std::vector<double> rho(total);
    cudaMemcpy(rho.data(), d_rho, total*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_rho);

    // Solve: host wrapper handles GPU alloc/free internally
    std::vector<double> u(total, 0.0);
    auto t1 = Clock::now();
    fmg_cycle_cuda_run(u, rho, N, 2, 2, 1.0);
    double t_solve = elapsed_ms(t1);

    std::cout << "  [" << name << "]"
              << "  deposit=" << t_deposit << "ms"
              << "  solve="   << t_solve   << "ms\n";
    write_field_csv(outfile, u, N);
}

// ============================================================
// main
// ============================================================

int main() {
    const int N = 129;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "=== Heat Conduction Demo (CUDA) ===\n";
    std::cout << "Grid: " << N << " x " << N
              << "  |  GPU: " << prop.name << "\n\n";

    std::cout << "[1/4] Single central source\n";
    run_gpu_scenario("Single Source", "results/heat_single_source.csv",
                     N, kernel_single_source);

    std::cout << "[2/4] Multiple sources and sinks\n";
    run_gpu_scenario("Multi Source", "results/heat_multi_source.csv",
                     N, kernel_multi_source);

    // Scenario 3: Boundary gradient — rho=0, BC driven
    std::cout << "[3/4] Boundary temperature gradient\n";
    {
        std::vector<double> rho(N*N, 0.0);
        std::vector<double> u(N*N, 0.0);
        for (int k=0;k<N;k++) {
            double frac=(double)k/(N-1);
            u[idx(k,   0, N)] = 1.0-frac;
            u[idx(k, N-1, N)] = 1.0-frac;
            u[idx(0,   k, N)] = 1.0;
            u[idx(N-1, k, N)] = 0.0;
        }
        auto t1 = Clock::now();
        for (int k=0;k<20;k++) {
            v_cycle_cuda_run(u, rho, N, 3, 3, 1.0, 1);
            // Re-enforce BC
            for (int k2=0;k2<N;k2++) {
                double frac=(double)k2/(N-1);
                u[idx(k2,   0, N)] = 1.0-frac;
                u[idx(k2, N-1, N)] = 1.0-frac;
                u[idx(0,   k2, N)] = 1.0;
                u[idx(N-1, k2, N)] = 0.0;
            }
        }
        std::cout << "  solve=" << elapsed_ms(t1) << "ms (20 V-cycles)\n";
        write_field_csv("results/heat_boundary_gradient.csv", u, N);
    }

    std::cout << "[4/4] Ring-shaped source\n";
    run_gpu_scenario("Ring Source", "results/heat_ring_source.csv",
                     N, kernel_ring_source);

    std::cout << "\nAll done. Run plot_heat.py to visualize.\n";
    return 0;
}
