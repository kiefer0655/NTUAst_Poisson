#include "../include/multigrid_cuda.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define THREADS_PER_BLOCK 256

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ __host__ inline int div_up(int a, int b) { return (a + b - 1) / b; }
__device__ inline int idx_d(int i, int j, int N) { return i * N + j; }

__global__ void sor_kernel(double* U, const double* Rho, int N, double h2, double w, int color) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N * N) return;
    
    int i = id / N;
    int j = id % N;

    // Boundary condition (Dirichlet = 0)
    if (i == 0 || i == N - 1 || j == 0 || j == N - 1) return;

    // Red-Black checkerboard update
    if (((i + j) % 2) == color) {
        double gs_value = 0.25 * (
            U[idx_d(i - 1, j, N)] +
            U[idx_d(i + 1, j, N)] +
            U[idx_d(i, j - 1, N)] +
            U[idx_d(i, j + 1, N)] -
            h2 * Rho[id]);
        U[id] = U[id] + w * (gs_value - U[id]);
    }
}

__global__ void get_residual_kernel(const double* U, const double* Rho, double* R, int N, double h2) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N * N) return;

    int i = id / N;
    int j = id % N;

    if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
        R[id] = 0.0;
        return;
    }

    double laplacian = (
        U[idx_d(i - 1, j, N)] +
        U[idx_d(i + 1, j, N)] +
        U[idx_d(i, j - 1, N)] +
        U[idx_d(i, j + 1, N)] -
        4.0 * U[id]) / h2;

    R[id] = Rho[id] - laplacian;
}

__global__ void restrict_kernel(const double* fine, double* coarse, int N_fine, int N_coarse) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N_coarse * N_coarse) return;

    int ic = id / N_coarse;
    int jc = id % N_coarse;

    if (ic == 0 || ic == N_coarse - 1 || jc == 0 || jc == N_coarse - 1) {
        coarse[id] = 0.0;
        return;
    }

    int if_ = 2 * ic;
    int jf  = 2 * jc;

    coarse[id] = (
        4.0 * fine[idx_d(if_, jf, N_fine)] +
        2.0 * fine[idx_d(if_ - 1, jf, N_fine)] +
        2.0 * fine[idx_d(if_ + 1, jf, N_fine)] +
        2.0 * fine[idx_d(if_, jf - 1, N_fine)] +
        2.0 * fine[idx_d(if_, jf + 1, N_fine)] +
              fine[idx_d(if_ - 1, jf - 1, N_fine)] +
              fine[idx_d(if_ - 1, jf + 1, N_fine)] +
              fine[idx_d(if_ + 1, jf - 1, N_fine)] +
              fine[idx_d(if_ + 1, jf + 1, N_fine)]
    ) / 16.0;
}

__global__ void prolong_kernel(const double* coarse, double* fine, int N_fine, int N_coarse) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N_fine * N_fine) return;

    int i = id / N_fine;
    int j = id % N_fine;

    if (i == 0 || i == N_fine - 1 || j == 0 || j == N_fine - 1) {
        fine[id] = 0.0;
        return;
    }

    int ic = i / 2;
    int jc = j / 2;

    if (i % 2 == 0 && j % 2 == 0) {
        fine[id] = coarse[idx_d(ic, jc, N_coarse)];
    } else if (i % 2 == 0 && j % 2 != 0) {
        fine[id] = 0.5 * (coarse[idx_d(ic, jc, N_coarse)] + coarse[idx_d(ic, jc + 1, N_coarse)]);
    } else if (i % 2 != 0 && j % 2 == 0) {
        fine[id] = 0.5 * (coarse[idx_d(ic, jc, N_coarse)] + coarse[idx_d(ic + 1, jc, N_coarse)]);
    } else {
        fine[id] = 0.25 * (
            coarse[idx_d(ic, jc, N_coarse)] +
            coarse[idx_d(ic, jc + 1, N_coarse)] +
            coarse[idx_d(ic + 1, jc, N_coarse)] +
            coarse[idx_d(ic + 1, jc + 1, N_coarse)]);
    }
}

__global__ void vector_add_kernel(double* u, const double* e, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N * N) {
        u[id] += e[id];
    }
}

// CPU function driving the GPU execution recursively
void v_cycle_device(double* d_u, const double* d_phi, int N, int nu1, int nu2, double w) {
    double h2 = 1.0 / ((N - 1) * (N - 1));
    int blocks = div_up(N * N, THREADS_PER_BLOCK);

    if (N <= 3) {
        for (int i = 0; i < 500; i++) {
            sor_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_u, d_phi, N, h2, w, 0);
            sor_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_u, d_phi, N, h2, w, 1);
        }
        return;
    }

    // 1. Pre-smooth
    for (int i = 0; i < nu1; i++) {
        sor_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_u, d_phi, N, h2, w, 0);
        sor_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_u, d_phi, N, h2, w, 1);
    }

    // 2. Residual
    double* d_r;
    cudaMalloc(&d_r, N * N * sizeof(double));
    get_residual_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_u, d_phi, d_r, N, h2);

    // 3. Restrict
    int N_coarse = (N - 1) / 2 + 1;
    double* d_rc;
    cudaMalloc(&d_rc, N_coarse * N_coarse * sizeof(double));
    int blocks_c = div_up(N_coarse * N_coarse, THREADS_PER_BLOCK);
    restrict_kernel<<<blocks_c, THREADS_PER_BLOCK>>>(d_r, d_rc, N, N_coarse);
    cudaFree(d_r);

    // 4. Coarse grid solve
    double* d_ec;
    cudaMalloc(&d_ec, N_coarse * N_coarse * sizeof(double));
    cudaMemset(d_ec, 0, N_coarse * N_coarse * sizeof(double));
    v_cycle_device(d_ec, d_rc, N_coarse, nu1, nu2, w);
    cudaFree(d_rc);

    // 5. Prolongate
    double* d_ef;
    cudaMalloc(&d_ef, N * N * sizeof(double));
    prolong_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_ec, d_ef, N, N_coarse);
    cudaFree(d_ec);

    // 6. Add correction
    vector_add_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_u, d_ef, N);
    cudaFree(d_ef);

    // 7. Post-smooth
    for (int i = 0; i < nu2; i++) {
        sor_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_u, d_phi, N, h2, w, 0);
        sor_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_u, d_phi, N, h2, w, 1);
    }
}

// Host wrapper
void v_cycle_cuda_run(std::vector<double>& u, const std::vector<double>& phi, int N, int nu1, int nu2, double w, int iters) {
    double* d_u;
    double* d_phi;
    size_t size = N * N * sizeof(double);

    checkCuda(cudaMalloc(&d_u, size));
    checkCuda(cudaMalloc(&d_phi, size));

    checkCuda(cudaMemcpy(d_u, u.data(), size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_phi, phi.data(), size, cudaMemcpyHostToDevice));

    for (int iter = 0; iter < iters; iter++) {
        v_cycle_device(d_u, d_phi, N, nu1, nu2, w);
        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaDeviceSynchronize());
    }

    checkCuda(cudaMemcpy(u.data(), d_u, size, cudaMemcpyDeviceToHost));

    cudaFree(d_u);
    cudaFree(d_phi);
}
