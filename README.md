# High-Performance Poisson Multigrid Solver

A comprehensive High-Performance Computing (HPC) implementation of the 2D Poisson Equation solver using Geometric Multigrid methods. This project accelerates the numerical solution across three distinct parallel computing architectures: **Shared Memory (OpenMP)**, **Distributed Memory (MPI)**, and **GPU Acceleration (CUDA)**.

## 1. Problem Description

This project solves the 2-Dimensional Poisson Equation:
```math
\nabla^2 u(x, y) = \rho(x, y)
```

Subject to Homogeneous Dirichlet Boundary Conditions ($u = 0$ on the boundaries).
To strictly verify our numerical accuracy, we employ the **Method of Manufactured Solutions (MMS)**. 
- We define the exact analytical solution as: 
  ```math
  u_{\text{exact}}(x,y) = \sin(\pi x)\sin(\pi y)
  ```
- This mathematically demands the source term: 
  ```math
  \rho(x,y) = 2\pi^2 \sin(\pi x)\sin(\pi y)
  ```

By forcing this exact mathematical setup, we can compute the precise $L_2$ error norm of our numerical approximations against the true analytical reality, proving the algorithm's correctness across all parallel architectures.

---

## 2. Directory Structure & Files

- **`include/`**: Header files defining the core solver interfaces.
  - `multigrid.h`, `multigrid_mpi.h`: API definitions for V/W/FMG cycles.
  - `smoother.h`: Red-Black Gauss-Seidel/SOR implementations.
  - `transfer.h`: Stencil operations for Restriction and Prolongation.
  - `utils.h`: Math utilities for generating grids and computing $L_2$ error norms.
- **`src/`**: Implementation files containing the heavy algorithmic lifting.
  - `multigrid.cpp`: Core C++ implementation with OpenMP pragmas for shared memory parallelization.
  - `multigrid_mpi.cpp`: Complex Distributed Memory implementation utilizing `MPI_Isend/Irecv` for Ghost Cell Exchange.
  - `multigrid_cuda.cu`: GPU-accelerated implementation using highly tuned CUDA kernels.
  - `benchmark_*.cpp/.cu`: Dedicated drivers to independently benchmark CPU, GPU, and MPI architectures.
- **`results/`**: Output directory for generated CSVs and benchmark images.
- **`tests/`**: Unit tests validating isolated components (Smoother, Transfer operations) to guarantee geometric ratios don't drift.
- **`plot_results.py`**: Python script generating Matplotlib data visualizations of the benchmark CSV files.
- **`run_benchmark_mpi.sh`**: Automated bash script to sweep through MPI core counts and bypass UCX hardware security limits.
- **`Makefile`**: Multi-compiler build system linking `g++`, `mpicxx`, and `nvcc`.

---

## 3. Numerical Methods & Parallel Architectures

### Algorithmic Methods
1. **Baseline / SOR (Successive Over-Relaxation):** A standard iterative solver. Used as our theoretical $O(N^2)$ baseline.
2. **V-Cycle:** A Geometric Multigrid method that traverses down to the coarsest grid ($3 \times 3$) and back up, drastically accelerating low-frequency error smoothing.
3. **W-Cycle:** Performs two recursive calls at each coarse level, acting as a stronger solver but increasing synchronization overhead.
4. **Full Multigrid (FMG):** The pinnacle algorithmic method. It begins at the coarsest grid, interpolates up, and runs V-Cycles. Achieves $O(N)$ algorithmic complexity.

### Parallel Computing Architectures
1. **Shared Memory (OpenMP):** Utilizes heavily nested `#pragma omp parallel for` loops. It implements a Red-Black ordering scheme to allow lock-free parallel execution of the Smoother.
2. **GPU Acceleration (CUDA):** Offloads the massive grid matrices to GPU VRAM. It launches 2D Thread Blocks (`dim3(16, 16)`) to map hardware warps directly over the spatial grid, achieving massive memory bandwidth throughput.
3. **Distributed Memory (MPI):** The most complex implementation. Uses a 1D Row-wise Domain Decomposition. Each processor handles a slice of the grid, requiring continuous **Ghost Cell Exchange** at the boundaries between the Red and Black smoothing passes to maintain mathematical coherency. 

---

## 4. Hardware Environments

Extensive benchmarking was performed across two distinct high-performance systems.

### 1. CPU Distributed/Shared Memory Server (`ws7`)
- **CPU:** Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
- **Architecture:** 144 Hardware Threads (72 Physical Cores over 2 NUMA sockets)
- **L3 Cache:** 108 MiB

### 2. GPU Acceleration Server (`meow2`)
- **GPU:** NVIDIA GeForce RTX 4090 (Using strictly GPU `ID: 3`)
- **VRAM:** 24 GB GDDR6X
- **Compute Power:** 16,384 CUDA Cores with immense Memory Bandwidth (the critical bottleneck for Stencil codes).

---

## 5. Verification & Accuracy Strategy

Accuracy was mathematically guaranteed before any benchmarking began:
1. **Double Precision:** All memory allocations (`std::vector<double>`) strictly enforce IEEE 754 64-bit precision to prevent recursive floating-point truncation during Deep W-Cycles.
2. **Cross-Architecture Validation:** Because the OpenMP CPU solver, the MPI Distributed Solver, and the CUDA GPU solver operate on the exact same Manufactured Solution, they produce identical mathematical $L_2$ error norms down to $\sim 10^{-7}$. 
3. **Consistent Boundary Enforcement:** The algorithm strictly zeros out boundaries after every Prolongation and Restriction step, ensuring Dirichlet conditions are never violated.

---

## 6. Implementation Challenges & Problem Resolution

Transitioning this from a basic serial code to a multi-architecture HPC codebase introduced severe technical challenges:

### 1. The MPI Multigrid Buffer Overflow (Geometric Mapping Collapse)
**Problem:** Implementing Distributed Memory Multigrid is incredibly dangerous because the grid shrinks recursively. When benchmarking `N=33` on $P=4$ processors, the script hit a violent Segmentation Fault.
**Cause:** At the coarsest levels, the grid became too small to divide evenly. Rank 3 received the entire coarse grid ($K_{coarse} = 0$), but only owned a fraction of the fine grid. During Prolongation, Rank 3 attempted to blindly write the entire coarse grid into its tiny fine grid array, shattering the memory boundaries.
**Resolution:** We engineered a "Look-Ahead" threshold. The algorithm now checks if `N_coarse - 1 < P`. If the *next* level will break the geometric mapping, it immediately executes an `MPI_Gatherv` to pull the grid to Rank 0, solves it sequentially, and uses `MPI_Scatterv` to distribute the results back safely.

### 2. UCX InfiniBand Memory Locking (`ulimit`) Restrictions
**Problem:** The 144-core `ws7` machine crashed during MPI execution, citing `Cannot allocate memory : Please set max locked memory (ulimit -l) to 'unlimited'`.
**Cause:** The OpenMPI UCX backend attempted to use InfiniBand Zero-Copy RDMA, which requires "pinning" RAM. As a non-root user, Linux blocked the memory lock.
**Resolution:** Modified `run_benchmark_mpi.sh` to inject `export UCX_TLS=sm,tcp,self`, intentionally bypassing the InfiniBand drivers and forcing standard shared-memory inter-process communication.

### 3. OpenMP "Oversubscription"
**Problem:** Running 144 threads natively was actually drastically *slower* than running 16 threads.
**Resolution:** Discovered the "Memory Wall" bottleneck (detailed below in Results Analysis).

---

## 7. Benchmarking Methodology

To generate fair and scientifically sound data:
1. **Isolated Binaries:** Separate `benchmark_cpu`, `benchmark_gpu`, and `benchmark_mpi` targets were built to ensure GPU libraries didn't bloat the CPU profiling.
2. **Time Measurement:** `std::chrono::high_resolution_clock` was used strictly around the mathematical kernels (excluding initialization and `MPI_Init` overhead).
3. **Scaling Grids:** Benchmarks sweep exponentially from $33 \times 33$ to $2049 \times 2049$ to prove Big-O theoretical limits.

---

## 8. Results Analysis & Performance Insights

*(Run `python plot_results.py` to generate the accompanying graphs in `results/images/`)*

### 1. Algorithmic Complexity: Multigrid vs SOR
Our results beautifully validate the theoretical math. The Baseline **SOR** method scales at $O(N^2)$; as the grid doubles, the execution time explodes by roughly 4x. 
However, **FMG (Full Multigrid)** and the **V-Cycle** showcase a flat linear $O(N)$ curve. Multigrid proves that intelligently moving data between frequencies mathematically dominates sheer brute-force smoothing.

![Algorithmic Scaling](results/images/algorithmic_scaling.png)

### 2. V-Cycle vs W-Cycle 
While W-Cycles theoretically offer better error reduction per cycle by spending more time on the coarse grids, our benchmarking proves it is highly inefficient for Parallel execution. The constant bouncing between levels creates immense **Thread Synchronization Overhead**. The V-Cycle (and by extension, FMG) drastically outperforms the W-Cycle in pure wall-clock execution time.

### 3. Hardware Scaling: CPU vs GPU vs MPI
The **RTX 4090 GPU** completely destroyed the CPU architectures on the $2049 \times 2049$ grid:
- It achieved nearly a **10x - 12x speedup** over the 144-core CPU.
- Stencil algorithms (like Poisson Smoothers) are overwhelmingly **Memory Bandwidth Bound** rather than Compute Bound. The RTX 4090's 1,008 GB/s GDDR6X VRAM bandwidth allowed it to chew through the grid infinitely faster than the Xeon Platinum's standard DDR4 memory controller could feed its 72 cores.

![Hardware Comparison](results/images/hardware_comparison.png)

### 4. Strong Scaling & Amdahl's Law
We ran a Strong Scaling benchmark on `ws7` (keeping $N=2049$ fixed and sweeping OpenMP threads from 1 to 144).
- **The Sweet Spot:** Performance scaled perfectly from 1 to 8 threads.
- **The Memory Wall:** Between 8 and 16 threads, performance flattened. The CPU cores were calculating the math faster than the RAM could supply the data.
- **The Overhead Collapse (Amdahl's Law):** At 144 threads, the FMG algorithm took *longer* than running on a single thread! Forcing 144 threads to wake up, synchronize, and divide a tiny $3 \times 3$ grid at the bottom of the V-Cycle meant the CPU spent 99% of its time coordinating threads and 1% doing math.

![Strong Scaling](results/images/strong_scaling.png)

---

## How to Compile & Test

Ensure you have the required compilers (`g++`, `mpicxx`, `nvcc`):

```bash
# Compile everything
make clean
make all

# Run OpenMP CPU Benchmarks (generates results_cpu.csv and scaling_cpu.csv)
./benchmark_cpu

# Run GPU CUDA Benchmarks (Ensure you are on the meow2 server with RTX 4090)
./benchmark_gpu

# Run MPI Benchmarks (Sweeps cores 1 through 64)
chmod +x run_benchmark_mpi.sh
./run_benchmark_mpi.sh

# Generate Graphs (Requires pandas, matplotlib, seaborn)
python plot_results.py
```
