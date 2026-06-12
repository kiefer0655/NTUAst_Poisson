# Makefile for Heat Conduction Demo
# Builds OpenMP, MPI, and CUDA versions

CXX      = g++
MPICXX   = mpicxx
NVCC     = nvcc
INCLUDE  = -I../../include

CXXFLAGS  = -O2 -std=c++17 $(INCLUDE)
OMPFLAGS  = -fopenmp
MPIFLAGS  =
CUDAFLAGS = -O2 -std=c++17 $(INCLUDE)

SRCS_COMMON = ../../src/multigrid.cpp

.PHONY: all omp mpi cuda clean

all: omp mpi cuda

# ── OpenMP version ─────────────────────────────────────────
omp: heat_demo

heat_demo: heat_demo.cpp $(SRCS_COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $^ -lm
	@echo "Built: heat_demo  (OpenMP)"

# ── MPI version ────────────────────────────────────────────
mpi: heat_demo_mpi

heat_demo_mpi: heat_demo_mpi.cpp ../../src/multigrid_mpi.cpp
	$(MPICXX) $(CXXFLAGS) $(MPIFLAGS) -o $@ $^ -lm
	@echo "Built: heat_demo_mpi  (MPI)"

# ── CUDA version ───────────────────────────────────────────
cuda: heat_demo_cuda

heat_demo_cuda: heat_demo_cuda.cu ../../src/multigrid_cuda.cu
	$(NVCC) $(CUDAFLAGS) -o $@ $^ -lm
	@echo "Built: heat_demo_cuda  (CUDA)"

# ── Clean ──────────────────────────────────────────────────
clean:
	rm -f heat_demo heat_demo_mpi heat_demo_cuda