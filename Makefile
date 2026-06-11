CXX = g++
CXXFLAGS = -O3 -fopenmp -Wall -Iinclude

MPICXX = mpicxx
MPICXXFLAGS = -O3 -fopenmp -Wall -Iinclude

NVCC = nvcc
NVCCFLAGS = -O3 -Iinclude -arch=sm_89

all: benchmark test_smoother test_transfer test_multigrid benchmark_cuda benchmark_cpu benchmark_gpu benchmark_mpi solve_canvas solve_canvas_cpu

benchmark: src/benchmark.cpp src/multigrid.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

benchmark_cuda: src/benchmark_cuda.cu src/multigrid_cuda.cu src/multigrid.cpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp $^ -o $@

benchmark_cpu: src/benchmark_cpu.cpp src/multigrid.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

benchmark_gpu: src/benchmark_gpu.cu src/multigrid_cuda.cu src/multigrid.cpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp $^ -o $@

benchmark_mpi: src/benchmark_mpi.cpp src/multigrid_mpi.cpp src/multigrid.cpp
	$(MPICXX) $(MPICXXFLAGS) $^ -o $@

solve_canvas: src/solve_canvas.cu src/multigrid_cuda.cu src/multigrid.cpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp $^ -o $@

solve_canvas_cpu: src/solve_canvas_cpu.cpp src/multigrid.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

test_smoother: tests/test_smoother.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

test_transfer: tests/test_transfer.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

test_multigrid: tests/test_multigrid.cpp src/multigrid.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f benchmark test_smoother test_transfer test_multigrid benchmark_cuda benchmark_cpu benchmark_gpu benchmark_mpi solve_canvas solve_canvas_cpu
