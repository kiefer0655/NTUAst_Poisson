CXX = g++
CXXFLAGS = -O3 -fopenmp -Wall -Iinclude

all: benchmark test_smoother test_transfer test_multigrid

benchmark: src/benchmark.cpp src/multigrid.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

test_smoother: tests/test_smoother.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

test_transfer: tests/test_transfer.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

test_multigrid: tests/test_multigrid.cpp src/multigrid.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f benchmark test_smoother test_transfer test_multigrid
