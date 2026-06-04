#pragma once
#include <vector>

// Host wrapper that allocates GPU memory, runs multiple V-cycles entirely on the GPU, and returns the result.
void v_cycle_cuda_run(std::vector<double>& u, const std::vector<double>& phi, int N, int nu1, int nu2, double w, int iters);
