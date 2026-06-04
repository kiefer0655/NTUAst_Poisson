#pragma once
#include <vector>

// Performs one V-cycle of the Multigrid method
// u: The current potential grid (will be updated in place)
// phi: The source grid (right hand side)
// N: Current grid dimension (assumes N x N)
// nu1: Number of pre-smoothing steps
// nu2: Number of post-smoothing steps
// w: Over-relaxation parameter for SOR
void v_cycle(std::vector<double>& u, const std::vector<double>& phi, int N, int nu1, int nu2, double w);

// Performs one W-cycle of the Multigrid method
void w_cycle(std::vector<double>& u, const std::vector<double>& phi, int N, int nu1, int nu2, double w);

// Full Multigrid (FMG)
void fmg_cycle(std::vector<double>& u, const std::vector<double>& phi, int N, int nu1, int nu2, double w);
