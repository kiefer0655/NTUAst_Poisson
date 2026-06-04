#pragma once
#include <vector>
#include <mpi.h>

void v_cycle_mpi(std::vector<double>& u_local, const std::vector<double>& phi_local, 
                 int N, int rank, int P, int nu1, int nu2, double w);

void w_cycle_mpi(std::vector<double>& u_local, const std::vector<double>& phi_local, 
                 int N, int rank, int P, int nu1, int nu2, double w);

void fmg_cycle_mpi(std::vector<double>& u_local, const std::vector<double>& phi_local, 
                   int N, int rank, int P, int nu1, int nu2, double w);

// Ghost cell exchange
void exchange_ghost_cells(std::vector<double>& u_local, int N, int local_rows, int rank, int P);

// Red-Black SOR for MPI
void sor_update_mpi(int N, int local_rows, std::vector<double>& u_local, 
                    const std::vector<double>& rho_local, double h2, double w, int rank, int P);

void SOR_smooth_mpi(int N, int local_rows, std::vector<double>& u_local, 
                    const std::vector<double>& rho_local, double h2, double w, int iters, int rank, int P);
