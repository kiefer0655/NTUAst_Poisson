#include "../include/multigrid_mpi.h"
#include "../include/multigrid.h"
#include "../include/smoother.h"
#include "../include/transfer.h"
#include "../include/utils.h"
#include <iostream>

void exchange_ghost_cells(std::vector<double>& u_local, int N, int local_rows, int rank, int P) {
    MPI_Request reqs[4];
    int req_cnt = 0;

    if (rank > 0) {
        MPI_Isend(&u_local[1 * N], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[req_cnt++]);
        MPI_Irecv(&u_local[0 * N], N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[req_cnt++]);
    }
    if (rank < P - 1) {
        MPI_Isend(&u_local[local_rows * N], N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[req_cnt++]);
        MPI_Irecv(&u_local[(local_rows + 1) * N], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[req_cnt++]);
    }

    if (req_cnt > 0) {
        MPI_Waitall(req_cnt, reqs, MPI_STATUSES_IGNORE);
    }
}

void sor_update_mpi(int N, int local_rows, std::vector<double>& u_local, 
                    const std::vector<double>& rho_local, double h2, double w, int rank, int P) {
    int K = (N - 1) / P;
    int start_row = rank * K;
    
    for (int color = 0; color < 2; color++) {
        for (int i = 1; i <= local_rows; i++) {
            int global_i = start_row + i - 1;
            if (global_i == 0 || global_i == N - 1) continue;
            
            int j_start = ((global_i + 1) % 2 == color) ? 1 : 2;
            for (int j = j_start; j < N - 1; j += 2) {
                int id = i * N + j;
                double gs_value = 0.25 * (
                    u_local[id - N] + u_local[id + N] +
                    u_local[id - 1] + u_local[id + 1] -
                    h2 * rho_local[id]);
                u_local[id] = u_local[id] + w * (gs_value - u_local[id]);
            }
        }
        exchange_ghost_cells(u_local, N, local_rows, rank, P);
    }
}

void SOR_smooth_mpi(int N, int local_rows, std::vector<double>& u_local, 
                    const std::vector<double>& rho_local, double h2, double w, int iters, int rank, int P) {
    for (int iter = 0; iter < iters; iter++) {
        sor_update_mpi(N, local_rows, u_local, rho_local, h2, w, rank, P);
    }
}

// Restriction
void restrict_mpi(const std::vector<double>& fine_local, std::vector<double>& coarse_local, 
                  int N_fine, int local_rows_coarse, int rank, int P) {
    int N_coarse = (N_fine - 1) / 2 + 1;
    int K_coarse = (N_coarse - 1) / P;
    int start_row_coarse = rank * K_coarse;

    for (int ic = 1; ic <= local_rows_coarse; ic++) {
        int global_ic = start_row_coarse + ic - 1;
        if (global_ic == 0 || global_ic == N_coarse - 1) continue; // Boundaries are 0
        
        int i_fine = 2 * ic - 1; // Correct mathematical mapping
        
        for (int jc = 1; jc < N_coarse - 1; jc++) {
            int j_fine = 2 * jc;
            
            double sum = 0.0;
            sum += 1.0 * fine_local[(i_fine - 1) * N_fine + (j_fine - 1)];
            sum += 2.0 * fine_local[(i_fine - 1) * N_fine + j_fine];
            sum += 1.0 * fine_local[(i_fine - 1) * N_fine + (j_fine + 1)];
            
            sum += 2.0 * fine_local[i_fine * N_fine + (j_fine - 1)];
            sum += 4.0 * fine_local[i_fine * N_fine + j_fine];
            sum += 2.0 * fine_local[i_fine * N_fine + (j_fine + 1)];
            
            sum += 1.0 * fine_local[(i_fine + 1) * N_fine + (j_fine - 1)];
            sum += 2.0 * fine_local[(i_fine + 1) * N_fine + j_fine];
            sum += 1.0 * fine_local[(i_fine + 1) * N_fine + (j_fine + 1)];
            
            coarse_local[ic * N_coarse + jc] = sum / 16.0;
        }
    }
}

// Prolongation
void prolong_mpi(const std::vector<double>& coarse_local, std::vector<double>& fine_local, 
                 int N_fine, int local_rows_coarse, int rank, int P) {
    int N_coarse = (N_fine - 1) / 2 + 1;
    int K_coarse = (N_coarse - 1) / P;
    int start_row_coarse = rank * K_coarse;

    // Zero out fine_local first
    int local_rows_fine = (rank == P - 1) ? (N_fine - rank * ((N_fine - 1) / P)) : ((N_fine - 1) / P);
    for(int i=1; i<=local_rows_fine; i++) {
        for(int j=1; j<N_fine-1; j++) {
            fine_local[i * N_fine + j] = 0.0;
        }
    }

    for (int ic = 1; ic <= local_rows_coarse; ic++) {
        int global_ic = start_row_coarse + ic - 1;
        if (global_ic == 0 || global_ic == N_coarse - 1) continue;
        
        int i_fine = 2 * ic - 1;
        
        for (int jc = 1; jc < N_coarse - 1; jc++) {
            int j_fine = 2 * jc;
            double v = coarse_local[ic * N_coarse + jc];
            
            fine_local[i_fine * N_fine + j_fine] += v;
            
            fine_local[i_fine * N_fine + (j_fine - 1)] += 0.5 * v;
            fine_local[i_fine * N_fine + (j_fine + 1)] += 0.5 * v;
            fine_local[(i_fine - 1) * N_fine + j_fine] += 0.5 * v;
            fine_local[(i_fine + 1) * N_fine + j_fine] += 0.5 * v;
            
            fine_local[(i_fine - 1) * N_fine + (j_fine - 1)] += 0.25 * v;
            fine_local[(i_fine - 1) * N_fine + (j_fine + 1)] += 0.25 * v;
            fine_local[(i_fine + 1) * N_fine + (j_fine - 1)] += 0.25 * v;
            fine_local[(i_fine + 1) * N_fine + (j_fine + 1)] += 0.25 * v;
        }
    }
}

// Global Gather and Solve
void gather_and_solve(std::vector<double>& u_local, const std::vector<double>& phi_local, 
                      int N, int rank, int P, int nu1, int nu2, double w, int cycle_type) {
    int K = (N - 1) / P;
    int local_rows = (rank == P - 1) ? (N - rank * K) : K;

    std::vector<int> recvcounts(P);
    std::vector<int> displs(P);
    for (int p = 0; p < P; p++) {
        int lr = (p == P - 1) ? (N - p * K) : K;
        recvcounts[p] = lr * N;
        displs[p] = (p * K) * N;
    }

    std::vector<double> u_global, phi_global;
    if (rank == 0) {
        u_global.resize(N * N, 0.0);
        phi_global.resize(N * N, 0.0);
    }

    MPI_Gatherv(&u_local[1 * N], local_rows * N, MPI_DOUBLE, 
                rank == 0 ? u_global.data() : nullptr, recvcounts.data(), displs.data(), 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
                
    MPI_Gatherv(&phi_local[1 * N], local_rows * N, MPI_DOUBLE, 
                rank == 0 ? phi_global.data() : nullptr, recvcounts.data(), displs.data(), 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (cycle_type == 1) v_cycle(u_global, phi_global, N, nu1, nu2, w);
        else if (cycle_type == 2) w_cycle(u_global, phi_global, N, nu1, nu2, w);
        else fmg_cycle(u_global, phi_global, N, nu1, nu2, w);
    }

    MPI_Scatterv(rank == 0 ? u_global.data() : nullptr, recvcounts.data(), displs.data(), 
                 MPI_DOUBLE, &u_local[1 * N], local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                 
    exchange_ghost_cells(u_local, N, local_rows, rank, P);
}

std::vector<double> get_residual_mpi(const std::vector<double>& u_local, const std::vector<double>& phi_local, 
                                     int N, int local_rows, double h2, int rank, int P) {
    int K = (N - 1) / P;
    int start_row = rank * K;
    std::vector<double> r_local((local_rows + 2) * N, 0.0);
    
    for (int i = 1; i <= local_rows; i++) {
        int global_i = start_row + i - 1;
        if (global_i == 0 || global_i == N - 1) continue;
        
        for (int j = 1; j < N - 1; j++) {
            int id = i * N + j;
            double Au = (4.0 * u_local[id] - u_local[id - 1] - u_local[id + 1] - u_local[id - N] - u_local[id + N]) / h2;
            r_local[id] = phi_local[id] - Au;
        }
    }
    return r_local;
}

void v_cycle_mpi(std::vector<double>& u_local, const std::vector<double>& phi_local, 
                 int N, int rank, int P, int nu1, int nu2, double w) {
    int N_coarse = (N - 1) / 2 + 1;
    if (N_coarse - 1 < P || N <= 3) {
        gather_and_solve(u_local, phi_local, N, rank, P, nu1, nu2, w, 1);
        return;
    }

    int K = (N - 1) / P;
    int local_rows = (rank == P - 1) ? (N - rank * K) : K;
    double h2 = 1.0 / ((N - 1) * (N - 1));

    SOR_smooth_mpi(N, local_rows, u_local, phi_local, h2, w, nu1, rank, P);

    // Compute residual. Need ghost cells!
    exchange_ghost_cells(u_local, N, local_rows, rank, P);
    std::vector<double> r_local = get_residual_mpi(u_local, phi_local, N, local_rows, h2, rank, P);
    exchange_ghost_cells(r_local, N, local_rows, rank, P);

    int N_coarse = (N - 1) / 2 + 1;
    int K_coarse = (N_coarse - 1) / P;
    int local_rows_coarse = (rank == P - 1) ? (N_coarse - rank * K_coarse) : K_coarse;

    std::vector<double> r_coarse_local((local_rows_coarse + 2) * N_coarse, 0.0);
    restrict_mpi(r_local, r_coarse_local, N, local_rows_coarse, rank, P);

    std::vector<double> e_coarse_local((local_rows_coarse + 2) * N_coarse, 0.0);
    v_cycle_mpi(e_coarse_local, r_coarse_local, N_coarse, rank, P, nu1, nu2, w);

    std::vector<double> e_fine_local((local_rows + 2) * N, 0.0);
    prolong_mpi(e_coarse_local, e_fine_local, N, local_rows_coarse, rank, P);

    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j < N - 1; j++) {
            u_local[i * N + j] += e_fine_local[i * N + j];
        }
    }

    SOR_smooth_mpi(N, local_rows, u_local, phi_local, h2, w, nu2, rank, P);
}

void w_cycle_mpi(std::vector<double>& u_local, const std::vector<double>& phi_local, 
                 int N, int rank, int P, int nu1, int nu2, double w) {
    int N_coarse = (N - 1) / 2 + 1;
    if (N_coarse - 1 < P || N <= 3) {
        gather_and_solve(u_local, phi_local, N, rank, P, nu1, nu2, w, 2);
        return;
    }

    int K = (N - 1) / P;
    int local_rows = (rank == P - 1) ? (N - rank * K) : K;
    double h2 = 1.0 / ((N - 1) * (N - 1));

    SOR_smooth_mpi(N, local_rows, u_local, phi_local, h2, w, nu1, rank, P);

    exchange_ghost_cells(u_local, N, local_rows, rank, P);
    std::vector<double> r_local = get_residual_mpi(u_local, phi_local, N, local_rows, h2, rank, P);
    exchange_ghost_cells(r_local, N, local_rows, rank, P);

    int N_coarse = (N - 1) / 2 + 1;
    int K_coarse = (N_coarse - 1) / P;
    int local_rows_coarse = (rank == P - 1) ? (N_coarse - rank * K_coarse) : K_coarse;

    std::vector<double> r_coarse_local((local_rows_coarse + 2) * N_coarse, 0.0);
    restrict_mpi(r_local, r_coarse_local, N, local_rows_coarse, rank, P);

    std::vector<double> e_coarse_local((local_rows_coarse + 2) * N_coarse, 0.0);
    w_cycle_mpi(e_coarse_local, r_coarse_local, N_coarse, rank, P, nu1, nu2, w);
    w_cycle_mpi(e_coarse_local, r_coarse_local, N_coarse, rank, P, nu1, nu2, w);

    std::vector<double> e_fine_local((local_rows + 2) * N, 0.0);
    prolong_mpi(e_coarse_local, e_fine_local, N, local_rows_coarse, rank, P);

    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j < N - 1; j++) {
            u_local[i * N + j] += e_fine_local[i * N + j];
        }
    }

    SOR_smooth_mpi(N, local_rows, u_local, phi_local, h2, w, nu2, rank, P);
}

void fmg_cycle_mpi(std::vector<double>& u_local, const std::vector<double>& phi_local, 
                   int N, int rank, int P, int nu1, int nu2, double w) {
    int N_coarse_check = (N - 1) / 2 + 1;
    if (N_coarse_check - 1 < P || N <= 3) {
        gather_and_solve(u_local, phi_local, N, rank, P, nu1, nu2, w, 3);
        return;
    }

    int K = (N - 1) / P;
    int local_rows = (rank == P - 1) ? (N - rank * K) : K;

    int N_coarse = (N - 1) / 2 + 1;
    int K_coarse = (N_coarse - 1) / P;
    int local_rows_coarse = (rank == P - 1) ? (N_coarse - rank * K_coarse) : K_coarse;

    std::vector<double> phi_coarse_local((local_rows_coarse + 2) * N_coarse, 0.0);
    
    // exchange ghost for phi before restriction? phi doesn't change, but we need ghosts to restrict.
    std::vector<double> phi_local_copy = phi_local;
    exchange_ghost_cells(phi_local_copy, N, local_rows, rank, P);
    
    restrict_mpi(phi_local_copy, phi_coarse_local, N, local_rows_coarse, rank, P);

    std::vector<double> u_coarse_local((local_rows_coarse + 2) * N_coarse, 0.0);
    fmg_cycle_mpi(u_coarse_local, phi_coarse_local, N_coarse, rank, P, nu1, nu2, w);

    prolong_mpi(u_coarse_local, u_local, N, local_rows_coarse, rank, P);

    v_cycle_mpi(u_local, phi_local, N, rank, P, nu1, nu2, w);
}
