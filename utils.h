#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <omp.h>

//indexing tools
int idx(int i, int j, int N){
        return i * N + j;
}

//residual
std::vector<double> get_residual_grid(
    const std::vector<double>& u,
    const std::vector<double>& phi,
    int N,
    double h2
){
    std::vector<double> r(N*N,0.0);

    #pragma omp parallel for collapse(2) schedule(static)
    for(int i = 1; i < N-1; i++)
    {
        for(int j = 1; j < N-1; j++){
            int id = idx(i,j,N);
            double u_i_laplace =    (
                                    u[id-N] +
                                    u[id+N] +
                                    u[id-1] +
                                    u[id+1] -
                                    4.0 * u[id]
                                    ) / h2;
            r[id] = phi[id] - u_i_laplace; 
        }
    }

    return r;
}

// for code testing
//test eqation: ∇^2 u=−2π2sin(πx)sin(πy)
std::vector<double> get_exact_solution(int N){
    std::vector<double> u_exact(N * N, 0.0);

    double h = 1.0 / (N - 1);
    double pi = M_PI;

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            double x = i * h;
            double y = j * h;

            u_exact[idx(i, j, N)] = sin(pi * x) * sin(pi * y);
        }
    }

    return u_exact;
}

std::vector<double> get_phi_exact_solution(int N){

    std::vector<double> phi(N*N,0.0);

    double h = 1.0 / (N - 1);
    double pi = M_PI;

    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            double x = i * h;
            double y = j * h;

            phi[idx(i,j,N)] = -2.0 * pi * pi 
                            * sin(pi * x) * sin(pi * y);
        }
        
    }
    return phi;
}

double get_error(
    const std::vector<double>&U,
    const std::vector<double>&U_exact,
    int N
){
    double total_error = 0.0;
    int    total_point = N*N;

    #pragma omp parallel for reduction(+:total_error) schedule(static)
    for (int i = 0; i < total_point; i++)
    {
        total_error += std::abs(U[i] - U_exact[i]);
    }
    
    return total_error/(total_point);
}