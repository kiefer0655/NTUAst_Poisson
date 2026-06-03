#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <omp.h>
#include "utils.h"

void sor_update(
    int N,
    std::vector<double>& U,
    const std::vector<double>& Rho,
    double h2,
    double w
){
    // this function will preform inplace update on U

    #pragma omp parallel reduction(max:r)
    {
    for(int color=0; color<2;color ++){
        // color = 0 orange, color = 1 blue
        #pragma omp for
        for(int i=1; i<N-1; i++){
        
            int j_start = ((i + 1) % 2 == color) ? 1 : 2;
            for (int j = j_start; j < N-1; j+=2)
            {
                int id = idx(i, j, N);
                double gs_value = 0.25 * (
                    U[id-N] +
                    U[id+N] +
                    U[id-1] +
                    U[id+1] -
                    h2 * Rho[id]);
                U[id] = U[id] + w * (gs_value - U[id]);
            }
        }
    }
    }
}

void SOR_smooth(
    int N,
    std::vector<double>& U,
    const std::vector<double>& Rho,
    double h2,
    double w,
    int iteration
){
    for (int i = 0; i < iteration; i++)
    {
        sor_update(N,U,Rho,h2,w);
    }
    
}