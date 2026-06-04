#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "utils.h"
#include "transfer/transfer.h"

int main() {
    std::cout << "Testing Person B's Transfer Functions (1D Vector Refactor)...\n";
    
    int N_fine = 5;
    int N_coarse;
    std::vector<double> coarse = make_coarse_grid(N_fine, &N_coarse);
    
    assert(N_coarse == 3 && "Coarse grid size calculation is wrong.");

    std::vector<double> fine(N_fine * N_fine, 0.0);
    
    // Set a block of 1s in the center of the fine grid
    for(int i = 1; i < N_fine - 1; i++) {
        for(int j = 1; j < N_fine - 1; j++) {
            fine[idx(i, j, N_fine)] = 1.0;
        }
    }

    std::cout << "Running restriction...\n";
    restrict_full_weighting(fine, coarse, N_fine);

    // The center of the coarse grid should have a positive value after restriction
    assert(coarse[idx(1, 1, N_coarse)] > 0.0 && "Restriction failed to transfer values.");

    std::cout << "Running prolongation...\n";
    std::vector<double> fine_reconstructed(N_fine * N_fine, 0.0);
    prolong_bilinear(coarse, fine_reconstructed, N_fine);

    // The center of the fine grid should have a positive value after prolongation
    assert(fine_reconstructed[idx(2, 2, N_fine)] > 0.0 && "Prolongation failed to transfer values.");

    std::cout << "Person B's transfer tests passed successfully!" << std::endl;
    return 0;
}
