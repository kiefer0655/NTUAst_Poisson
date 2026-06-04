#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "smoother.h"
#include "utils.h"

int main() {
    int N = 33;
    double h = 1.0 / (N - 1);
    double h2 = h * h;
    double w = 1.0; // w=1.0 is standard Gauss-Seidel for simplicity

    std::vector<double> u(N * N, 0.0);
    std::vector<double> phi = get_phi_exact_solution(N);
    std::vector<double> u_exact = get_exact_solution(N);

    std::cout << "Testing Person A's Smoother (SOR with OpenMP)...\n";
    
    // Run smoother for 2000 iterations to check convergence
    SOR_smooth(N, u, phi, h2, w, 2000);

    double err = get_error(u, u_exact, N);
    std::cout << "L1 Error after 2000 iterations: " << err << std::endl;
    
    // Assert the error is reasonably small, proving it converged towards the exact solution
    assert(err < 1e-2 && "Smoother did not converge properly!");
    
    std::cout << "Person A's smoother tests passed successfully!" << std::endl;
    return 0;
}
