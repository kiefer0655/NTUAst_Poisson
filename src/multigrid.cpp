#include "../include/multigrid.h"
#include "../include/smoother.h"
#include "../include/transfer.h"
#include "../include/utils.h"

void v_cycle(std::vector<double>& u, const std::vector<double>& phi, int N, int nu1, int nu2, double w) {
    double h2 = 1.0 / ((N - 1) * (N - 1));

    // Base case: if grid is small enough (3x3), solve exactly or smooth extensively
    if (N <= 3) {
        SOR_smooth(N, u, phi, h2, w, 500);
        return;
    }

    // 1. Pre-smooth
    SOR_smooth(N, u, phi, h2, w, nu1);

    // 2. Compute residual on the fine grid: r = phi - A*u
    std::vector<double> r = get_residual_grid(u, phi, N, h2);

    // 3. Restrict residual to coarse grid
    int N_coarse;
    std::vector<double> r_coarse = make_coarse_grid(N, &N_coarse);
    restrict_full_weighting(r, r_coarse, N);

    // 4. Solve coarse grid equation recursively for error 'e'
    std::vector<double> e_coarse(N_coarse * N_coarse, 0.0);
    v_cycle(e_coarse, r_coarse, N_coarse, nu1, nu2, w);

    // 5. Prolongate correction back to fine grid
    std::vector<double> e_fine(N * N, 0.0);
    prolong_bilinear(e_coarse, e_fine, N);

    // Add correction to u
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N * N; i++) {
        u[i] += e_fine[i];
    }

    // 6. Post-smooth
    SOR_smooth(N, u, phi, h2, w, nu2);
}

void w_cycle(std::vector<double>& u, const std::vector<double>& phi, int N, int nu1, int nu2, double w) {
    double h2 = 1.0 / ((N - 1) * (N - 1));

    // Base case
    if (N <= 3) {
        SOR_smooth(N, u, phi, h2, w, 500);
        return;
    }

    // 1. Pre-smooth
    SOR_smooth(N, u, phi, h2, w, nu1);

    // 2. Compute residual
    std::vector<double> r = get_residual_grid(u, phi, N, h2);

    // 3. Restrict residual
    int N_coarse;
    std::vector<double> r_coarse = make_coarse_grid(N, &N_coarse);
    restrict_full_weighting(r, r_coarse, N);

    // 4. Solve coarse grid equation recursively TWICE (W-cycle)
    std::vector<double> e_coarse(N_coarse * N_coarse, 0.0);
    w_cycle(e_coarse, r_coarse, N_coarse, nu1, nu2, w);
    w_cycle(e_coarse, r_coarse, N_coarse, nu1, nu2, w);

    // 5. Prolongate correction
    std::vector<double> e_fine(N * N, 0.0);
    prolong_bilinear(e_coarse, e_fine, N);

    // Add correction
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N * N; i++) {
        u[i] += e_fine[i];
    }

    // 6. Post-smooth
    SOR_smooth(N, u, phi, h2, w, nu2);
}
