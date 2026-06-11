#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>


#include "../include/utils.h"
#include "../include/multigrid.h"


std::vector<double> make_demo_charge_distribution(int N) {
    std::vector<double> rho(N * N, 0.0);
    double h = 1.0 / (N - 1);

    // Two smooth Gaussian charges: + charge and - charge.
    double x1 = 0.35;
    double y1 = 0.50;
    double q1 = 1.0;
    double x2 = 0.65;
    double y2 = 0.50;
    double q2 = -1.0;
    double sigma = 0.05;

    for (int i = 0; i < N; i++) {
        double x = i * h;

        for (int j = 0; j < N; j++) {
            double y = j * h;

            double r1_sq = (x - x1) * (x - x1) + (y - y1) * (y - y1);
            double r2_sq = (x - x2) * (x - x2) + (y - y2) * (y - y2);

            double rho1 = q1 * std::exp(-r1_sq / (2.0 * sigma * sigma));
            double rho2 = q2 * std::exp(-r2_sq / (2.0 * sigma * sigma));

            rho[idx(i, j, N)] = rho1 + rho2;
        }
    }

    return rho;
}

std::vector<double> make_parallel_plate_capacitor_charge_distribution(int N) {
    std::vector<double> rho(N * N, 0.0);
    double h = 1.0 / (N - 1);

    // Two finite vertical plates with opposite charge density.
    double left_plate_x = 0.40;
    double right_plate_x = 0.60;
    double plate_y_min = 0.25;
    double plate_y_max = 0.75;
    double plate_thickness = 2.0 * h;
    double edge_smoothing = 2.0 * h;
    double charge_density = 1.0;

    for (int i = 0; i < N; i++) {
        double x = i * h;

        for (int j = 0; j < N; j++) {
            double y = j * h;

            double lower_edge = 0.5 * (1.0 + std::tanh((y - plate_y_min) / edge_smoothing));
            double upper_edge = 0.5 * (1.0 + std::tanh((plate_y_max - y) / edge_smoothing));
            double plate_window = lower_edge * upper_edge;

            double left_plate = std::exp(
                -(x - left_plate_x) * (x - left_plate_x) /
                (2.0 * plate_thickness * plate_thickness)
            );
            double right_plate = std::exp(
                -(x - right_plate_x) * (x - right_plate_x) /
                (2.0 * plate_thickness * plate_thickness)
            );

            rho[idx(i, j, N)] = charge_density * plate_window * (left_plate - right_plate);
        }
    }

    return rho;
}

std::vector<double> make_circular_capacitor_charge_distribution(int N) {
    std::vector<double> rho(N * N, 0.0);
    double h = 1.0 / (N - 1);

    // Negative inner circular shell surrounded by a positive outer shell.
    double center_x = 0.5;
    double center_y = 0.5;
    double inner_radius = 0.08;
    double outer_radius = 0.2;
    double shell_half_width = 2.0 * h;
    double inner_charge_density = -1.0;
    double outer_charge_density = 0.0;
    double inner_weight_sum = 0.0;
    double outer_weight_sum = 0.0;
    std::vector<double> inner_weight(N * N, 0.0);
    std::vector<double> outer_weight(N * N, 0.0);

    for (int i = 0; i < N; i++) {
        double x = i * h;

        for (int j = 0; j < N; j++) {
            double y = j * h;
            double dx = x - center_x;
            double dy = y - center_y;
            double r = std::sqrt(dx * dx + dy * dy);

            double negative_inner_shell =
                std::abs(r - inner_radius) <= shell_half_width ? 1.0 : 0.0;
            double positive_outer_shell =
                std::abs(r - outer_radius) <= shell_half_width ? 1.0 : 0.0;

            int id = idx(i, j, N);
            inner_weight[id] = negative_inner_shell;
            outer_weight[id] = positive_outer_shell;
            inner_weight_sum += negative_inner_shell;
            outer_weight_sum += positive_outer_shell;
        }
    }

    if (outer_weight_sum > 0.0) {
        outer_charge_density = -inner_charge_density * inner_weight_sum / outer_weight_sum;
    }

    for (int i = 0; i < N * N; i++) {
        rho[i] = inner_charge_density * inner_weight[i]
               + outer_charge_density * outer_weight[i];
    }


    return rho;
}

double total_charge(
    const std::vector<double>& rho,
    int N
) {
    double h = 1.0 / (N - 1);
    double total = 0.0;

    #pragma omp parallel for reduction(+:total) schedule(static)
    for (int i = 0; i < N * N; i++) {
        total += rho[i];
    }

    return total * h * h;
}

void write_grid_csv(
    const std::string& filename,
    const std::vector<double>& grid,
    int N
) {
    std::ofstream file(filename);

    if (!file) {
        throw std::runtime_error("Cannot write file: " + filename);
    }

    file << std::setprecision(12);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            file << grid[idx(i, j, N)];

            if (j < N - 1) {
                file << ",";
            }
        }

        file << "\n";
    }
}

void write_field_csv(
    const std::string& filename,
    const std::vector<double>& rho,
    const std::vector<double>& V,
    int N
) {
    std::ofstream file(filename);

    if (!file) {
        throw std::runtime_error("Cannot write file: " + filename);
    }

    double h = 1.0 / (N - 1);

    file << "x,y,rho,V,Ex,Ey\n";
    file << std::setprecision(12);

    for (int i = 1; i < N - 1; i++) {
        double x = i * h;

        for (int j = 1; j < N - 1; j++) {
            double y = j * h;

            double dVdx = (V[idx(i + 1, j, N)] - V[idx(i - 1, j, N)]) / (2.0 * h);
            double dVdy = (V[idx(i, j + 1, N)] - V[idx(i, j - 1, N)]) / (2.0 * h);

            double Ex = -dVdx;
            double Ey = -dVdy;

            file << x << ","
                 << y << ","
                 << rho[idx(i, j, N)] << ","
                 << V[idx(i, j, N)] << ","
                 << Ex << ","
                 << Ey << "\n";
        }
    }
}

double residual_norm(
    const std::vector<double>& V,
    const std::vector<double>& rhs,
    int N
) {
    double h2 = 1.0 / ((N - 1) * (N - 1));
    std::vector<double> r = get_residual_grid(V, rhs, N, h2);

    double total = 0.0;

    #pragma omp parallel for reduction(+:total) schedule(static)
    for (int i = 0; i < N * N; i++) {
        total += r[i] * r[i];
    }

    return std::sqrt(total / (N * N));
}

int main() {
    int N = 129;
    std::string output_prefix = "results/charge_demo";
    std::vector<double> rho = make_demo_charge_distribution(N);
    double net_charge = total_charge(rho, N);
    // Electrostatic Poisson equation:
    //     ∇² V = -rho / epsilon0
    // Here we use dimensionless units: epsilon0 = 1.
    double epsilon0 = 1.0;

    std::vector<double> rhs(N * N, 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N * N; i++) {
        rhs[i] = -rho[i] / epsilon0;
    }

    // Boundary condition:
    // V = 0 on the boundary, equivalent to a grounded square box.
    for (int i = 0; i < N; i++) {
        rhs[idx(i, 0, N)] = 0.0;
        rhs[idx(i, N - 1, N)] = 0.0;
        rhs[idx(0, i, N)] = 0.0;
        rhs[idx(N - 1, i, N)] = 0.0;
    }

    std::vector<double> V(N * N, 0.0);

    std::cout << "Solving Poisson equation with FMG + V-cycles...\n";
    std::cout << "Total charge = " << std::scientific << net_charge << "\n";

    fmg_cycle(V, rhs, N, 2, 2, 1.0);

    // A few extra V-cycles make the result cleaner for arbitrary charge distributions.
    for (int iter = 0; iter < 5; iter++) {
        v_cycle(V, rhs, N, 2, 2, 1.0);
    }

    double res = residual_norm(V, rhs, N);

    std::cout << "Final residual norm = " << std::scientific << res << "\n";

    write_grid_csv(output_prefix + "_rho.csv", rho, N);
    write_grid_csv(output_prefix + "_potential.csv", V, N);
    write_field_csv(output_prefix + "_field.csv", rho, V, N);

    std::cout << "Saved:\n";
    std::cout << "  " << output_prefix << "_rho.csv\n";
    std::cout << "  " << output_prefix << "_potential.csv\n";
    std::cout << "  " << output_prefix << "_field.csv\n";

    return 0;
}
