#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>

#include "../../include/utils.h"
#include "../../include/multigrid.h"

struct Particle {
    double x, y;
    double vx, vy;
    double m;
};

double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(v, hi));
}

/// =============================================================================================
/// convert nbody to poisson problem
/// =============================================================================================


void deposit_mass_to_grid(
    const std::vector<Particle>& particles,
    std::vector<double>& rho,
    int N,
    double h,
    double G
) {
    // reset rho
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < N * N; k++) {
        rho[k] = 0.0;
    }

    #pragma omp parallel for schedule(static)
    for (int p_id = 0; p_id < (int)particles.size(); p_id++) {
        const Particle& p = particles[p_id];

        double gx = p.x / h;
        double gy = p.y / h;

        // keep gx, gy inside the safe interpolation region
        gx = std::max(1.0, std::min(gx, (double)(N - 3)));
        gy = std::max(1.0, std::min(gy, (double)(N - 3)));

        int i = (int)std::floor(gx);
        int j = (int)std::floor(gy);

        double tx = gx - i;
        double ty = gy - j;

        double w00 = (1.0 - tx) * (1.0 - ty);
        double w10 = tx * (1.0 - ty);
        double w01 = (1.0 - tx) * ty;
        double w11 = tx * ty;

        double source = 4.0 * M_PI * G * p.m / (h * h);

        
        int id00 = idx(i,     j,     N);
        int id10 = id00+N;
        int id01 = id00+1;
        int id11 = id10+1;

        #pragma omp atomic
        rho[id00] += source * w00;

        #pragma omp atomic
        rho[id10] += source * w10;

        #pragma omp atomic
        rho[id01] += source * w01;

        #pragma omp atomic
        rho[id11] += source * w11;
    }

    // Dirichlet boundary = 0
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < N; k++) {
        rho[idx(0,     k, N)] = 0.0;
        rho[idx(N - 1, k, N)] = 0.0;
        rho[idx(k,     0, N)] = 0.0;
        rho[idx(k, N - 1, N)] = 0.0;
    }
}

void compute_acceleration_grid(
    const std::vector<double>& phi,
    const std::vector<double>& rho,
    std::vector<double>& ax,
    std::vector<double>& ay,
    int N,
    double h
) {
    // Reset acceleration grids
    int total_grid = N * N;
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < total_grid; k++) {
        ax[k] = 0.0;
        ay[k] = 0.0;
    }

    // Only compute acceleration where rho has mass
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            int id = idx(i, j, N);

            if (rho[id] != 0.0) {
                ax[id] = -(
                    phi[idx(i + 1, j, N)] -
                    phi[idx(i - 1, j, N)]
                ) / (2.0 * h);

                ay[id] = -(
                    phi[idx(i, j + 1, N)] -
                    phi[idx(i, j - 1, N)]
                ) / (2.0 * h);
            }
        }
    }
}

double bilinear_sample(
    const std::vector<double>& grid,
    double x,
    double y,
    int N,
    double h
) {
    double gx = x / h;
    double gy = y / h;

    int i = (int)std::floor(gx);
    int j = (int)std::floor(gy);

    i = std::max(1, std::min(i, N - 3));
    j = std::max(1, std::min(j, N - 3));

    double tx = gx - i;
    double ty = gy - j;

    double g00 = grid[idx(i,     j,     N)];
    double g10 = grid[idx(i + 1, j,     N)];
    double g01 = grid[idx(i,     j + 1, N)];
    double g11 = grid[idx(i + 1, j + 1, N)];

    return (1.0 - tx) * (1.0 - ty) * g00
         + tx         * (1.0 - ty) * g10
         + (1.0 - tx) * ty         * g01
         + tx         * ty         * g11;
}

void write_particles(
    std::ofstream& file,
    const std::vector<Particle>& particles,
    int step
) {
    for (int k = 0; k < (int)particles.size(); k++) {
        file << step << ","
             << k << ","
             << particles[k].x << ","
             << particles[k].y << ","
             << particles[k].vx << ","
             << particles[k].vy << "\n";
    }
}
/// =============================================================================================
/// motion update
/// =============================================================================================

bool particle_active(const Particle& p) {
    return p.m > 0.0;
}

void deactivate_particle(Particle& p) {
    p.m = 0.0;
    p.x = -1.0;
    p.y = -1.0;
    p.vx = 0.0;
    p.vy = 0.0;
}

void apply_particle_boundary_condition(Particle& p) {
    if (!particle_active(p)) {
        return;
    }

    const double xmin = 0.02;
    const double xmax = 0.98;
    const double ymin = 0.02;
    const double ymax = 0.98;

    bool outside =
        p.x < xmin || p.x > xmax ||
        p.y < ymin || p.y > ymax;

    if (outside) {
        deactivate_particle(p);
    }
}

void dkd_step(
    std::vector<Particle>& particles,
    std::vector<double>& rho,
    std::vector<double>& phi,
    std::vector<double>& ax_grid,
    std::vector<double>& ay_grid,
    int N,
    double h,
    double G,
    double dt
) {
    int num_particles = (int)particles.size();

    // Drift half step
    #pragma omp parallel for schedule(static)
    for (int p_id = 0; p_id < num_particles; p_id++) {
        Particle& p = particles[p_id];

        p.x += 0.5 * dt * p.vx;
        p.y += 0.5 * dt * p.vy;

        apply_particle_boundary_condition(p);
    }

    // Compute acceleration at half-step position
    deposit_mass_to_grid(particles, rho, N, h, G);

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < N * N; k++) {
        phi[k] = 0.0;
    }

    for (int k = 0; k < 3; k++) {
        v_cycle(phi, rho, N, 2, 2, 1.0);
    }

    fmg_cycle(phi, rho, N, 2, 2, 1.0);

    compute_acceleration_grid(
        phi,
        rho,
        ax_grid,
        ay_grid,
        N,
        h
    );

    // Kick full step
    #pragma omp parallel for schedule(static)
    for (int p_id = 0; p_id < num_particles; p_id++) {
        Particle& p = particles[p_id];

        double ax = bilinear_sample(ax_grid, p.x, p.y, N, h);
        double ay = bilinear_sample(ay_grid, p.x, p.y, N, h);

        p.vx += dt * ax;
        p.vy += dt * ay;
    }

    #pragma omp parallel for schedule(static)
    for (int p_id = 0; p_id < num_particles; p_id++) {
        Particle& p = particles[p_id];

        p.x += 0.5 * dt * p.vx;
        p.y += 0.5 * dt * p.vy;

        apply_particle_boundary_condition(p);
    }
}

/// ============================
/// main
/// ============================
int main() {
    const int N = 129;              // 2^k + 1 grid size
    const double h = 1.0 / (N - 1);

    const double G = 0.0005;        
    const double dt = 0.001;

    const int steps = 3000;
    const int output_every = 10;

    std::vector<double> rho(N * N, 0.0);
    std::vector<double> phi(N * N, 0.0);
    std::vector<double> ax_grid(N * N, 0.0);
    std::vector<double> ay_grid(N * N, 0.0);

    std::vector<Particle> particles;

    // Central heavy particle
    particles.push_back({
        0.50, 0.50,     // x, y
        0.00, 0.00,     // vx, vy
        10.0            // mass
    });

    // Ring of particles
    int n_ring = 20;
    double r = 0.22;
    double speed = 0.15;

    for (int k = 0; k < n_ring; k++) {
        double theta = 2.0 * M_PI * k / n_ring;

        double x = 0.5 + r * std::cos(theta);
        double y = 0.5 + r * std::sin(theta);

        double vx = -speed * std::sin(theta);
        double vy =  speed * std::cos(theta);

        particles.push_back({
            x, y,
            vx, vy,
            1.0
        });
    }

    std::ofstream file("results/nbody_particles.csv");

    if (!file.is_open()) {
        std::cerr << "Error: cannot open results/nbody_particles.csv\n";
        return 1;
    }

    file << "step,id,x,y,vx,vy\n";

    write_particles(file, particles, 0);

    for (int step = 1; step <= steps; step++) {
        dkd_step(
            particles,
            rho,
            phi,
            ax_grid,
            ay_grid,
            N,
            h,
            G,
            dt
        );

        if (step % output_every == 0) {
            write_particles(file, particles, step);
            std::cout << "step = " << step << "\n";
        }
    }

    file.close();

    std::cout << "Saved results/nbody_particles.csv\n";

    return 0;
}