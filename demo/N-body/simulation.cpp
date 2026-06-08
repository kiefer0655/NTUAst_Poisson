#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <random>

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

std::vector<Particle> make_ring_initial_condition(
    int n_ring,
    double cx,
    double cy,
    double r,
    double speed
) {
    std::vector<Particle> particles;

    // Central heavy particle
    particles.push_back({
        cx, cy,        // x, y
        0.0, 0.0,      // vx, vy
        20.0           // mass
    });

    // Ring particles
    for (int k = 0; k < n_ring; k++) {
        double theta = 2.0 * M_PI * k / n_ring;

        double x = cx + r * std::cos(theta);
        double y = cy + r * std::sin(theta);

        double vx = -speed * std::sin(theta);
        double vy =  speed * std::cos(theta);

        particles.push_back({
            x, y,
            vx, vy,
            1.0
        });
    }

    return particles;
}

std::vector<Particle> make_rotating_disk_initial_condition(
    int num_particles,
    double cx,
    double cy
) {
    std::vector<Particle> particles;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    std::normal_distribution<double> normal01(0.0, 1.0);

    // Central massive object
    particles.push_back({
        cx, cy,
        0.0, 0.0,
        80.0
    });

    const double particle_mass = 1.0;

    const double r_min = 0.035;
    const double r_max = 0.22;

    const double v0 = 0.16;
    const double core_radius = 0.07;
    const double velocity_noise = 0.006;

    for (int n = 0; n < num_particles; n++) {
        double u = uniform01(rng);
        double theta = 2.0 * M_PI * uniform01(rng);

        double r = r_min + (r_max - r_min) * std::sqrt(u);

        double x = cx + r * std::cos(theta);
        double y = cy + r * std::sin(theta);

        double v = v0 * r / std::sqrt(r * r + core_radius * core_radius);

        double vx = -v * std::sin(theta);
        double vy =  v * std::cos(theta);

        vx += velocity_noise * normal01(rng);
        vy += velocity_noise * normal01(rng);

        particles.push_back({
            x, y,
            vx, vy,
            particle_mass
        });
    }

    return particles;
}

std::vector<Particle> make_spiral_galaxy_initial_condition(
    int num_particles,
    double cx,
    double cy
) {
    std::vector<Particle> particles;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    std::normal_distribution<double> normal01(0.0, 1.0);

    particles.push_back({
        cx, cy,
        0.0, 0.0,
        100.0
    });

    int arms = 3;

    double r_min = 0.025;
    double r_max = 0.24;

    double v0 = 0.16;
    double core_radius = 0.07;

    double angle_noise = 0.18;
    double velocity_noise = 0.005;

    for (int n = 0; n < num_particles; n++) {
        int arm = n % arms;

        double u = uniform01(rng);
        double r = r_min + (r_max - r_min) * std::sqrt(u);

        double base_angle = 2.0 * M_PI * arm / arms;

        // Spiral angle: angle changes with radius
        double theta = base_angle + 9.0 * r + angle_noise * normal01(rng);

        double x = cx + r * std::cos(theta);
        double y = cy + r * std::sin(theta);

        double v = v0 * r / std::sqrt(r * r + core_radius * core_radius);

        double vx = -v * std::sin(theta);
        double vy =  v * std::cos(theta);

        vx += velocity_noise * normal01(rng);
        vy += velocity_noise * normal01(rng);

        particles.push_back({
            x, y,
            vx, vy,
            1.0
        });
    }

    return particles;
}

std::vector<Particle> make_two_galaxy_merger_initial_condition(
    int n_per_galaxy
) {
    std::vector<Particle> particles;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    std::normal_distribution<double> normal01(0.0, 1.0);

    auto add_galaxy = [&](double cx, double cy,
                          double vx_center, double vy_center,
                          double spin_sign) {
        // central mass
        particles.push_back({
            cx, cy,
            vx_center, vy_center,
            60.0
        });

        double r_min = 0.025;
        double r_max = 0.13;
        double v0 = 0.12;
        double core_radius = 0.04;
        double velocity_noise = 0.004;

        for (int n = 0; n < n_per_galaxy; n++) {
            double u = uniform01(rng);
            double theta = 2.0 * M_PI * uniform01(rng);

            double r = r_min + (r_max - r_min) * std::sqrt(u);

            double x = cx + r * std::cos(theta);
            double y = cy + r * std::sin(theta);

            double v = v0 * r / std::sqrt(r * r + core_radius * core_radius);

            double vx = vx_center + spin_sign * (-v * std::sin(theta));
            double vy = vy_center + spin_sign * ( v * std::cos(theta));

            vx += velocity_noise * normal01(rng);
            vy += velocity_noise * normal01(rng);

            particles.push_back({
                x, y,
                vx, vy,
                1.0
            });
        }
    };

    // Two galaxies moving toward each other
    add_galaxy(
        0.35, 0.50,
        0.045, 0.000,
        1.0
    );

    add_galaxy(
        0.65, 0.50,
       -0.045, 0.000,
       -1.0
    );

    return particles;
}

std::vector<Particle> make_multi_ring_initial_condition(
    int particles_per_ring,
    double cx,
    double cy
) {
    std::vector<Particle> particles;

    particles.push_back({
        cx, cy,
        0.0, 0.0,
        120.0
    });

    std::vector<double> radii = {0.08, 0.12, 0.16, 0.20};
    std::vector<double> speeds = {0.18, 0.16, 0.14, 0.12};

    for (int ring = 0; ring < (int)radii.size(); ring++) {
        double r = radii[ring];
        double speed = speeds[ring];

        for (int k = 0; k < particles_per_ring; k++) {
            double theta = 2.0 * M_PI * k / particles_per_ring;

            double x = cx + r * std::cos(theta);
            double y = cy + r * std::sin(theta);

            double vx = -speed * std::sin(theta);
            double vy =  speed * std::cos(theta);

            particles.push_back({
                x, y,
                vx, vy,
                0.5
            });
        }
    }

    return particles;
}

std::vector<Particle> make_orbit_initial_condition(
    double cx,
    double cy,
    double orbit_radius,
    double orbit_speed
) {
    std::vector<Particle> particles;

    // Central massive particle
    particles.push_back({
        cx, cy,
        0.0, 0.0,
        200.0
    });

    // Orbiting particle
    particles.push_back({
        cx + orbit_radius, cy,
        0.0, orbit_speed,
        1.0
    });

    return particles;
}

int main() {
    const int N = 129;              // Must be 2^k + 1 for multigrid
    const double h = 1.0 / (N - 1);

    const double G = 0.0005;
    const double dt = 0.0005;
    
    const int steps = 12000;
    const int output_every = 30;

    std::string output_file = "results/orbit.csv";

    std::vector<double> rho(N * N, 0.0);
    std::vector<double> phi(N * N, 0.0);
    std::vector<double> ax_grid(N * N, 0.0);
    std::vector<double> ay_grid(N * N, 0.0);

    std::vector<Particle> particles =
    make_orbit_initial_condition(
        0.5,    // center x
        0.5,    // center y
        0.18,   // orbit radius
        0.18    // orbit speed
    );


    std::ofstream file(output_file);

    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << output_file << "\n";
        return 1;
    }

    file << "step,id,x,y,vx,vy\n";

    write_particles(file, particles, 0);

    // Initial mass deposit
    deposit_mass_to_grid(
        particles,
        rho,
        N,
        h,
        G
    );

    // Initial solve from scratch
    fmg_cycle(
        phi,
        rho,
        N,
        2,
        2,
        1.0
    );

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

    std::cout << "Saved " << output_file << "\n";

    return 0;
}