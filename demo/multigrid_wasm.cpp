#include <emscripten/bind.h>
#include <vector>
#include <cmath>
#include "../include/multigrid.h"
#include "../include/utils.h"

using namespace emscripten;

class PoissonSolver {
private:
    int N;
    std::vector<double> u;
    std::vector<double> phi;
    double h2;

public:
    PoissonSolver(int n) : N(n), u(n * n, 0.0), phi(n * n, 0.0) {
        h2 = 1.0 / ((N - 1) * (N - 1));
    }

    void reset() {
        std::fill(u.begin(), u.end(), 0.0);
        std::fill(phi.begin(), phi.end(), 0.0);
    }

    void add_charge(int cx, int cy, int radius, double amount) {
        for (int i = std::max(1, cx - radius); i <= std::min(N - 2, cx + radius); i++) {
            for (int j = std::max(1, cy - radius); j <= std::min(N - 2, cy + radius); j++) {
                double dist = std::sqrt((i - cx) * (i - cx) + (j - cy) * (j - cy));
                if (dist <= radius) {
                    // Smooth falloff
                    double falloff = 1.0 - (dist / radius);
                    phi[idx(i, j, N)] += amount * falloff;
                }
            }
        }
    }

    void step() {
        // Run 1 V-Cycle
        v_cycle(u, phi, N, 2, 2, 1.0);
    }

    double get_residual_norm() {
        std::vector<double> r = get_residual_grid(u, phi, N, h2);
        double total = 0.0;
        for (double val : r) {
            total += val * val;
        }
        return std::sqrt(total / (N * N));
    }

    // Return memory view to JS
    val get_u() {
        return val(typed_memory_view(N * N, u.data()));
    }
};

EMSCRIPTEN_BINDINGS(poisson_module) {
    class_<PoissonSolver>("PoissonSolver")
        .constructor<int>()
        .function("reset", &PoissonSolver::reset)
        .function("add_charge", &PoissonSolver::add_charge)
        .function("step", &PoissonSolver::step)
        .function("get_residual_norm", &PoissonSolver::get_residual_norm)
        .function("get_u", &PoissonSolver::get_u);
}
