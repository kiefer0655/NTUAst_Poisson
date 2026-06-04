#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "../include/utils.h"
#include "../include/multigrid.h"
#include "../include/multigrid_cuda.cuh"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./solve_canvas <N> <input_rho.bin> <output_u.bin>\n";
        return 1;
    }
    int N = std::stoi(argv[1]);
    std::string in_file = argv[2];
    std::string out_file = argv[3];

    std::vector<double> phi(N * N, 0.0);
    std::ifstream in(in_file, std::ios::binary);
    if (in) {
        in.read(reinterpret_cast<char*>(phi.data()), N * N * sizeof(double));
        in.close();
    } else {
        std::cerr << "Failed to read input file.\n";
        return 1;
    }

    std::vector<double> u(N * N, 0.0);
    
    // Solve instantly using Full Multigrid on the GPU!
    fmg_cycle_cuda_run(u, phi, N, 2, 2, 1.0);

    std::ofstream out(out_file, std::ios::binary);
    if (out) {
        out.write(reinterpret_cast<const char*>(u.data()), N * N * sizeof(double));
        out.close();
    } else {
        std::cerr << "Failed to write output file.\n";
        return 1;
    }
    
    return 0;
}
