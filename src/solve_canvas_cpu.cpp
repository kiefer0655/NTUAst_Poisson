#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/multigrid.h"

bool is_power_of_two_plus_one(int N) {
    if (N < 3) {
        return false;
    }

    int value = N - 1;
    return (value & (value - 1)) == 0;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./solve_canvas_cpu N input_rho.bin output_u.bin\n";
        return 1;
    }

    int N = 0;
    try {
        N = std::stoi(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Invalid N: " << argv[1] << "\n";
        return 1;
    }

    if (!is_power_of_two_plus_one(N)) {
        std::cerr << "Invalid N = " << N << ". N must be of the form 2^k + 1.\n";
        return 1;
    }

    std::string input_file = argv[2];
    std::string output_file = argv[3];
    std::vector<double> phi(N * N, 0.0);

    std::ifstream in(input_file, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open input file: " << input_file << "\n";
        return 1;
    }

    std::streamsize expected_bytes = static_cast<std::streamsize>(N) *
                                     static_cast<std::streamsize>(N) *
                                     static_cast<std::streamsize>(sizeof(double));
    in.read(reinterpret_cast<char*>(phi.data()), expected_bytes);
    if (in.gcount() != expected_bytes) {
        std::cerr << "Failed to read " << expected_bytes << " bytes from "
                  << input_file << ". Read " << in.gcount() << " bytes.\n";
        return 1;
    }

    std::vector<double> u(N * N, 0.0);
    fmg_cycle(u, phi, N, 2, 2, 1.0);

    std::ofstream out(output_file, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output file: " << output_file << "\n";
        return 1;
    }

    out.write(
        reinterpret_cast<const char*>(u.data()),
        expected_bytes
    );
    if (!out) {
        std::cerr << "Failed to write output file: " << output_file << "\n";
        return 1;
    }

    return 0;
}
