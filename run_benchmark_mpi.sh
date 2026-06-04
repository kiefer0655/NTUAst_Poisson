#!/bin/bash
# run_benchmark_mpi.sh

echo "========================================"
echo " Starting Automated MPI Benchmark Sweep"
echo "========================================"

# Ensure results folder exists and clear old MPI results
mkdir -p results
rm -f results/results_mpi.csv

for np in 1 2 4 8 16 32 64 128; do
    echo "----------------------------------------"
    echo "Running benchmark with $np MPI processes..."
    mpirun -np $np ./benchmark_mpi
done

echo "----------------------------------------"
echo "MPI Benchmarking Complete! Data saved to results/results_mpi.csv"
