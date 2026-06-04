# Changes Made for Phase 1: Unification & Unit Testing

## 1. Refactored `transfer/transfer.h` (Person B's Work)
* **What I did:** Converted the grid storage from C-style 2D pointers (`double**`) to modern C++ 1D vectors (`std::vector<double>`).
* **Why:** To make it fully compatible with Person A's `smoother.h` and `Relax.cpp`, which use `std::vector<double>`. This 1D contiguous memory layout is also essential for performance, OpenMP caching, and the upcoming CUDA (GPU) and MPI implementations.
* **Respecting the Author:** I preserved Person B's exact mathematical logic, coefficients, loop structures, and comments. The only thing that changed was `grid[i][j]` becoming `grid[idx(i, j, N)]`, using the helper function from `utils.h`.

## 2. Added `test_smoother.cpp` (Person A's Test)
* **What I did:** Wrote a standard `main()` test with `cassert`.
* **Why:** To rigorously prove that Person A's SOR smoother works. It initializes a grid with a known exact solution (`-2*pi*pi*sin*sin`) and runs the smoother for 2000 iterations, then asserts that the error norm is significantly reduced.

## 3. Added `test_transfer.cpp` (Person B's Test)
* **What I did:** Wrote a standard `main()` test with `cassert`.
* **Why:** To verify that the newly refactored `transfer.h` code compiles and mathematically correctly moves values from the fine grid down to the coarse grid (restriction) and back up (prolongation).

## How to Compile and Run Tests for Commit Verification:
You can verify the changes work by running:
```bash
g++ -fopenmp test_smoother.cpp -o test_smoother
./test_smoother

g++ -fopenmp test_transfer.cpp -o test_transfer
./test_transfer
```

Once you test these and commit, we can move on to **Phase 2: Core Solver Integration (V-cycle & W-cycle)**.
