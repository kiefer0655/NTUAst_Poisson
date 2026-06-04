

#include "transfer.h" 
in the head of the code to use the finctions.


Here are the Function Overview, look for more detailed information in "transfer_functions"


Function						Purpose

alloc_grid(N)						Allocate a zeroed N×N double array
free_grid(g, N)						Free memory allocated by alloc_grid
make_coarse_grid(N_fine, &N_coarse)			Allocate the matching coarse grid and return its size
restrict_full_weighting(fine, coarse, N_fine)		Transfer residual from fine grid to coarse grid
prolong_bilinear(coarse, fine, N_fine)			Interpolate correction from coarse grid back to fine grid


All the functions can be tested in test_transfer.