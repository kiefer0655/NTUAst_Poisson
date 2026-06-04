#ifndef TRANSFER_H
#define TRANSFER_H

#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "../utils.h"


static inline std::vector<double> make_coarse_grid(int N_fine, int* N_coarse_out) {
    int N_coarse = (N_fine - 1) / 2 + 1;
    if (N_coarse_out) *N_coarse_out = N_coarse;
    return std::vector<double>(N_coarse * N_coarse, 0.0);
}


static inline void restrict_full_weighting(const std::vector<double>& fine,
                                           std::vector<double>& coarse,
                                           int N_fine)
{
    int N_coarse = (N_fine - 1) / 2 + 1;

    /* Dirichlet BC */
    for (int i = 0; i < N_coarse; i++) {
        coarse[idx(i, 0, N_coarse)]          = 0.0;
        coarse[idx(i, N_coarse-1, N_coarse)] = 0.0;
        coarse[idx(0, i, N_coarse)]          = 0.0;
        coarse[idx(N_coarse-1, i, N_coarse)] = 0.0;
    }

    /* full-weighting */
    for (int ic = 1; ic < N_coarse - 1; ic++) {
        for (int jc = 1; jc < N_coarse - 1; jc++) {
            int if_ = 2 * ic;   /* fine grid row */
            int jf  = 2 * jc;   /* fine grid col */

            coarse[idx(ic, jc, N_coarse)] =
                ( 4.0 * fine[idx(if_  , jf  , N_fine)]
                + 2.0 * fine[idx(if_-1, jf  , N_fine)]
                + 2.0 * fine[idx(if_+1, jf  , N_fine)]
                + 2.0 * fine[idx(if_  , jf-1, N_fine)]
                + 2.0 * fine[idx(if_  , jf+1, N_fine)]
                +       fine[idx(if_-1, jf-1, N_fine)]
                +       fine[idx(if_-1, jf+1, N_fine)]
                +       fine[idx(if_+1, jf-1, N_fine)]
                +       fine[idx(if_+1, jf+1, N_fine)]
                ) / 16.0;
        }
    }
}


static inline void prolong_bilinear(const std::vector<double>& coarse,
                                    std::vector<double>& fine,
                                    int N_fine)
{
    int N_coarse = (N_fine - 1) / 2 + 1;

    for (int i = 0; i < N_fine; i++)
        for (int j = 0; j < N_fine; j++)
            fine[idx(i, j, N_fine)] = 0.0;

    for (int ic = 0; ic < N_coarse; ic++)
        for (int jc = 0; jc < N_coarse; jc++)
            fine[idx(2*ic, 2*jc, N_fine)] = coarse[idx(ic, jc, N_coarse)];

    for (int ic = 0; ic < N_coarse; ic++)
        for (int jc = 0; jc < N_coarse - 1; jc++)
            fine[idx(2*ic, 2*jc+1, N_fine)] =
                0.5 * (coarse[idx(ic, jc, N_coarse)] + coarse[idx(ic, jc+1, N_coarse)]);

    for (int ic = 0; ic < N_coarse - 1; ic++)
        for (int jc = 0; jc < N_coarse; jc++)
            fine[idx(2*ic+1, 2*jc, N_fine)] =
                0.5 * (coarse[idx(ic, jc, N_coarse)] + coarse[idx(ic+1, jc, N_coarse)]);

    for (int ic = 0; ic < N_coarse - 1; ic++)
        for (int jc = 0; jc < N_coarse - 1; jc++)
            fine[idx(2*ic+1, 2*jc+1, N_fine)] =
                0.25 * (coarse[idx(ic  , jc  , N_coarse)] + coarse[idx(ic  , jc+1, N_coarse)]
                      + coarse[idx(ic+1, jc  , N_coarse)] + coarse[idx(ic+1, jc+1, N_coarse)]);


    for (int i = 0; i < N_fine; i++) {
        fine[idx(i, 0, N_fine)]        = 0.0;
        fine[idx(i, N_fine-1, N_fine)] = 0.0;
        fine[idx(0, i, N_fine)]        = 0.0;
        fine[idx(N_fine-1, i, N_fine)] = 0.0;
    }
}

#endif
