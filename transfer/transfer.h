#ifndef TRANSFER_H
#define TRANSFER_H

#include <stdlib.h>
#include <stdio.h>



static inline double** alloc_grid(int N) {
    double** g = (double**)malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        g[i] = (double*)calloc(N, sizeof(double));
    }
    return g;
}


static inline void free_grid(double** g, int N) {
    for (int i = 0; i < N; i++) free(g[i]);
    free(g);
}


static inline double** make_coarse_grid(int N_fine, int* N_coarse_out) {
    int N_coarse = (N_fine - 1) / 2 + 1;
    if (N_coarse_out) *N_coarse_out = N_coarse;
    return alloc_grid(N_coarse);
}


static inline void restrict_full_weighting(double** fine,
                                           double** coarse,
                                           int N_fine)
{
    int N_coarse = (N_fine - 1) / 2 + 1;

    /* Dirichlet BC */
    for (int i = 0; i < N_coarse; i++) {
        coarse[i][0]          = 0.0;
        coarse[i][N_coarse-1] = 0.0;
        coarse[0][i]          = 0.0;
        coarse[N_coarse-1][i] = 0.0;
    }

    /* full-weighting */
    for (int ic = 1; ic < N_coarse - 1; ic++) {
        for (int jc = 1; jc < N_coarse - 1; jc++) {
            int if_ = 2 * ic;   /* fine grid row */
            int jf  = 2 * jc;   /* fine grid col */

            coarse[ic][jc] =
                ( 4.0 * fine[if_  ][jf  ]
                + 2.0 * fine[if_-1][jf  ]
                + 2.0 * fine[if_+1][jf  ]
                + 2.0 * fine[if_  ][jf-1]
                + 2.0 * fine[if_  ][jf+1]
                +       fine[if_-1][jf-1]
                +       fine[if_-1][jf+1]
                +       fine[if_+1][jf-1]
                +       fine[if_+1][jf+1]
                ) / 16.0;
        }
    }
}


static inline void prolong_bilinear(double** coarse,
                                    double** fine,
                                    int N_fine)
{
    int N_coarse = (N_fine - 1) / 2 + 1;

    for (int i = 0; i < N_fine; i++)
        for (int j = 0; j < N_fine; j++)
            fine[i][j] = 0.0;

    for (int ic = 0; ic < N_coarse; ic++)
        for (int jc = 0; jc < N_coarse; jc++)
            fine[2*ic][2*jc] = coarse[ic][jc];

    for (int ic = 0; ic < N_coarse; ic++)
        for (int jc = 0; jc < N_coarse - 1; jc++)
            fine[2*ic][2*jc+1] =
                0.5 * (coarse[ic][jc] + coarse[ic][jc+1]);

    for (int ic = 0; ic < N_coarse - 1; ic++)
        for (int jc = 0; jc < N_coarse; jc++)
            fine[2*ic+1][2*jc] =
                0.5 * (coarse[ic][jc] + coarse[ic+1][jc]);

    for (int ic = 0; ic < N_coarse - 1; ic++)
        for (int jc = 0; jc < N_coarse - 1; jc++)
            fine[2*ic+1][2*jc+1] =
                0.25 * (coarse[ic  ][jc  ] + coarse[ic  ][jc+1]
                      + coarse[ic+1][jc  ] + coarse[ic+1][jc+1]);


    for (int i = 0; i < N_fine; i++) {
        fine[i][0]        = 0.0;
        fine[i][N_fine-1] = 0.0;
        fine[0][i]        = 0.0;
        fine[N_fine-1][i] = 0.0;
    }
}

#endif /* TRANSFER_H */
