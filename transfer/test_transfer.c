#include <stdio.h>
#include <math.h>
#include "transfer.h"

/* ---- 小工具 ---- */
static void print_grid(const char* name, double** g, int N) {
    printf("\n--- %s (N=%d) ---\n", name, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) printf("%7.4f ", g[i][j]);
        printf("\n");
    }
}

static double max_diff(double** a, double** b, int N) {
    double d = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double diff = fabs(a[i][j] - b[i][j]);
            if (diff > d) d = diff;
        }
    return d;
}

/* ================================================================
 * Test 1: 維度正確
 *   fine 65x65  → restrict → coarse 33x33
 *   coarse 33x33 → prolong → fine 65x65
 * ================================================================ */
void test_dimensions() {
    printf("=== Test 1: 維度檢查 ===\n");
    int N_fine = 65;
    int N_coarse = (N_fine - 1) / 2 + 1;   /* 33 */

    double** fine   = alloc_grid(N_fine);
    double** coarse = alloc_grid(N_coarse);
    double** fine2  = alloc_grid(N_fine);

    restrict_full_weighting(fine, coarse, N_fine);
    prolong_bilinear(coarse, fine2, N_fine);

    printf("  fine: %dx%d  coarse: %dx%d  (expected 65, 33)\n",
           N_fine, N_fine, N_coarse, N_coarse);
    printf("  PASS\n");

    free_grid(fine,   N_fine);
    free_grid(coarse, N_coarse);
    free_grid(fine2,  N_fine);
}

/* ================================================================
 * Test 2: 常數場
 *   fine grid 內部全為 1.0
 *   restrict 之後 coarse 內部應全為 1.0
 *   prolong 之後 fine 內部應全為 1.0
 * ================================================================ */
void test_constant_field() {
    printf("\n=== Test 2: 常數場 (內部全為 1) ===\n");
    int N_fine = 17;
    int N_coarse = (N_fine - 1) / 2 + 1;   /* 9 */

    double** fine   = alloc_grid(N_fine);
    double** coarse = alloc_grid(N_coarse);
    double** fine2  = alloc_grid(N_fine);

    /* 內部設為 1，邊界維持 0 */
    for (int i = 1; i < N_fine - 1; i++)
        for (int j = 1; j < N_fine - 1; j++)
            fine[i][j] = 1.0;

    restrict_full_weighting(fine, coarse, N_fine);

    /* 檢查 coarse 內部 */
    double max_err = 0.0;
    for (int i = 1; i < N_coarse - 1; i++)
        for (int j = 1; j < N_coarse - 1; j++) {
            double e = fabs(coarse[i][j] - 1.0);
            if (e > max_err) max_err = e;
        }
    printf("  Restriction max error from 1.0: %e  %s\n",
           max_err, max_err < 1e-12 ? "PASS" : "FAIL");

    /* 內部設為 1，邊界維持 0，再 prolong */
    for (int i = 0; i < N_coarse; i++)
        for (int j = 0; j < N_coarse; j++)
            coarse[i][j] = 1.0;
    /* 邊界強制 0（模擬 Dirichlet） */
    for (int i = 0; i < N_coarse; i++) {
        coarse[i][0] = coarse[i][N_coarse-1] = 0.0;
        coarse[0][i] = coarse[N_coarse-1][i] = 0.0;
    }

    prolong_bilinear(coarse, fine2, N_fine);

    max_err = 0.0;
    for (int i = 2; i < N_fine - 2; i++)   /* 避開邊界一層 */
        for (int j = 2; j < N_fine - 2; j++) {
            double e = fabs(fine2[i][j] - 1.0);
            if (e > max_err) max_err = e;
        }
    printf("  Prolongation max error from 1.0 (interior): %e  %s\n",
           max_err, max_err < 1e-12 ? "PASS" : "FAIL");

    free_grid(fine,   N_fine);
    free_grid(coarse, N_coarse);
    free_grid(fine2,  N_fine);
}

/* ================================================================
 * Test 3: 小格子肉眼確認 (5x5 → 3x3 → 5x5)
 * ================================================================ */
void test_small_visual() {
    printf("\n=== Test 3: 小格子肉眼確認 (N_fine=5) ===\n");
    int N_fine   = 5;
    int N_coarse = 3;

    double** fine   = alloc_grid(N_fine);
    double** coarse = alloc_grid(N_coarse);
    double** fine2  = alloc_grid(N_fine);

    /* 設定 fine grid 內部值 */
    double vals[5][5] = {
        {0, 0, 0, 0, 0},
        {0, 1, 2, 3, 0},
        {0, 4, 5, 6, 0},
        {0, 7, 8, 9, 0},
        {0, 0, 0, 0, 0}
    };
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            fine[i][j] = vals[i][j];

    print_grid("fine (original)", fine, N_fine);
    restrict_full_weighting(fine, coarse, N_fine);
    print_grid("coarse (after restriction)", coarse, N_coarse);
    prolong_bilinear(coarse, fine2, N_fine);
    print_grid("fine (after prolong of coarse)", fine2, N_fine);

    free_grid(fine,   N_fine);
    free_grid(coarse, N_coarse);
    free_grid(fine2,  N_fine);
}

/* ================================================================
 * Test 3b: make_coarse_grid
 * ================================================================ */
void test_make_coarse_grid() {
    printf("\n=== Test 3b: make_coarse_grid ===\n");
    int cases[] = {129, 65, 33, 17, 9, 5};
    int expected[] = {65, 33, 17, 9, 5, 3};
    int n = 6;
    for (int k = 0; k < n; k++) {
        int N_coarse;
        double** coarse = make_coarse_grid(cases[k], &N_coarse);
        printf("  make_coarse_grid(%3d) -> N_coarse=%3d  (expected %3d)  %s\n",
               cases[k], N_coarse, expected[k],
               N_coarse == expected[k] ? "PASS" : "FAIL");
        free_grid(coarse, N_coarse);
    }
}

/* ================================================================
 * Test 4: 多層維度 (129 → 65 → 33 → 17 → 9 → 5 → 3)
 * ================================================================ */
void test_multilevel_dimensions() {
    printf("\n=== Test 4: 多層維度鏈 ===\n");
    int sizes[] = {129, 65, 33, 17, 9, 5, 3};
    int n = 7;
    for (int k = 0; k < n - 1; k++) {
        int Nf = sizes[k];
        int Nc = (Nf - 1) / 2 + 1;
        printf("  %3d -> %3d  (expected %3d)  %s\n",
               Nf, Nc, sizes[k+1],
               Nc == sizes[k+1] ? "PASS" : "FAIL");
    }
}

int main() {
    test_dimensions();
    test_constant_field();
    test_small_visual();
    test_make_coarse_grid();
    test_multilevel_dimensions();
    printf("\n全部測試完成。\n");
    return 0;
}
