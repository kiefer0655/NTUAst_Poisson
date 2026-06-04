## Functions Overview

| Function | Purpose |
| --- | --- |
| `alloc_grid(N)` | Allocate a zeroed $N \times N$ double array |
| `free_grid(g, N)` | Free memory allocated by `alloc_grid` |
| `make_coarse_grid(N_fine, &N_coarse)` | Allocate the matching coarse grid and return its size |
| `restrict_full_weighting(fine, coarse, N_fine)` | Transfer residual from fine grid to coarse grid |
| `prolong_bilinear(coarse, fine, N_fine)` | Interpolate correction from coarse grid back to fine grid |

### Grid Size Convention

All grid sizes must follow the form $N = 2^k + 1$, which ensures clean halving at every level:

| Level | Grid Size |
| --- | --- |
| Fine ($k=7$) | $129 \times 129$ |
| Level 2 | $65 \times 65$ |
| Level 3 | $33 \times 33$ |
| Level 4 | $17 \times 17$ |
| Level 5 | $9 \times 9$ |
| Level 6 | $5 \times 5$ |
| Coarsest ($k=1$) | $3 \times 3$ |

The coarse grid size is always calculated as:

$$N_{\text{coarse}} = \frac{N_{\text{fine}} - 1}{2} + 1$$

---

## `restrict_full_weighting`

### Purpose

Maps the residual from the fine grid to the coarse grid. This reduces the problem size so that low-frequency error can be corrected efficiently on the coarser level.

### Method: Full-Weighting Stencil

Each coarse-grid interior point is computed as a weighted average of 9 surrounding fine-grid points:

$$\begin{matrix}
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1
\end{matrix} \times \frac{1}{16}$$

The fine-grid point directly below the coarse point gets a weight of $4/16$, its 4 edge neighbors get $2/16$ each, and the 4 diagonal neighbors get $1/16$ each. The weights sum to 1.

### Index Mapping

Coarse index `(ic, jc)` corresponds to fine index `(2*ic, 2*jc)`. The stencil reads from:
`fine[2ic-1..2ic+1][2jc-1..2jc+1]`

### Boundary Condition

All boundary points of the coarse grid are set to `0.0`, consistent with the homogeneous Dirichlet boundary condition on the error equation.

### Function Signature

```c
void restrict_full_weighting(double** fine, double** coarse, int N_fine);

```

---

## `prolong_bilinear`

### Purpose

Maps the coarse-grid error correction back to the fine grid so it can be applied as a correction to the fine-grid solution. This is the adjoint of restriction.

### Method: Bilinear Interpolation

Three cases are handled separately:

| Case | Fine Index | Formula |
| --- | --- | --- |
| **Coarse point copied directly** | row even, col even | `fine[2ic][2jc] = coarse[ic][jc]` |
| **Horizontal edge midpoint** | row even, col odd | `0.5 * (coarse[ic][jc] + coarse[ic][jc+1])` |
| **Vertical edge midpoint** | row odd, col even | `0.5 * (coarse[ic][jc] + coarse[ic+1][jc])` |
| **Cell center point** | row odd, col odd | `0.25 * (sum of 4 surrounding coarse corners)` |

### Boundary Condition

After interpolation, all boundary points of the fine grid are set to `0.0`.

### Function Signature

```c
void prolong_bilinear(double** coarse, double** fine, int N_fine);

```

---

## `make_coarse_grid`

A convenience function for Person C to use inside the recursive V/W-cycle. It allocates a correctly sized coarse grid and also outputs `N_coarse` through a pointer, so the caller does not need to manually compute the size.

```c
int N_coarse;
double** coarse_res = make_coarse_grid(N_fine, &N_coarse);

restrict_full_weighting(residual, coarse_res, N_fine);

// ... recursive call ...

free_grid(coarse_res, N_coarse);

```

---

## Unit Tests

All tests are located in `test_transfer.c` and compiled using:

```bash
gcc test_transfer.c -lm

```

| Test | What it checks |
| --- | --- |
| **Test 1: Dimension check** | Fine $65 \times 65 \rightarrow$ coarse $33 \times 33$, prolong back to $65 \times 65$ |
| **Test 2: Constant field** | Interior all-1 field is preserved through restrict and prolong |
| **Test 3: Visual (5×5)** | Manual inspection of a $5 \times 5 \rightarrow 3 \times 3 \rightarrow 5 \times 5$ round-trip |
| **Test 3b: make_coarse_grid** | Correct coarse size returned for 129, 65, 33, 17, 9, 5 |
| **Test 4: Full level chain** | $129 \rightarrow 65 \rightarrow 33 \rightarrow 17 \rightarrow 9 \rightarrow 5 \rightarrow 3$ dimension chain correct |
