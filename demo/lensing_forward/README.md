# Moving Lens Deep-Field Demo

This demo uses the thin-lens approximation with a moving lens mass in front of a fixed deep-field background image.

For each animation frame, the script builds a lens convergence map `kappa`, solves

```text
∇²ψ = 2κ
```

with the repo's CPU/OpenMP full multigrid solver through `solve_canvas_cpu`, then computes

```text
α = ∇ψ
β = θ - α(θ)
```

The deep-field image is sampled at `β`, so the moving lens distorts the background.

## Run

From the repository root:

```bash
make solve_canvas_cpu
python3 demo/lensing_forward/moving_source_lensing.py
```

## Input

```text
demo/lensing_forward/background/deep field.png
```

## Outputs

```text
demo/lensing_forward/results/moving_lens_deep_field.mp4
demo/lensing_forward/results/moving_lens_deep_field_last_frame.png
demo/lensing_forward/results/psi.csv
demo/lensing_forward/results/psi_frames/psi_000.csv ...
```

Note: this demo is vibe coded with Codex