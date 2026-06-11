from pathlib import Path
import subprocess

import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


N = 257
L = 3.0
N_FRAMES = 100
DEFLECTION_STRENGTH = 0.85
INCLUDE_SUBSTRUCTURE_CLUMP = False


def elliptical_gaussian(X, Y, x0, y0, sigma_x, sigma_y, angle, amplitude=1.0):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    dx = X - x0
    dy = Y - y0
    xp = cos_a * dx + sin_a * dy
    yp = -sin_a * dx + cos_a * dy
    return amplitude * np.exp(-0.5 * ((xp / sigma_x) ** 2 + (yp / sigma_y) ** 2))


def make_kappa(X, Y, lens_x):
    kappa_main = elliptical_gaussian(
        X,
        Y,
        lens_x,
        0.0,
        sigma_x=0.72,
        sigma_y=0.42,
        angle=0.35,
        amplitude=1.0,
    )
    if not INCLUDE_SUBSTRUCTURE_CLUMP:
        return kappa_main

    kappa_clump = elliptical_gaussian(
        X,
        Y,
        lens_x + 0.48,
        -0.35,
        sigma_x=0.16,
        sigma_y=0.16,
        angle=0.0,
        amplitude=0.35,
    )
    return kappa_main + kappa_clump


def sample_background(background, beta_x, beta_y):
    height, width, channels = background.shape
    col = (beta_x + L) / (2.0 * L) * (width - 1)
    row = (beta_y + L) / (2.0 * L) * (height - 1)
    inside = (col >= 0.0) & (col <= width - 1) & (row >= 0.0) & (row <= height - 1)

    col = np.clip(col, 0.0, width - 1)
    row = np.clip(row, 0.0, height - 1)

    col0 = np.floor(col).astype(int)
    row0 = np.floor(row).astype(int)
    col1 = np.clip(col0 + 1, 0, width - 1)
    row1 = np.clip(row0 + 1, 0, height - 1)

    wx = col - col0
    wy = row - row0

    top_left = background[row0, col0]
    top_right = background[row0, col1]
    bottom_left = background[row1, col0]
    bottom_right = background[row1, col1]

    sampled = (
        (1.0 - wx)[..., None] * (1.0 - wy)[..., None] * top_left
        + wx[..., None] * (1.0 - wy)[..., None] * top_right
        + (1.0 - wx)[..., None] * wy[..., None] * bottom_left
        + wx[..., None] * wy[..., None] * bottom_right
    )
    sampled[~inside] = 0.0
    return sampled


def solve_lensing_potential(repo_root, solver, N, kappa, source_bin, psi_bin):
    poisson_source = 2.0 * kappa
    poisson_source.astype(np.float64).tofile(source_bin)

    subprocess.run(
        [
            "./solve_canvas_cpu",
            str(N),
            str(source_bin.relative_to(repo_root)),
            str(psi_bin.relative_to(repo_root)),
        ],
        cwd=repo_root,
        check=True,
    )

    psi = np.fromfile(psi_bin, dtype=np.float64)
    if psi.size != N * N:
        raise RuntimeError(f"Expected {N * N} values in {psi_bin}, found {psi.size}")
    return psi.reshape((N, N))


def main():
    repo_root = Path(__file__).resolve().parents[2]
    demo_dir = repo_root / "demo" / "lensing_forward"
    background_path = demo_dir / "background" / "deep field.png"
    results_dir = demo_dir / "results"
    psi_csv_dir = results_dir / "psi_frames"
    results_dir.mkdir(parents=True, exist_ok=True)
    psi_csv_dir.mkdir(parents=True, exist_ok=True)

    solver = repo_root / "solve_canvas_cpu"
    if not solver.exists():
        raise FileNotFoundError(
            "Missing solve_canvas_cpu. Build it first with: make solve_canvas_cpu"
        )
    if not background_path.exists():
        raise FileNotFoundError(f"Missing background image: {background_path}")

    background = mpimg.imread(background_path)
    if background.ndim != 3:
        raise RuntimeError("Background image must be RGB or RGBA")
    background = np.flipud(background[..., :3]).astype(np.float64)
    if background.max() > 1.0:
        background /= 255.0

    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    unlensed_background = sample_background(background, X, Y)

    source_bin = results_dir / "lens_source.bin"
    psi_bin = results_dir / "psi.bin"
    movie_path = results_dir / "moving_lens_deep_field.mp4"
    last_frame_path = results_dir / "moving_lens_deep_field_last_frame.png"
    lens_positions = np.linspace(-0.95, 0.95, N_FRAMES)

    frames = []
    for frame, lens_x in enumerate(lens_positions):
        print(f"Solving frame {frame + 1}/{N_FRAMES}: lens x = {lens_x:.3f}")
        kappa = make_kappa(X, Y, lens_x)
        psi = solve_lensing_potential(repo_root, solver, N, kappa, source_bin, psi_bin)

        psi_csv = psi_csv_dir / f"psi_{frame:03d}.csv"
        np.savetxt(psi_csv, psi, delimiter=",")

        alpha_x, alpha_y = np.gradient(psi, dx, dx)
        alpha_strength = np.sqrt(alpha_x * alpha_x + alpha_y * alpha_y)
        norm = np.percentile(alpha_strength, 99.0)
        if norm <= 0.0:
            raise RuntimeError("Cannot normalize deflection because alpha is zero everywhere")

        alpha_x = DEFLECTION_STRENGTH * alpha_x / norm
        alpha_y = DEFLECTION_STRENGTH * alpha_y / norm
        beta_x = X - alpha_x
        beta_y = Y - alpha_y
        lensed = sample_background(background, beta_x, beta_y)
        frames.append((kappa, lensed, lens_x))

    final_psi_csv = results_dir / "psi.csv"
    np.savetxt(final_psi_csv, psi, delimiter=",") # type: ignore
    print(f"Wrote final lensing potential CSV: {final_psi_csv}")
    print(f"Wrote per-frame potential CSVs: {psi_csv_dir}")
    print(f"Wrote latest Poisson source binary: {source_bin}")
    print(f"Wrote latest potential binary: {psi_bin}")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)

    kappa_image = axes[0].imshow(
        frames[0][0].T,
        origin="lower",
        extent=(-L, L, -L, L),
        cmap="magma",
        vmin=0.0,
        vmax=max(np.max(frame[0]) for frame in frames),
    )
    fig.colorbar(kappa_image, ax=axes[0], label="kappa")
    axes[0].set_title("Moving lens mass")

    background_image = axes[1].imshow(
        np.swapaxes(unlensed_background, 0, 1),
        origin="lower",
        extent=(-L, L, -L, L),
    )
    axes[1].set_title("Deep field background")

    lensed_image = axes[2].imshow(
        np.swapaxes(frames[0][1], 0, 1),
        origin="lower",
        extent=(-L, L, -L, L),
    )
    axes[2].set_title("Lensed deep field")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def update(frame):
        kappa, lensed, lens_x = frames[frame]
        kappa_image.set_data(kappa.T)
        lensed_image.set_data(np.swapaxes(lensed, 0, 1))
        fig.suptitle(f"moving lens x = {lens_x:.2f}")
        return kappa_image, background_image, lensed_image

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=N_FRAMES,
        interval=100,
        blit=False,
    )

    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "Matplotlib cannot find an ffmpeg writer, so MP4 export is unavailable."
        )

    writer = animation.FFMpegWriter(fps=12, bitrate=2200)
    anim.save(movie_path, writer=writer, dpi=160)
    print(f"Wrote MP4: {movie_path}")

    update(N_FRAMES - 1)
    fig.savefig(last_frame_path, dpi=200)
    print(f"Wrote last frame PNG: {last_frame_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
