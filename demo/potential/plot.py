import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prefix",
        nargs="?",
        default="results/charge_demo",
        help="File prefix, for example results/charge_demo"
    )
    args = parser.parse_args()

    rho = np.loadtxt(args.prefix + "_rho.csv", delimiter=",")
    V = np.loadtxt(args.prefix + "_potential.csv", delimiter=",")

    N = rho.shape[0]
    h = 1.0 / (N - 1)

    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)

    # V[i, j]: i is x-index, j is y-index
    dVdx, dVdy = np.gradient(V, h, h)

    Ex = -dVdx
    Ey = -dVdy

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # Charge density plot
    rho_image = axes[0].imshow(
        rho.T,
        origin="lower",
        extent=(0, 1, 0, 1),
        aspect="equal"
    )
    fig.colorbar(rho_image, ax=axes[0], label="charge density rho")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Input charge distribution")

    # Potential + electric field plot
    potential_image = axes[1].imshow(
        V.T,
        origin="lower",
        extent=(0, 1, 0, 1),
        aspect="equal"
    )
    fig.colorbar(potential_image, ax=axes[1], label="potential V")

    levels = np.linspace(np.min(V), np.max(V), 20)

    # contour also expects rows = y, columns = x
    axes[1].contour(
        x,
        y,
        V.T,
        levels=levels,
        linewidths=0.7
    )

    # Quiver expects arrays with shape = (len(y), len(x)), so transpose fields.
    stride = max(1, N // 32)
    x_quiver = x[::stride]
    y_quiver = y[::stride]
    X_quiver, Y_quiver = np.meshgrid(x_quiver, y_quiver)
    Ex_quiver = Ex.T[::stride, ::stride]
    Ey_quiver = Ey.T[::stride, ::stride]
    field_strength = np.sqrt(Ex_quiver * Ex_quiver + Ey_quiver * Ey_quiver)
    max_strength = np.max(field_strength)
    min_visible_strength = 0.05 * max_strength
    visible = field_strength >= min_visible_strength
    Ex_direction = np.divide(
        Ex_quiver,
        field_strength,
        out=np.zeros_like(Ex_quiver),
        where=field_strength > 0.0
    )
    Ey_direction = np.divide(
        Ey_quiver,
        field_strength,
        out=np.zeros_like(Ey_quiver),
        where=field_strength > 0.0
    )

    vector_plot = axes[1].quiver(
        X_quiver[visible],
        Y_quiver[visible],
        Ex_direction[visible],
        Ey_direction[visible],
        field_strength[visible],
        cmap="Reds",
        angles="xy",
        scale_units="xy",
        scale=35,
        width=0.003,
        pivot="mid"
    )
    fig.colorbar(vector_plot, ax=axes[1], label="field strength |E|")

    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Potential and electric field")
    fig.savefig(args.prefix + "_combined.png", dpi=200)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
