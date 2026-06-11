import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

scenarios = [
    (r"D:\NTU\大二\計天\NTUAst_Poisson\demo\Thermal\results\heat_single_source.csv",     "Single Central Source"),
    (r"D:\NTU\大二\計天\NTUAst_Poisson\demo\Thermal\results\heat_multi_source.csv",      "Multiple Sources & Sinks"),
    (r"D:\NTU\大二\計天\NTUAst_Poisson\demo\Thermal\results\heat_boundary_gradient.csv", "Boundary Temperature Gradient"),
    (r"D:\NTU\大二\計天\NTUAst_Poisson\demo\Thermal\results\heat_ring_source.csv",       "Ring-Shaped Source"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.patch.set_facecolor("#0f0f1a")
axes = axes.flatten()

for ax, (filepath, title) in zip(axes, scenarios):
    if not os.path.exists(filepath):
        ax.text(0.5, 0.5, f"Missing:\n{filepath}",
                ha='center', va='center', color='white', fontsize=9)
        ax.set_facecolor("#0f0f1a")
        ax.set_title(title, color='white')
        continue

    df = pd.read_csv(filepath)
    N = int(df['i'].max()) + 1
    U = df['u'].values.reshape(N, N)

    im = ax.imshow(
        U.T, origin='lower', cmap='inferno',
        extent=[0, 1, 0, 1], interpolation='bilinear'
    )

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    ax.contour(x, y, U.T, levels=12, colors='white', alpha=0.25, linewidths=0.6)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=7)
    cbar.set_label('Temperature u', color='white', fontsize=8)

    ax.set_facecolor("#0f0f1a")
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('x', color='#aaaaaa', fontsize=8)
    ax.set_ylabel('y', color='#aaaaaa', fontsize=8)
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

fig.suptitle(
    "Heat Conduction via Poisson Solver  ∇²u = ρ",
    color='white', fontsize=14, fontweight='bold', y=0.98
)
plt.tight_layout()

os.makedirs("results/images", exist_ok=True)
out = "results/images/heat_conduction.png"
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved " + out)
plt.show()
