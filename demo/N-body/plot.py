import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

INPUT_FILE = "results/nbody_particles.csv"

df = pd.read_csv(INPUT_FILE)

steps = sorted(df["step"].unique())

fig, ax = plt.subplots(figsize=(6, 6))

def update(frame):
    ax.clear()

    d = df[df["step"] == frame]

    ax.scatter(d["x"], d["y"], s=20)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"N-body demo, step = {frame}")

ani = FuncAnimation(
    fig,
    update, # type: ignore
    frames=steps,
    interval=30,
    repeat=True
)

plt.show()