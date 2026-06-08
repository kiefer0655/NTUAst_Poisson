import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


INPUT_FILE = "results/orbit.csv"

GIF_FILE = "results/orbit.gif"
MP4_FILE = "results/orbit.mp4"

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

# Save as GIF
ani.save(GIF_FILE, writer="pillow", fps=30)
print(f"Saved {GIF_FILE}")

# Save as MP4
ani.save(MP4_FILE, writer="ffmpeg", fps=30, dpi=150)
print(f"Saved {MP4_FILE}")

plt.show()