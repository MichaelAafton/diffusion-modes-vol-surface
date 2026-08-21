import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from src.data_pipeline import (
    ReparamConfig, load_optionmetrics, preprocess_options_data,
)
from pathlib import Path

wide = ReparamConfig(z_min=-5.0, z_max=5.0)
out = preprocess_options_data(load_optionmetrics(), wide)
s30 = out[out["days"] == 30].sort_values(["date", "z"])

groups = dict(list(s30.groupby("date")))
dates = sorted(groups)
frames = dates[::5]

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-2.6, 1.6)
ax.set_ylim(0, float(s30["implied_vol"].max()) * 1.05)
ax.set_xlabel("z  =  log(K/S) / (σ_ATM √T)")
ax.set_ylabel("implied vol")
ax.grid(alpha=0.3)

line, = ax.plot([], [], "o-", lw=1.5, ms=4)
label = ax.text(0.02, 0.94, "", transform=ax.transAxes)

def update(d):
    g = groups[d]
    line.set_data(g["z"].values, g["implied_vol"].values)
    label.set_text(d.strftime("%Y-%m-%d"))
    return line, label

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

anim = FuncAnimation(fig, update, frames=frames, interval=40)
anim.save(FIGURES / "smile_30d.gif", writer=PillowWriter(fps=25))