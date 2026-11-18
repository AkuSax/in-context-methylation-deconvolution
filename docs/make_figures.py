"""Build docs/hero.png and docs/missing_type.gif from saved GSE110554 development results.

Reads only CSVs under deconv/results/. Run from the repository root:
    python docs/make_figures.py
"""
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "deconv" / "results"
OUT = Path(__file__).resolve().parent
PER_SIDE = 25  # markers per side, the setting quoted in the README


def kept_rmse(path, method, regime):
    df = pd.read_csv(path)
    row = df[(df.per_side == PER_SIDE) & (df.regime == regime) & (df.subset == "all") & (df.method == method)]
    return float(row.kept_rmse_pts.iloc[0])


v5 = RES / "dev_gse110554_gpu_v5" / "summary.csv"
v6 = RES / "real_check_gpu_v6" / "dev_nocohort" / "summary.csv"
METHODS = [("RPC (classical)", v5, "RPC", "#8c8c8c"), ("Model v5", v5, "model", "#4c78a8"), ("Model v6", v6, "model", "#1b5e8c")]
SCENES = [("Complete reference", "clean"), ("One cell type missing", "drop_one")]
vals = {(m[0], s[1]): kept_rmse(m[1], m[2], s[1]) for m in METHODS for s in SCENES}
YMAX = 12.5

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})


def draw(ax, progress):
    ax.clear()
    width = 0.24
    for j, (label, _, _, color) in enumerate(METHODS):
        xs = [i + (j - 1) * width for i in range(len(SCENES))]
        hs = [vals[(label, s[1])] * progress for s in SCENES]
        ax.bar(xs, hs, width * 0.92, color=color, label=label)
        if progress >= 1:
            for x, h in zip(xs, hs):
                ax.text(x, h + 0.25, f"{h:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(SCENES)))
    ax.set_xticklabels([s[0] for s in SCENES], fontsize=12)
    ax.set_ylim(0, YMAX)
    ax.set_ylabel("RMSE on retained cell types\n(percentage points, lower is better)")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.set_title("GSE110554 development mixtures (12 mixtures, 25 markers per side)", fontsize=11, loc="left")


def fig_ax():
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=130)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.12)
    return fig, ax


fig, ax = fig_ax()
draw(ax, 1.0)
fig.text(0.12, 0.01, "Development results on a small constructed set; not evidence of generalization.", fontsize=8, color="#555")
fig.savefig(OUT / "hero.png")

frames = []
steps = list(range(0, 11)) + [10] * 8
for k in steps:
    draw(ax, k / 10)
    fig.canvas.draw()
    frames.append(__import__("numpy").asarray(fig.canvas.buffer_rgba())[..., :3].copy())
imageio.mimsave(OUT / "missing_type.gif", frames, duration=0.07, loop=0)
print({k: round(v, 2) for k, v in vals.items()})
