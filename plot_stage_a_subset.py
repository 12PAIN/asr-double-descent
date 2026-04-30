"""
Stage A: capacity sweep plots (25k-sample subset).
Input:  results/sweep_subset_25k.json
Output: stage_a_subset_plots/
X-axis: epoch number (not gradient steps).
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from collections import defaultdict

# -- Configuration -------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "stage_a_subset_plots")
os.makedirs(OUT_DIR, exist_ok=True)

LAYERS = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24]
SUBSET_COLOR = "#4e79a7"

# -- Data loading --------------------------------------------------------------


def load_dataset(rel_path):
    path = os.path.join(BASE, rel_path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    groups = defaultdict(list)
    for run in raw:
        groups[run["model_config"]["num_layers"]].append(run)

    models = []
    for L in LAYERS:
        runs = groups.get(L)
        if not runs:
            continue
        n_params = runs[0]["n_params"]
        epochs = np.array([e["epoch"] for e in runs[0]["epochs"]])
        metrics = {}
        for key in runs[0]["epochs"][0]:
            if key in ("epoch", "step", "eval_timestamp"):
                continue
            try:
                arr = np.stack([[e[key] for e in r["epochs"]] for r in runs])
                metrics[key] = {"mean": arr.mean(0), "lo": arr.min(0), "hi": arr.max(0)}
            except (TypeError, KeyError, ValueError):
                pass
        models.append({"L": L, "n_params": n_params, "epochs": epochs, "metrics": metrics})
    return sorted(models, key=lambda m: m["n_params"])


models = load_dataset("results/sweep_subset_25k.json")
n_models = len(models)
n_params_arr = np.array([m["n_params"] for m in models])

# -- Colour palettes -----------------------------------------------------------

_cmap_plasma = plt.colormaps["plasma"]
_pts = np.linspace(0.05, 0.80, n_models)
model_colors = [_cmap_plasma(p) for p in _pts]

_cmap_disc = mcolors.ListedColormap(model_colors)
_norm_disc = mcolors.Normalize(vmin=-0.5, vmax=n_models - 0.5)

# -- Helpers -------------------------------------------------------------------


def final(m, key):
    return m["metrics"][key]["mean"][-1]


def param_fmt(x, _):
    return f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}k"


def style_ax(ax, xlabel, ylabel, log_x=False, log_y=False, wer=False):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(param_fmt))
    if log_y:
        ax.set_yscale("log")
    if wer:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def add_model_colorbar(fig, rect=(0.06, 0.04, 0.90, 0.96)):
    fig.tight_layout(rect=list(rect))
    l, b, r, t = rect
    cbar_ax = fig.add_axes([r + 0.015, b + 0.02, 0.018, t - b - 0.04])
    sm = plt.cm.ScalarMappable(cmap=_cmap_disc, norm=_norm_disc)
    sm.set_array([])
    cb = fig.colorbar(
        sm,
        cax=cbar_ax,
        boundaries=np.arange(n_models + 1) - 0.5,
        ticks=np.arange(n_models),
    )
    cb.set_ticklabels([f"L{m['L']}" for m in models], fontsize=8)
    cb.set_label("Layers", fontsize=9)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {name}")
    plt.close(fig)


def final_lo(m, key):
    return m["metrics"][key]["lo"][-1]


def final_hi(m, key):
    return m["metrics"][key]["hi"][-1]


# ==============================================================================
# 1. WER capacity sweep (final epoch)
# ==============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("WER capacity sweep (25k subset)", fontsize=13, fontweight="bold")

vw = np.array([final(m, "val_wer") for m in models])
vwl = np.array([final_lo(m, "val_wer") for m in models])
vwh = np.array([final_hi(m, "val_wer") for m in models])
tw = np.array([final(m, "train_wer") for m in models])
tsw = np.array([final(m, "test_wer") for m in models])

ax.fill_between(n_params_arr, vwl, vwh, color=SUBSET_COLOR, alpha=0.12)
ax.plot(n_params_arr, vw, "o-", color=SUBSET_COLOR, linewidth=2.0, markersize=7,
        label="val WER", zorder=3)
ax.plot(n_params_arr, tw, "s--", color=SUBSET_COLOR, linewidth=1.4, markersize=5,
        alpha=0.65, label="train WER", zorder=2)
ax.scatter(n_params_arr, tsw, marker="D", s=55,
           facecolors="none", edgecolors=SUBSET_COLOR, linewidths=1.5,
           label="test WER", zorder=5)

style_ax(ax, "parameters", "WER", log_x=True, wer=True)
ax.legend(fontsize=9, frameon=True)
fig.tight_layout()
save(fig, "wer_capacity.png")


# ==============================================================================
# 2. Loss capacity sweep (final epoch)
# ==============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Loss capacity sweep (25k subset)", fontsize=13, fontweight="bold")

vl = np.array([final(m, "val_loss") for m in models])
vll = np.array([final_lo(m, "val_loss") for m in models])
vlh = np.array([final_hi(m, "val_loss") for m in models])
tl = np.array([final(m, "train_eval_loss") for m in models])
tsl = np.array([final(m, "test_loss") for m in models])

ax.fill_between(n_params_arr, vll, vlh, color=SUBSET_COLOR, alpha=0.12)
ax.plot(n_params_arr, vl, "o-", color=SUBSET_COLOR, linewidth=2.0, markersize=7,
        label="val loss", zorder=3)
ax.plot(n_params_arr, tl, "s--", color=SUBSET_COLOR, linewidth=1.4, markersize=5,
        alpha=0.65, label="train loss", zorder=2)
ax.scatter(n_params_arr, tsl, marker="D", s=55,
           facecolors="none", edgecolors=SUBSET_COLOR, linewidths=1.5,
           label="test loss", zorder=5)

style_ax(ax, "parameters", "CTC loss", log_x=True)
ax.legend(fontsize=9, frameon=True)
fig.tight_layout()
save(fig, "loss_vs_capacity.png")


# ==============================================================================
# 3. Generalization gap vs. capacity (final epoch)
# ==============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Generalization gap vs. capacity (25k subset)", fontsize=13, fontweight="bold")

gg = np.array([final(m, "generalization_gap") for m in models])
ggl = np.array([final_lo(m, "generalization_gap") for m in models])
ggh = np.array([final_hi(m, "generalization_gap") for m in models])

ax.fill_between(n_params_arr, ggl, ggh, color=SUBSET_COLOR, alpha=0.12)
ax.plot(n_params_arr, gg, "o-", color=SUBSET_COLOR, linewidth=2.0, markersize=7, zorder=3)
ax.scatter(n_params_arr, gg,
           c=model_colors, s=75, zorder=4, edgecolors="white", linewidths=0.8)

style_ax(ax, "parameters", r"$\mathcal{L}_\mathrm{test} - \mathcal{L}_\mathrm{train}$",
         log_x=True)
fig.tight_layout()
save(fig, "gen_gap_vs_capacity.png")


# ==============================================================================
# 4. Sharpness proxy (over epochs)
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Sharpness proxy (25k subset)", fontsize=13, fontweight="bold")

for i, m in enumerate(models):
    c = model_colors[i]
    met = m["metrics"]["sharpness_rel"]
    ax.fill_between(m["epochs"], met["lo"], met["hi"], color=c, alpha=0.18)
    ax.plot(m["epochs"], met["mean"], color=c, linewidth=1.8)

style_ax(ax, "epoch", "rel. sharpness")
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

add_model_colorbar(fig)
save(fig, "sharpness_proxy.png")


# ==============================================================================
# 5. Weight norm (over epochs)
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Weight norm (25k subset)", fontsize=13, fontweight="bold")

for i, m in enumerate(models):
    c = model_colors[i]
    met = m["metrics"]["weight_norm_sum"]
    ax.fill_between(m["epochs"], met["lo"], met["hi"], color=c, alpha=0.18)
    ax.plot(m["epochs"], met["mean"], color=c, linewidth=1.8)

style_ax(ax, "epoch", r"$\|\theta\|_W$")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

add_model_colorbar(fig)
save(fig, "weight_norm_sum.png")


# ==============================================================================
# 6. Loss over epochs (train dashed, val solid)
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Loss over epochs (25k subset)", fontsize=13, fontweight="bold")

for i, m in enumerate(models):
    c = model_colors[i]
    epochs = m["epochs"]
    vl_met = m["metrics"]["val_loss"]
    ax.fill_between(epochs, vl_met["lo"], vl_met["hi"], color=c, alpha=0.12)
    ax.plot(epochs, m["metrics"]["train_eval_loss"]["mean"],
            color=c, linewidth=1.3, linestyle="--", alpha=0.65)
    ax.plot(epochs, vl_met["mean"], color=c, linewidth=1.8)

style_ax(ax, "epoch", "CTC loss")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
fig.text(0.5, 0.01, "Solid: val loss.  Dashed: train loss.",
         ha="center", fontsize=8.5, color="dimgray")

add_model_colorbar(fig, rect=(0.06, 0.06, 0.90, 0.96))
save(fig, "train_eval_loss.png")


# ==============================================================================
# 7. WER over epochs (train dashed, val solid)
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("WER over epochs (25k subset)", fontsize=13, fontweight="bold")

for i, m in enumerate(models):
    c = model_colors[i]
    epochs = m["epochs"]
    vw_met = m["metrics"]["val_wer"]
    ax.fill_between(epochs, vw_met["lo"], vw_met["hi"], color=c, alpha=0.12)
    ax.plot(epochs, m["metrics"]["train_wer"]["mean"],
            color=c, linewidth=1.3, linestyle="--", alpha=0.65)
    ax.plot(epochs, vw_met["mean"], color=c, linewidth=1.8)

style_ax(ax, "epoch", "WER", wer=True)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
fig.text(0.5, 0.01, "Solid: val WER.  Dashed: train WER.",
         ha="center", fontsize=8.5, color="dimgray")

add_model_colorbar(fig, rect=(0.06, 0.06, 0.90, 0.96))
save(fig, "val_wer.png")
