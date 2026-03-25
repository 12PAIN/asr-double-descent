"""
Stage A: capacity sweep plots (full training set).
Input:  outputs/outputs_sweep_full.json
Output: stage_a_plots/
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
OUT_DIR = os.path.join(BASE, "stage_a_plots")
os.makedirs(OUT_DIR, exist_ok=True)

LAYERS = [1, 2, 3, 4, 6, 8, 12, 16]
FULL_COLOR = "#59a14f"

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
        steps = np.array([s["step"] for s in runs[0]["steps"]])
        metrics = {}
        for key in runs[0]["steps"][0]:
            if key == "step":
                continue
            try:
                arr = np.stack([[s[key] for s in r["steps"]] for r in runs])
                metrics[key] = {"mean": arr.mean(0), "lo": arr.min(0), "hi": arr.max(0)}
            except (TypeError, KeyError, ValueError):
                pass
        models.append({"L": L, "n_params": n_params, "steps": steps, "metrics": metrics})
    return sorted(models, key=lambda m: m["n_params"])


models = load_dataset("outputs/outputs_sweep_full.json")
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


def final_lo(m, key):
    return m["metrics"][key]["lo"][-1]


def final_hi(m, key):
    return m["metrics"][key]["hi"][-1]


def param_fmt(x, _):
    return f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}k"


def step_fmt(x, _):
    return f"{int(x/1000)}k" if x >= 1000 else str(int(x))


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


# ==============================================================================
# 1. WER capacity sweep
# ==============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("WER capacity sweep", fontsize=13, fontweight="bold")

vw = np.array([final(m, "val_wer") for m in models])
vwl = np.array([final_lo(m, "val_wer") for m in models])
vwh = np.array([final_hi(m, "val_wer") for m in models])
tw = np.array([final(m, "train_wer") for m in models])
tsw = np.array([final(m, "test_wer") for m in models])

ax.fill_between(n_params_arr, vwl, vwh, color=FULL_COLOR, alpha=0.12)
ax.plot(n_params_arr, vw, "o-", color=FULL_COLOR, linewidth=2.0, markersize=7,
        label="val WER", zorder=3)
ax.plot(n_params_arr, tw, "s--", color=FULL_COLOR, linewidth=1.4, markersize=5,
        alpha=0.65, label="train WER", zorder=2)
ax.scatter(n_params_arr, tsw, marker="D", s=55,
           facecolors="none", edgecolors=FULL_COLOR, linewidths=1.5,
           label="test WER", zorder=5)

style_ax(ax, "parameters", "WER", log_x=True, wer=True)
ax.legend(fontsize=9, frameon=True)
fig.tight_layout()
save(fig, "wer_capacity.png")


# ==============================================================================
# 2. Loss capacity sweep
# ==============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Loss capacity sweep", fontsize=13, fontweight="bold")

vl = np.array([final(m, "val_loss") for m in models])
vll = np.array([final_lo(m, "val_loss") for m in models])
vlh = np.array([final_hi(m, "val_loss") for m in models])
tl = np.array([final(m, "train_eval_loss") for m in models])
tsl = np.array([final(m, "test_loss") for m in models])

ax.fill_between(n_params_arr, vll, vlh, color=FULL_COLOR, alpha=0.12)
ax.plot(n_params_arr, vl, "o-", color=FULL_COLOR, linewidth=2.0, markersize=7,
        label="val loss", zorder=3)
ax.plot(n_params_arr, tl, "s--", color=FULL_COLOR, linewidth=1.4, markersize=5,
        alpha=0.65, label="train loss", zorder=2)
ax.scatter(n_params_arr, tsl, marker="D", s=55,
           facecolors="none", edgecolors=FULL_COLOR, linewidths=1.5,
           label="test loss", zorder=5)

style_ax(ax, "parameters", "CTC loss", log_x=True)
ax.legend(fontsize=9, frameon=True)
fig.tight_layout()
save(fig, "loss_vs_capacity.png")


# ==============================================================================
# 3. Generalization gap vs. capacity
# ==============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Generalization gap vs. capacity", fontsize=13, fontweight="bold")

gg = np.array([final(m, "generalization_gap") for m in models])
ggl = np.array([final_lo(m, "generalization_gap") for m in models])
ggh = np.array([final_hi(m, "generalization_gap") for m in models])

ax.fill_between(n_params_arr, ggl, ggh, color=FULL_COLOR, alpha=0.12)
ax.plot(n_params_arr, gg, "o-", color=FULL_COLOR, linewidth=2.0, markersize=7, zorder=3)
ax.scatter(n_params_arr, gg,
           c=model_colors, s=75, zorder=4, edgecolors="white", linewidths=0.8)

style_ax(ax, "parameters", r"$\mathcal{L}_\mathrm{test} - \mathcal{L}_\mathrm{train}$",
         log_x=True)
fig.tight_layout()
save(fig, "gen_gap_vs_capacity.png")


# ==============================================================================
# 4. Sharpness proxy (relative, over steps)
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Sharpness proxy", fontsize=13, fontweight="bold")

for i, m in enumerate(models):
    c = model_colors[i]
    mean = m["metrics"]["sharpness_rel"]["mean"]
    lo = m["metrics"]["sharpness_rel"]["lo"]
    hi = m["metrics"]["sharpness_rel"]["hi"]
    ax.fill_between(m["steps"], lo, hi, color=c, alpha=0.18)
    ax.plot(m["steps"], mean, color=c, linewidth=1.8)

style_ax(ax, "steps", "rel. sharpness")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(step_fmt))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

add_model_colorbar(fig)
save(fig, "sharpness_proxy.png")


# ==============================================================================
# 5. Weight norm (over steps)
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Weight norm", fontsize=13, fontweight="bold")

for i, m in enumerate(models):
    c = model_colors[i]
    mean = m["metrics"]["weight_norm_sum"]["mean"]
    lo = m["metrics"]["weight_norm_sum"]["lo"]
    hi = m["metrics"]["weight_norm_sum"]["hi"]
    ax.fill_between(m["steps"], lo, hi, color=c, alpha=0.18)
    ax.plot(m["steps"], mean, color=c, linewidth=1.8)

style_ax(ax, "steps", r"$\|\theta\|_W$")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(step_fmt))

add_model_colorbar(fig)
save(fig, "weight_norm_sum.png")


# ==============================================================================
# 6. Epochwise loss (train dashed, val solid, no test)
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Epochwise loss", fontsize=13, fontweight="bold")

for i, m in enumerate(models):
    c = model_colors[i]
    steps = m["steps"]
    tr = m["metrics"]["train_eval_loss"]["mean"]
    vl = m["metrics"]["val_loss"]["mean"]
    vl_lo = m["metrics"]["val_loss"]["lo"]
    vl_hi = m["metrics"]["val_loss"]["hi"]

    ax.fill_between(steps, vl_lo, vl_hi, color=c, alpha=0.12)
    ax.plot(steps, tr, color=c, linewidth=1.3, linestyle="--", alpha=0.65)
    ax.plot(steps, vl, color=c, linewidth=1.8)

style_ax(ax, "steps", "CTC loss")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(step_fmt))
fig.text(0.5, 0.01, "Solid: val loss.  Dashed: train loss.",
         ha="center", fontsize=8.5, color="dimgray")

add_model_colorbar(fig, rect=(0.06, 0.06, 0.90, 0.96))
save(fig, "train_eval_loss.png")


# ==============================================================================
# 7. Epochwise WER (train dashed, val solid, no test)
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Epochwise WER", fontsize=13, fontweight="bold")

for i, m in enumerate(models):
    c = model_colors[i]
    steps = m["steps"]
    tr = m["metrics"]["train_wer"]["mean"]
    vl = m["metrics"]["val_wer"]["mean"]
    vl_lo = m["metrics"]["val_wer"]["lo"]
    vl_hi = m["metrics"]["val_wer"]["hi"]

    ax.fill_between(steps, vl_lo, vl_hi, color=c, alpha=0.12)
    ax.plot(steps, tr, color=c, linewidth=1.3, linestyle="--", alpha=0.65)
    ax.plot(steps, vl, color=c, linewidth=1.8)

style_ax(ax, "steps", "WER", wer=True)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(step_fmt))
fig.text(0.5, 0.01, "Solid: val WER.  Dashed: train WER.",
         ha="center", fontsize=8.5, color="dimgray")

add_model_colorbar(fig, rect=(0.06, 0.06, 0.90, 0.96))
save(fig, "val_wer.png")
