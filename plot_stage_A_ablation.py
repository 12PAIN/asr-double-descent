"""
Stage A - capacity and subset-ablation plots.

Edit DATASETS below to add or remove training subsets.
All figures are written to OUT_DIR (stage_a_plots/).
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from collections import defaultdict

# -- Configuration -------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "stage_a_abl_plots")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    {"label": "5k", "color": "#e15759", "file": "outputs/outputs_sweep_subset_5k.json"},
    {"label": "50k", "color": "#f4a261", "file": "outputs/outputs_sweep_subset_50k.json"},
    {"label": "150k", "color": "#4e79a7", "file": "outputs/outputs_sweep_subset_150K.json"},
    {"label": "Full", "color": "#59a14f", "file": "outputs/outputs_sweep_full.json"},
]

LAYERS = [1, 2, 3, 4, 6, 8, 12, 16]

# -- Data loading ---------------------------------------------------------------


def load_dataset(rel_path):
    """Load and aggregate runs by num_layers.

    Returns list of model dicts sorted by n_params:
        {L, n_params, steps, metrics: {key: {mean, lo, hi}}, raw_finals: {key: [per-seed]}}
    """
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
        raw_finals = {}
        for key in runs[0]["steps"][0]:
            if key == "step":
                continue
            try:
                arr = np.stack([[s[key] for s in r["steps"]] for r in runs])
                metrics[key] = {"mean": arr.mean(0), "lo": arr.min(0), "hi": arr.max(0)}
                raw_finals[key] = arr[:, -1]
            except (TypeError, KeyError, ValueError):
                pass
        models.append(
            {
                "L": L,
                "n_params": n_params,
                "steps": steps,
                "metrics": metrics,
                "raw_finals": raw_finals,
            }
        )
    return sorted(models, key=lambda m: m["n_params"])


all_data = [(ds["label"], ds["color"], load_dataset(ds["file"])) for ds in DATASETS]

n_subsets = len(all_data)
n_models = len(all_data[0][2])
n_params_arr = np.array([m["n_params"] for m in all_data[0][2]])

# -- Colour palettes ------------------------------------------------------------

_cmap_plasma = plt.colormaps["plasma"]
_pts = np.linspace(0.05, 0.80, n_models)
model_colors = [_cmap_plasma(p) for p in _pts]

_cmap_disc = mcolors.ListedColormap(model_colors)
_norm_disc = mcolors.Normalize(vmin=-0.5, vmax=n_models - 0.5)

_n_joint = n_subsets * n_models
_cmap_turbo = plt.colormaps["turbo"]
joint_colors = [_cmap_turbo(p) for p in np.linspace(0.05, 0.95, _n_joint)]

# -- Helpers --------------------------------------------------------------------


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


def add_model_colorbar(fig, rect=(0.06, 0.04, 0.87, 0.96)):
    """Reserve right margin with tight_layout, then add colorbar in the freed space.
    rect = (left, bottom, right, top) in figure-normalised coords for subplots.
    """
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
    cb.set_ticklabels([f"L{m['L']}" for m in all_data[0][2]], fontsize=8)
    cb.set_label("Layers", fontsize=9)


def luminance_text_color(cmap_name, val, vmin, vmax):
    """Black or white text depending on background luminance at this value."""
    norm = (val - vmin) / (vmax - vmin + 1e-12)
    r, g, b, _ = plt.colormaps[cmap_name](norm)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if lum > 0.45 else "white"


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {name}")
    plt.close(fig)


def subset_legend(ax, title="Dataset"):
    handles = [mpatches.Patch(color=ds["color"], label=ds["label"]) for ds in DATASETS]
    return ax.legend(handles=handles, fontsize=9, frameon=True, title=title)


# ==============================================================================
# 1. WER capacity sweep
# ==============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("WER vs. parameter count", fontsize=13, fontweight="bold")

for label, color, models in all_data:
    vw = np.array([final(m, "val_wer") for m in models])
    vwl = np.array([final_lo(m, "val_wer") for m in models])
    vwh = np.array([final_hi(m, "val_wer") for m in models])
    tw = np.array([final(m, "train_wer") for m in models])
    tsw = np.array([final(m, "test_wer") for m in models])

    ax.fill_between(n_params_arr, vwl, vwh, color=color, alpha=0.12)
    ax.plot(n_params_arr, vw, "o-", color=color, linewidth=2.0, markersize=7, label=label, zorder=3)
    ax.plot(n_params_arr, tw, "s--", color=color, linewidth=1.4, markersize=5, alpha=0.65, zorder=2)
    ax.scatter(
        n_params_arr,
        tsw,
        marker="D",
        s=55,
        facecolors="none",
        edgecolors=color,
        linewidths=1.5,
        zorder=5,
    )

style_ax(ax, "Parameters", "WER", log_x=True, wer=True)

leg1 = subset_legend(ax)
ax.add_artist(leg1)
style_handles = [
    mlines.Line2D(
        [],
        [],
        color="gray",
        linestyle="-",
        linewidth=2,
        marker="o",
        markersize=6,
        label="Validation WER",
    ),
    mlines.Line2D(
        [],
        [],
        color="gray",
        linestyle="--",
        linewidth=1.4,
        marker="s",
        markersize=5,
        alpha=0.65,
        label="Train WER",
    ),
    mlines.Line2D(
        [],
        [],
        color="gray",
        linestyle="",
        marker="D",
        markersize=6,
        markerfacecolor="none",
        markeredgewidth=1.5,
        label="Test WER",
    ),
]
ax.legend(handles=style_handles, fontsize=9, frameon=True, loc="upper center")

fig.tight_layout()
save(fig, "wer_capacity.png")


# ==============================================================================
# 2. Loss capacity sweep
# ==============================================================================

fig, (ax, ax_log) = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Loss vs. parameter count", fontsize=13, fontweight="bold")

for label, color, models in all_data:
    vl = np.array([final(m, "val_loss") for m in models])
    vll = np.array([final_lo(m, "val_loss") for m in models])
    vlh = np.array([final_hi(m, "val_loss") for m in models])
    tl = np.array([final(m, "train_eval_loss") for m in models])
    tsl = np.array([final(m, "test_loss") for m in models])

    for a in (ax, ax_log):
        a.fill_between(n_params_arr, vll, vlh, color=color, alpha=0.12)
        a.plot(n_params_arr, vl, "o-", color=color, linewidth=2.0, markersize=7, label=label, zorder=3)
        a.plot(n_params_arr, tl, "s--", color=color, linewidth=1.4, markersize=5, alpha=0.65, zorder=2)
        a.scatter(
            n_params_arr,
            tsl,
            marker="D",
            s=55,
            facecolors="none",
            edgecolors=color,
            linewidths=1.5,
            zorder=5,
        )

style_ax(ax, "Parameters", "CTC loss", log_x=True, log_y=False)
style_ax(ax_log, "Parameters", "CTC loss (log scale)", log_x=True, log_y=True)
ax.set_title("Linear scale", fontsize=10)
ax_log.set_title("Log scale", fontsize=10)

style_handles = [
    mlines.Line2D(
        [],
        [],
        color="gray",
        linestyle="-",
        linewidth=2,
        marker="o",
        markersize=6,
        label="Validation Loss",
    ),
    mlines.Line2D(
        [],
        [],
        color="gray",
        linestyle="--",
        linewidth=1.4,
        marker="s",
        markersize=5,
        alpha=0.65,
        label="Train Loss",
    ),
    mlines.Line2D(
        [],
        [],
        color="gray",
        linestyle="",
        marker="D",
        markersize=6,
        markerfacecolor="none",
        markeredgewidth=1.5,
        label="Test Loss",
    ),
]
leg1 = subset_legend(ax)
ax.add_artist(leg1)
ax.legend(handles=style_handles, fontsize=9, frameon=True, loc="upper center")
# ax_log.legend(handles=style_handles, fontsize=9, frameon=True, loc="upper center")

fig.tight_layout()
save(fig, "loss_capacity.png")


# ==============================================================================
# 3+4. Epochwise grid  (n_subsets rows × 2 cols: WER | Loss)
#      Row labels show dataset; column headers show metric.
# ==============================================================================

fig, axes = plt.subplots(n_subsets, 2, figsize=(13, 3.8 * n_subsets), squeeze=False)
fig.suptitle("Training curves by dataset size", fontsize=13, fontweight="bold")

for row, (label, _color, models) in enumerate(all_data):
    for col, (train_key, val_key, test_key, ylabel, log_y, is_wer) in enumerate(
        [
            ("train_wer", "val_wer", "test_wer", "WER", False, True),
            ("train_eval_loss", "val_loss", "test_loss", "CTC loss", False, False),
        ]
    ):
        ax = axes[row][col]
        for i, m in enumerate(models):
            c = model_colors[i]
            steps = m["steps"]
            tr = m["metrics"][train_key]["mean"]
            vl = m["metrics"][val_key]["mean"]
            vl_lo = m["metrics"][val_key]["lo"]
            vl_hi = m["metrics"][val_key]["hi"]
            ts_val = m["metrics"][test_key]["mean"][-1]

            ax.fill_between(steps, vl_lo, vl_hi, color=c, alpha=0.12)
            ax.plot(steps, tr, color=c, linewidth=1.3, linestyle="--", alpha=0.65)
            ax.plot(steps, vl, color=c, linewidth=1.8)
            ax.scatter(
                [steps[-1]],
                [ts_val],
                marker="D",
                s=38,
                facecolors="none",
                edgecolors=c,
                linewidths=1.2,
                zorder=6,
            )

        # Column header on top row only
        if row == 0:
            ax.set_title("WER" if is_wer else "Loss", fontsize=11, fontweight="bold")

        ax.set_xlabel("Steps", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(step_fmt))
        if is_wer:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    # Dataset label on the left of each row
    axes[row][0].annotate(
        label,
        xy=(0, 0.5),
        xycoords="axes fraction",
        xytext=(-0.32, 0.5),
        textcoords="axes fraction",
        fontsize=11,
        fontweight="bold",
        rotation=90,
        va="center",
        ha="center",
        annotation_clip=False,
    )

fig.text(
    0.5,
    0.01,
    "Solid: Validation.  Dashed: Train.  Open diamond: Test at final step.",
    ha="center",
    fontsize=8.5,
    color="dimgray",
)
add_model_colorbar(fig, rect=(0.08, 0.04, 0.87, 0.97))
save(fig, "epochwise_grid.png")


# ==============================================================================
# 5. Generalization gap vs. parameters  (twin axis: Train Loss)
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Generalization gap vs. parameter count", fontsize=13, fontweight="bold")
ax2 = ax.twinx()

for label, color, models in all_data:
    gg = np.array([final(m, "generalization_gap") for m in models])
    ggl = np.array([final_lo(m, "generalization_gap") for m in models])
    ggh = np.array([final_hi(m, "generalization_gap") for m in models])
    tl = np.array([final(m, "train_eval_loss") for m in models])

    ax.fill_between(n_params_arr, ggl, ggh, color=color, alpha=0.12)
    ax.plot(n_params_arr, gg, "o-", color=color, linewidth=2.0, markersize=7, label=label, zorder=3)
    ax2.plot(n_params_arr, tl, color=color, linewidth=1.4, linestyle=":", alpha=0.6)

ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(param_fmt))
ax.set_xlabel("Parameters", fontsize=11)
ax.set_ylabel("Generalization gap", fontsize=11)
ax2.set_yscale("log")
ax2.set_ylabel("Train loss", fontsize=11, color="gray")
ax2.tick_params(axis="y", labelcolor="gray", labelsize=9)
ax.tick_params(labelsize=9)
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9, frameon=True, title="Dataset")
fig.text(0.98, 0.02, "Dotted lines: Train loss (right axis)", ha="right", fontsize=8, color="gray")
fig.tight_layout()
save(fig, "gengap_capacity.png")


# ==============================================================================
# 6. Sharpness heatmap  (X: model, Y: dataset, value: final sharpness)
# ==============================================================================

subset_labels = [ds["label"] for ds in DATASETS]
model_labels = [f"L{m['L']}\n{param_fmt(m['n_params'], None)}" for m in all_data[0][2]]

sharp_mat = np.array(
    [[final(m, "sharpness_proxy") for m in models] for _label, _color, models in all_data]
)

fig, ax = plt.subplots(figsize=(max(8, n_models * 1.2), max(3, n_subsets * 1.0)))
fig.suptitle("Sharpness at final step", fontsize=13, fontweight="bold")

im = ax.imshow(sharp_mat, cmap="YlOrBr", aspect="auto")
fig.colorbar(im, ax=ax, pad=0.02, label="Sharpness")

ax.set_xticks(range(n_models))
ax.set_xticklabels(model_labels, fontsize=9)
ax.set_yticks(range(n_subsets))
ax.set_yticklabels(subset_labels, fontsize=10)
ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Dataset", fontsize=11)

vmin, vmax = sharp_mat.min(), sharp_mat.max()
for r in range(n_subsets):
    for c in range(n_models):
        val = sharp_mat[r, c]
        tc = luminance_text_color("YlOrBr", val, vmin, vmax)
        ax.text(c, r, f"{val:.2e}", ha="center", va="center", fontsize=8, color=tc)

fig.tight_layout()
save(fig, "sharpness.png")


# ==============================================================================
# 6b. Sharpness grid  (time-series, one panel per dataset)
# ==============================================================================

ncols_sh = min(n_subsets, 4)
nrows_sh = (n_subsets + ncols_sh - 1) // ncols_sh

fig, axes = plt.subplots(nrows_sh, ncols_sh, figsize=(5 * ncols_sh, 4 * nrows_sh), squeeze=False)
fig.suptitle("Sharpness over training steps", fontsize=13, fontweight="bold")

for idx, (label, _color, models) in enumerate(all_data):
    ax = axes[idx // ncols_sh][idx % ncols_sh]
    for i, m in enumerate(models):
        c = model_colors[i]
        mean = m["metrics"]["sharpness_proxy"]["mean"]
        lo = m["metrics"]["sharpness_proxy"]["lo"]
        hi = m["metrics"]["sharpness_proxy"]["hi"]
        ax.fill_between(m["steps"], lo, hi, color=c, alpha=0.18)
        ax.plot(m["steps"], mean, color=c, linewidth=1.8)

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlabel("Steps", fontsize=9)
    ax.set_ylabel("Sharpness", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(step_fmt))
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

for idx in range(n_subsets, nrows_sh * ncols_sh):
    axes[idx // ncols_sh][idx % ncols_sh].set_visible(False)

add_model_colorbar(fig, rect=(0.06, 0.05, 0.87, 0.95))
save(fig, "sharpness_grid.png")


# ==============================================================================
# 7-10. Heatmaps
# ==============================================================================

HEATMAPS = [
    ("test_wer", "Test WER", "heatmap_test_wer.png", "RdYlGn_r", True),
    ("test_loss", "Test Loss", "heatmap_test_loss.png", "viridis_r", False),
    ("generalization_gap", "Generalization gap", "heatmap_gengap.png", "YlOrRd", False),
    ("weight_norm_sum", "Weight norm", "heatmap_weight_norm.png", "plasma", False),
]

for metric, title, fname, cmap_name, is_wer in HEATMAPS:
    mat = np.array([[final(m, metric) for m in models] for _label, _color, models in all_data])

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.2), max(3, n_subsets * 1.0)))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    im = ax.imshow(mat, cmap=cmap_name, aspect="auto")
    fig.colorbar(im, ax=ax, pad=0.02, label=title)

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_yticks(range(n_subsets))
    ax.set_yticklabels(subset_labels, fontsize=10)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Dataset", fontsize=11)

    vmin, vmax = mat.min(), mat.max()
    for r in range(n_subsets):
        for c in range(n_models):
            val = mat[r, c]
            txt = f"{val:.1%}" if is_wer else f"{val:.3f}"
            tc = luminance_text_color(cmap_name, val, vmin, vmax)
            ax.text(c, r, txt, ha="center", va="center", fontsize=8, color=tc)

    fig.tight_layout()
    save(fig, fname)


# ==============================================================================
# 11+12. Epochwise all  (all subsets x models, one panel per metric)
#        Subset labels placed at the start of each group.
# ==============================================================================

fig, (ax_wer, ax_loss) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Training curves, all datasets and model sizes", fontsize=13, fontweight="bold")

# Collect final-step medians per subset for right-side annotations
_end_medians = {}
for s_idx, (label, ds_color, models) in enumerate(all_data):
    _end_medians[s_idx] = {
        "wer": np.median([m["metrics"]["val_wer"]["mean"][-1] for m in models]),
        "loss": np.median([m["metrics"]["val_loss"]["mean"][-1] for m in models]),
        "color": ds_color,
        "label": label,
        "steps": models[0]["steps"],
    }

for s_idx, (_label, _color, models) in enumerate(all_data):
    for m_idx, m in enumerate(models):
        c = joint_colors[s_idx * n_models + m_idx]
        steps = m["steps"]

        for ax, train_key, val_key in [
            (ax_wer, "train_wer", "val_wer"),
            (ax_loss, "train_eval_loss", "val_loss"),
        ]:
            ax.plot(
                steps,
                m["metrics"][train_key]["mean"],
                color=c,
                linewidth=0.9,
                linestyle="--",
                alpha=0.55,
            )
            ax.plot(steps, m["metrics"][val_key]["mean"], color=c, linewidth=1.3, alpha=0.85)

for ax, ylabel, is_wer, log_y in [
    (ax_wer, "WER", True, False),
    (ax_loss, "CTC loss", False, False),
]:
    ax.set_xlabel("Steps", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(step_fmt))
    if is_wer:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

ax_wer.set_title("WER", fontsize=11, fontweight="bold")
ax_loss.set_title("Loss", fontsize=11, fontweight="bold")

_sm = plt.cm.ScalarMappable(
    cmap=_cmap_turbo,
    norm=mcolors.Normalize(0, _n_joint),
)
_sm.set_array([])
fig.tight_layout(rect=[0, 0.04, 0.88, 1])
cbar_ax_joint = fig.add_axes([0.895, 0.08, 0.018, 0.86])
cb = fig.colorbar(_sm, cax=cbar_ax_joint)
tick_pos = [s * n_models + n_models / 2 for s in range(n_subsets)]
cb.set_ticks(tick_pos)
cb.set_ticklabels([ds["label"] for ds in DATASETS], fontsize=9)
cb.set_label("Dataset", fontsize=9)

fig.text(
    0.5, 0.01, "Solid: Validation.  Dashed: Train.", ha="center", fontsize=8.5, color="dimgray"
)
save(fig, "epochwise_all.png")


# ==============================================================================
# 13. Generalization gap over steps
# ==============================================================================

ncols_gg = min(n_subsets, 4)
nrows_gg = (n_subsets + ncols_gg - 1) // ncols_gg

fig, axes = plt.subplots(nrows_gg, ncols_gg, figsize=(5 * ncols_gg, 4 * nrows_gg), squeeze=False)
fig.suptitle("Generalization gap over training steps", fontsize=13, fontweight="bold")

for idx, (label, _color, models) in enumerate(all_data):
    ax = axes[idx // ncols_gg][idx % ncols_gg]
    for i, m in enumerate(models):
        c = model_colors[i]
        mean = m["metrics"]["generalization_gap"]["mean"]
        lo = m["metrics"]["generalization_gap"]["lo"]
        hi = m["metrics"]["generalization_gap"]["hi"]
        ax.fill_between(m["steps"], lo, hi, color=c, alpha=0.18)
        ax.plot(m["steps"], mean, color=c, linewidth=1.8)

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlabel("Steps", fontsize=9)
    ax.set_ylabel("Generalization gap", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(step_fmt))
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

for idx in range(n_subsets, nrows_gg * ncols_gg):
    axes[idx // ncols_gg][idx % ncols_gg].set_visible(False)

add_model_colorbar(fig, rect=(0.06, 0.05, 0.87, 0.95))
save(fig, "gengap_curves.png")
