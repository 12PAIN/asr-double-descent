"""
Stage A: capacity sweep plots.
Input:  outputs_sweep_full_fixed.json  (multiple seeds per model config)
Output: stage_a_plots/
  capacity_sweep.png
  train_eval_loss.png / test_loss.png
  train_wer.png / test_wer.png
  generalization_gap.png / sharpness_proxy.png / weight_norm_sum.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
from collections import defaultdict

# - Config -----------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
IN_FILE = os.path.join(BASE, "outputs/outputs_sweep_full.json")
OUT_DIR = os.path.join(BASE, "stage_a_plots")
os.makedirs(OUT_DIR, exist_ok=True)

INTERP_THRESHOLD = 0.05  # train WER < 5%

# - Load & group by model config (average/band over seeds) ----------

raw = json.load(open(IN_FILE))

groups = defaultdict(list)
for r in raw:
    mc = r["model_config"]
    key = (mc["d_model"], mc["num_layers"], r["n_params"])
    groups[key].append(r)

METRIC_KEYS = [
    "train_eval_loss",
    "val_loss",
    "test_loss",
    "train_wer",
    "val_wer",
    "test_wer",
    "generalization_gap",
    "weight_norm_sum",
    "sharpness_proxy",
]

models = []
for (d, L, n_params), runs in sorted(groups.items(), key=lambda x: x[0][2]):
    steps = np.array([s["step"] for s in runs[0]["steps"]])
    m = {
        "n_params": n_params,
        "d": d,
        "L": L,
        "label": f"L={L} ({n_params/1e6:.0f}M)",
        "short": f"L{L:02d}",
        "steps": steps,
        "mean": {},
        "lo": {},  # min across seeds
        "hi": {},  # max across seeds
    }
    for key in METRIC_KEYS:
        arr = np.array([[s[key] for s in r["steps"]] for r in runs])
        m["mean"][key] = arr.mean(0)
        m["lo"][key] = arr.min(0)
        m["hi"][key] = arr.max(0)
    models.append(m)

n_models = len(models)

# - Colormap: sample plasma from 0.05->0.80, avoids the near-white bright end -
#    Small model = dark purple, large model = warm orange. Both clearly visible.

_palette_pts = np.linspace(0.05, 0.80, n_models)
_cmap_base = cm.get_cmap("plasma")
for i, m in enumerate(models):
    m["color"] = _cmap_base(_palette_pts[i])

# Discrete colormap for the colorbar (same sample points)
_cmap_discrete = mcolors.ListedColormap([_cmap_base(x) for x in _palette_pts])

# - Helpers ----------------------------------


def step_fmt(x, _):
    return f"{int(x/1000)}k" if x >= 1000 else str(int(x))


def style_ax(ax, ylabel="", log=False, threshold_line=None):
    ax.set_xlabel("Steps", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(step_fmt))
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if log:
        ax.set_yscale("log")
    if threshold_line is not None:
        ax.axhline(
            threshold_line,
            color="black",
            linewidth=1.2,
            linestyle="--",
            alpha=0.6,
            label=f"WER = {threshold_line:.0%}",
        )


def add_colorbar(fig, ax):
    sm = plt.cm.ScalarMappable(
        cmap=_cmap_discrete, norm=mcolors.BoundaryNorm(np.arange(n_models + 1), n_models)
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_ticks(np.arange(n_models) + 0.5)
    cb.set_ticklabels([m["short"] for m in models], fontsize=8)
    cb.set_label("Layers (d=512)", fontsize=9)


def find_interp_step(m):
    vals = m["mean"]["train_wer"]
    idxs = np.where(vals < INTERP_THRESHOLD)[0]
    return int(m["steps"][idxs[0]]) if len(idxs) else None


# - 1. Capacity sweep (final step only) ---------
# Val WER is used for the training curve; test WER is shown as the final
# held-out result (single point at 100k, not used to pick checkpoints).

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Stage A — Capacity Sweep at Step 100k (d=512, mean ± seed range)",
             fontsize=13, fontweight="bold")

n_params_arr = np.array([m["n_params"] for m in models])

tw_mean,  tw_lo,  tw_hi  = [], [], []
vw_mean,  vw_lo,  vw_hi  = [], [], []
tsw_mean, tsw_lo, tsw_hi = [], [], []

for m in models:
    tw_mean.append(m["mean"]["train_wer"][-1])
    tw_lo.append(m["lo"]["train_wer"][-1]);  tw_hi.append(m["hi"]["train_wer"][-1])
    vw_mean.append(m["mean"]["val_wer"][-1])
    vw_lo.append(m["lo"]["val_wer"][-1]);    vw_hi.append(m["hi"]["val_wer"][-1])
    tsw_mean.append(m["mean"]["test_wer"][-1])
    tsw_lo.append(m["lo"]["test_wer"][-1]);  tsw_hi.append(m["hi"]["test_wer"][-1])

tw_mean,  tw_lo,  tw_hi  = map(np.array, (tw_mean,  tw_lo,  tw_hi))
vw_mean,  vw_lo,  vw_hi  = map(np.array, (vw_mean,  vw_lo,  vw_hi))
tsw_mean, tsw_lo, tsw_hi = map(np.array, (tsw_mean, tsw_lo, tsw_hi))

ax.fill_between(n_params_arr, vw_lo,  vw_hi,  color="#f4a261", alpha=0.15)
ax.fill_between(n_params_arr, tw_lo,  tw_hi,  color="#4e79a7", alpha=0.15)
ax.plot(n_params_arr, vw_mean, "o-",  color="#f4a261", linewidth=2,   markersize=8,
        label="Val WER (dev-clean)",  zorder=3)
ax.plot(n_params_arr, tw_mean, "s--", color="#4e79a7", linewidth=1.5, markersize=7,
        label="Train WER", zorder=3, alpha=0.85)

# test-clean shown as final held-out markers only (no connecting line)
ax.scatter(n_params_arr, tsw_mean,
           c=[m["color"] for m in models], s=90, zorder=5,
           edgecolors="black", linewidths=0.7, marker="D",
           label="Test WER @ 100k (test-clean, held-out)")

# Interpolation threshold
has_interp = [(m["n_params"], find_interp_step(m))
              for m in models if find_interp_step(m) is not None]
if has_interp:
    first_n = min(has_interp, key=lambda x: x[0])[0]
    ax.axvline(first_n, color="black", linewidth=1.2, linestyle=":",
               alpha=0.7, label="Interp. threshold (train WER<5%)")

ax.axhline(INTERP_THRESHOLD, color="gray", linewidth=1, linestyle="--",
           alpha=0.5, label="WER = 5%")

ax.set_xscale("log")
ax.set_xlabel("# Parameters", fontsize=11)
ax.set_ylabel("WER", fontsize=11)
ax.tick_params(labelsize=9)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9, frameon=True)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "capacity_sweep.png"), dpi=150, bbox_inches="tight")
print("Saved -> capacity_sweep.png")
plt.close(fig)

# - 1b. Loss vs capacity (train / val / test) -----------------

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Stage A - Loss vs. Capacity at Final Step (d=512)", fontsize=13, fontweight="bold")

_cap_splits = [
    ("train_eval_loss", "#9970ab", "-.", 0.70, "Train-eval loss"),
    ("val_loss", "#f4a261", "--", 0.85, "Val loss (dev-clean)"),
    ("test_loss", "#2166ac", "-", 1.00, "Test loss (test-clean)"),
]

for metric, color, ls, alpha, label in _cap_splits:
    mn = np.array([m["mean"][metric][-1] for m in models])
    lo = np.array([m["lo"][metric][-1] for m in models])
    hi = np.array([m["hi"][metric][-1] for m in models])
    ax.fill_between(n_params_arr, lo, hi, color=color, alpha=0.12)
    ax.plot(
        n_params_arr,
        mn,
        linestyle=ls,
        color=color,
        linewidth=2.0,
        alpha=alpha,
        label=label,
        zorder=3,
        marker="o",
        markersize=7,
    )

# Coloured scatter on test curve to indicate model identity
test_mn = np.array([m["mean"]["test_loss"][-1] for m in models])
ax.scatter(
    n_params_arr,
    test_mn,
    c=[m["color"] for m in models],
    s=75,
    zorder=4,
    edgecolors="white",
    linewidths=0.8,
)

ax.set_xscale("log")
ax.set_xlabel("# Parameters", fontsize=11)
ax.set_ylabel("CTC Loss", fontsize=11)
ax.tick_params(labelsize=9)
ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}k")
)
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9, frameon=True)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "loss_vs_capacity.png"), dpi=150, bbox_inches="tight")
print("Saved -> loss_vs_capacity.png")
plt.close(fig)

# - 1c. Generalization gap vs capacity --------------------

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle(
    "Stage A - Generalization Gap vs. Capacity at Final Step (d=512)",
    fontsize=13,
    fontweight="bold",
)

gg_mean = np.array([m["mean"]["generalization_gap"][-1] for m in models])
gg_lo = np.array([m["lo"]["generalization_gap"][-1] for m in models])
gg_hi = np.array([m["hi"]["generalization_gap"][-1] for m in models])

GG_COLOR = "#d62728"
ax.fill_between(n_params_arr, gg_lo, gg_hi, color=GG_COLOR, alpha=0.15)
ax.plot(n_params_arr, gg_mean, "o-", color=GG_COLOR, linewidth=2.2, markersize=8, zorder=3)
ax.scatter(
    n_params_arr,
    gg_mean,
    c=[m["color"] for m in models],
    s=80,
    zorder=4,
    edgecolors="white",
    linewidths=0.8,
)

# Mark interpolation threshold (first model to cross 5% train WER)
has_interp = [
    (m["n_params"], find_interp_step(m)) for m in models if find_interp_step(m) is not None
]
if has_interp:
    first_n = min(has_interp, key=lambda x: x[0])[0]
    ax.axvline(
        first_n,
        color="black",
        linewidth=1.2,
        linestyle=":",
        alpha=0.7,
        label="Interp. threshold (L=3, 26M)",
    )
    ax.legend(fontsize=9, frameon=True)

ax.set_xscale("log")
ax.set_xlabel("# Parameters", fontsize=11)
ax.set_ylabel(r"$\mathcal{L}_\mathrm{test} - \mathcal{L}_\mathrm{train}$", fontsize=11)
ax.tick_params(labelsize=9)
ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}k")
)
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)

# Annotate the interpolation threshold line
if has_interp:
    ax.text(
        first_n * 1.06,
        gg_mean.max() * 0.99,
        "Interpolation threshold\n(first train WER $<$5\\%,\nL=3, 26M params)",
        fontsize=8,
        va="top",
        color="black",
        alpha=0.75,
    )

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gen_gap_vs_capacity.png"), dpi=150, bbox_inches="tight")
print("Saved -> gen_gap_vs_capacity.png")
plt.close(fig)

# - 1d. Epoch-wise double descent: test WER and test loss vs step -------
#    Only interpolating models (those that crossed 5% train WER threshold).

interp_models = [m for m in models if find_interp_step(m) is not None]

fig, (ax_wer, ax_loss) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Stage A - Epoch-wise Trajectory for Interpolating Models (d=512)",
    fontsize=13,
    fontweight="bold",
)

for m in interp_models:
    c = m["color"]
    steps = m["steps"]
    ist = find_interp_step(m)
    idx = np.where(steps == ist)[0][0]

    for ax, metric, ylabel, use_log in [
        (ax_wer,  "val_wer",  "WER",      False),
        (ax_loss, "val_loss", "CTC Loss", True),
    ]:
        mean = m["mean"][metric]
        lo   = m["lo"][metric]
        hi   = m["hi"][metric]

        ax.fill_between(steps, lo, hi, color=c, alpha=0.15)
        ax.plot(steps, mean, color=c, linewidth=1.9, label=m["label"])

        # Vertical tick at interpolation step
        yrange = mean.max() - mean.min()
        ax.plot([ist, ist],
                [mean[idx] - 0.04 * yrange, mean[idx] + 0.04 * yrange],
                color=c, linewidth=2.5, solid_capstyle="round", zorder=5)

    # Mark minimum val WER with a star
    min_idx = np.argmin(m["mean"]["val_wer"])
    ax_wer.scatter([steps[min_idx]], [m["mean"]["val_wer"][min_idx]],
                   color=c, s=55, zorder=6, marker="*",
                   edgecolors="black", linewidths=0.5)

    # Test WER at final step as a held-out diamond marker
    ax_wer.scatter([steps[-1]], [m["mean"]["test_wer"][-1]],
                   color=c, s=70, zorder=7, marker="D",
                   edgecolors="black", linewidths=0.7)

for ax, ylabel, use_log in [
    (ax_wer,  "WER",      False),
    (ax_loss, "CTC Loss", True),
]:
    ax.set_xlabel("Steps", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(step_fmt))
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if use_log:
        ax.set_yscale("log")

ax_wer.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax_wer.set_title("Val WER (dev-clean)", fontsize=11, fontweight="bold")
ax_wer.legend(fontsize=8, frameon=True, loc="upper right", ncol=2)

ax_loss.set_title("Val Loss (dev-clean)", fontsize=11, fontweight="bold")

# Shared note
fig.text(
    0.5, 0.01,
    "Tick mark = interpolation step (train WER first <5%)."
    "  Star (*) = val WER minimum.  Diamond = test-clean WER at step 100k (held-out).",
    ha="center", fontsize=8.5, color="dimgray",
)

fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(os.path.join(OUT_DIR, "epoch_descent.png"), dpi=150, bbox_inches="tight")
print("Saved -> epoch_descent.png")
plt.close(fig)

# - 2–8. Time-series: mean line + seed band ------------------

SERIES = [
    ("train_eval_loss", "Train Loss",                        "CTC Loss", True,  None),
    ("val_loss",        "Val Loss (dev-clean)",               "CTC Loss", True,  None),
    ("train_wer",       "Train WER  (first step < 5%)",  "WER",      False, INTERP_THRESHOLD),
    ("val_wer",         "Val WER (dev-clean)",                "WER",      False, None),
    ("generalization_gap", "Generalization Gap",             "test_loss − train_loss", False, None),
    ("sharpness_proxy", "Sharpness Proxy",                   "Sharpness", False, None),
    ("weight_norm_sum", "Weight Norm",                       "‖θ‖₂",     False, None),
]

for metric, title, ylabel, use_log, thresh in SERIES:
    fig, ax = plt.subplots(figsize=(10, 5))

    for m in models:
        mean = m["mean"][metric]
        lo = m["lo"][metric]
        hi = m["hi"][metric]
        c = m["color"]

        ax.fill_between(m["steps"], lo, hi, color=c, alpha=0.18)
        ax.plot(m["steps"], mean, label=m["label"], color=c, linewidth=1.9)

        if metric == "train_wer":
            ist = find_interp_step(m)
            if ist is not None:
                idx = np.where(m["steps"] == ist)[0][0]
                ax.scatter(
                    [ist],
                    [mean[idx]],
                    color=c,
                    s=65,
                    zorder=5,
                    marker="v",
                    edgecolors="black",
                    linewidths=0.6,
                )

    style_ax(ax, ylabel=ylabel, log=use_log, threshold_line=thresh)

    if metric in ("train_wer", "test_wer"):
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.legend(fontsize=8, frameon=True, loc="upper right", ncol=2 if n_models > 5 else 1)
    add_colorbar(fig, ax)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, metric + ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved -> {metric}.png")
    plt.close(fig)
