"""
CMVN comparison plots.
Input:  outputs_sweep_regularization_{none,global,utt}.json
Output: cmvn_comparison/
  wer.png   - train / val / test WER for all 3 CMVN modes
  loss.png  - train-eval / val / test loss for all 3 CMVN modes
  generalization_gap.png
  sharpness_proxy.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os

FILES = {
    "none":   "outputs/outputs_regularization_none.json",
    "global": "outputs/outputs_regularization_global.json",
    "utt":    "outputs/outputs_regularization_utt.json",
}

COLORS = {
    "none": "#e15759",
    "global": "#4e79a7",
    "utt": "#59a14f",
}
LABELS = {
    "none": "No CMVN",
    "global": "Global CMVN",
    "utt": "Utterance CMVN",
}

# (loss_metric, wer_metric, linestyle, alpha, split_label)
# test-clean is held out; only train and val are shown as training curves.
SPLIT_STYLES = [
    ("train_eval_loss", "train_wer", "-.", 0.55, "Train"),
    ("val_loss",        "val_wer",   "--", 0.90, "Val (dev-clean)"),
]

base = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base, "cmvn_comparison")
os.makedirs(out_dir, exist_ok=True)


def load(fname):
    with open(os.path.join(base, fname)) as f:
        return json.load(f)


def extract(runs, metric):
    steps = np.array([s["step"] for s in runs[0]["steps"]])
    arr = np.stack([[s[metric] for s in r["steps"]] for r in runs])
    return steps, arr.mean(0), arr.min(0), arr.max(0)


data = {k: load(v) for k, v in FILES.items()}


def step_fmt(x, _):
    return f"{int(x/1000)}k" if x >= 1000 else str(int(x))


def make_combined(metric_idx, title, ylabel, use_log, fname):
    """
    metric_idx: 0 for loss metrics, 1 for WER metrics from SPLIT_STYLES.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)

    for norm in ["none", "global", "utt"]:
        runs = data[norm]
        color = COLORS[norm]
        for loss_m, wer_m, ls, alpha, _ in SPLIT_STYLES:
            metric = loss_m if metric_idx == 0 else wer_m
            steps, mean, lo, hi = extract(runs, metric)
            ax.plot(
                steps,
                mean,
                color=color,
                linestyle=ls,
                linewidth=1.9,
                alpha=alpha,
                zorder=3,
            )
            ax.fill_between(steps, lo, hi, color=color, alpha=0.08)

    ax.set_xlabel("Steps", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(step_fmt))
    if use_log:
        ax.set_yscale("log")
    if metric_idx == 1:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend: colour = CMVN mode, linestyle = split
    color_handles = [
        mpatches.Patch(color=COLORS[n], label=LABELS[n]) for n in ["none", "global", "utt"]
    ]
    style_handles = [
        mlines.Line2D(
            [],
            [],
            color="black",
            linestyle=ls,
            linewidth=1.8,
            alpha=alpha,
            label=slabel,
        )
        for _, _, ls, alpha, slabel in SPLIT_STYLES
    ]
    leg1 = ax.legend(
        handles=color_handles,
        fontsize=9,
        frameon=True,
        loc="upper right",
        title="CMVN mode",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=style_handles,
        fontsize=9,
        frameon=True,
        loc="upper center",
        title="Split",
    )

    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {path}")
    plt.close(fig)


# - Combined loss & WER ----------------------------

make_combined(0, "CMVN Comparison - Loss", "CTC Loss", True, "loss.png")
make_combined(1, "CMVN Comparison - WER", "WER", False, "wer.png")

# - Scalar extras (one line per CMVN mode) ------------------

EXTRAS = [
    (
        "generalization_gap",
        "Generalization Gap",
        r"$\mathcal{L}_\mathrm{test} - \mathcal{L}_\mathrm{train}$",
        False,
    ),
    ("sharpness_proxy", "Sharpness Proxy", "Sharpness", False),
]

for metric, title, ylabel, use_log in EXTRAS:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)

    for norm in ["none", "global", "utt"]:
        steps, mean, lo, hi = extract(data[norm], metric)
        color = COLORS[norm]
        ax.plot(steps, mean, label=LABELS[norm], color=color, linewidth=2.2)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.15)

    ax.set_xlabel("Steps", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(step_fmt))
    if use_log:
        ax.set_yscale("log")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, frameon=True, loc="best")

    fig.tight_layout()
    path = os.path.join(out_dir, metric + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {path}")
    plt.close(fig)
