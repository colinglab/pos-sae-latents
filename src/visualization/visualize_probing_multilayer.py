#!/usr/bin/env python3
"""
Visualize probing F1 performance across all layers for each POS tag.

Output:
  outputs/layer_wise_performances_<layout>.png

Layout: 3 panels (open | closed | other), each overlaying all POS tags in that
class. Designed for 2-column academic papers.
Use --layout vertical for a 3×1 single-column-width figure instead.
"""

import os
import re
import colorsys
import pickle
import argparse
import warnings
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Toggle ─────────────────────────────────────────────────────────────────────
SHOW_CONFIDENCE_INTERVALS = False
# ──────────────────────────────────────────────────────────────────────────────

OPEN_CLASS   = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "INTJ"}
CLOSED_CLASS = {"DET", "ADP", "CONJ", "CCONJ", "SCONJ", "PART", "PRON", "AUX", "NUM"}
OTHER_CLASS  = {"PUNCT", "SYM", "X"}

# Base accent color per class — palette variants are generated around these
POS_COLOR = {"open": "#EE6352", "closed": "#00A6FB", "other": "#2B4570"}

CLASS_TITLE = {"open": "Open Class", "closed": "Closed Class", "other": "Other"}

MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "p"]

LINE_STYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),       # densely dashdotted
    (0, (5, 1)),             # densely dashed
    (0, (1, 1)),             # densely dotted
    (0, (3, 1, 1, 1, 1, 1)), # densely dashdotdotted
    (0, (5, 5)),             # loosely dashed
]


# ── Color palette generation ───────────────────────────────────────────────────

def _hex_to_hls(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r, g, b)


def _hls_to_hex(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def generate_palette(base_hex, n):
    """
    Generate n colors centered on base_hex by shifting hue and alternating
    lightness slightly, keeping the family resemblance.
    """
    if n == 1:
        return [base_hex]

    h, l, s = _hex_to_hls(base_hex)

    hue_step   = 0.035   # ~12 degrees per step
    light_step = 0.07    # lightness nudge alternating dark/light

    offsets = []
    for i in range(n):
        # Spread symmetrically: 0, -1, +1, -2, +2, ...
        if i == 0:
            offsets.append(0)
        elif i % 2 == 1:
            offsets.append(-(i + 1) // 2)
        else:
            offsets.append(i // 2)

    colors = []
    for rank, off in enumerate(offsets):
        new_h = (h + off * hue_step) % 1.0
        # Alternate darker / lighter around the base lightness
        new_l = np.clip(l + (off % 2) * light_step, 0.18, 0.82)
        colors.append(_hls_to_hex(new_h, new_l, s))

    return colors


def get_pos_type(pos):
    if pos in OPEN_CLASS:   return "open"
    if pos in CLOSED_CLASS: return "closed"
    if pos in OTHER_CLASS:  return "other"
    return "other"


# ── File discovery ─────────────────────────────────────────────────────────────

def discover_layers(probing_dir):
    pattern = re.compile(r"^(.+)_(\d+)_([^_]+)_results\.pkl$")
    entries = []
    for fname in sorted(os.listdir(probing_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        entries.append({
            "model_stem":   m.group(1),
            "layer":        int(m.group(2)),
            "target":       m.group(3),
            "results_path": os.path.join(probing_dir, fname),
        })
    entries.sort(key=lambda e: e["layer"])
    return entries


def load_results(entry):
    with open(entry["results_path"], "rb") as f:
        return pickle.load(f)


# ── Collect results ────────────────────────────────────────────────────────────

def collect_layer_data(entries):
    data = defaultdict(dict)
    for entry in entries:
        layer   = entry["layer"]
        results = load_results(entry)
        for pos, r in results.items():
            data[pos][layer] = {
                "cv_f1":     r["cv_f1"],
                "cv_f1_std": r.get("cv_f1_std", None),
            }
    return data


# ── Plotting ───────────────────────────────────────────────────────────────────

def _draw_panel(ax, pos_list, pos_class, layer_data, all_layers, show_ci,
                colors, layout, is_first, is_last_row):
    for i, pos in enumerate(pos_list):
        color  = colors[i]
        ls     = LINE_STYLES[i % len(LINE_STYLES)]
        marker = MARKERS[i % len(MARKERS)]
        layers = sorted(layer_data[pos].keys())
        f1s    = [layer_data[pos][l]["cv_f1"]     for l in layers]
        stds   = [layer_data[pos][l]["cv_f1_std"] for l in layers]

        ax.plot(
            layers, f1s,
            color=color, linewidth=1.8,
            linestyle=ls, marker=marker, markersize=5,
            markeredgewidth=0.6, markeredgecolor="white",
            label=pos,
        )

        if show_ci:
            has_std = all(s is not None for s in stds)
            if has_std:
                stds_arr = np.array(stds)
                f1s_arr  = np.array(f1s)
                ax.fill_between(
                    layers,
                    np.clip(f1s_arr - 1.96 * stds_arr, 0, 1),
                    np.clip(f1s_arr + 1.96 * stds_arr, 0, 1),
                    color=color, alpha=0.12,
                )

    ax.set_xlim(all_layers[0] - 0.5, all_layers[-1] + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(all_layers)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)

    # Seaborn white style gives clean axes; add only a subtle y-grid for readability
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="#888888")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel title colored with class accent
    ax.set_title(
        CLASS_TITLE[pos_class],
        fontsize=12, fontweight="bold",
        color=POS_COLOR[pos_class], pad=10,
    )

    # Axis labels
    if layout == "horizontal":
        ax.set_xlabel("Layer", fontsize=12)
        if is_first:
            ax.set_ylabel("F1 score", fontsize=12)
    else:
        if is_last_row:
            ax.set_xlabel("Layer", fontsize=6)
        ax.set_ylabel("F1 score", fontsize=6)

    ax.legend(
        fontsize=9,
        framealpha=0.9,
        edgecolor="#cccccc",
        loc="best",
        ncol=3,
        handlelength=2.0,
        handletextpad=0.6,
        borderpad=0.7,
    )


def plot_layer_wise_performances(layer_data, output_path, layout="horizontal",
                                 show_ci=SHOW_CONFIDENCE_INTERVALS):
    sns.set_theme(style="white")

    all_layers = sorted({layer for pd in layer_data.values() for layer in pd})

    by_class = {"open": [], "closed": [], "other": []}
    for pos in sorted(layer_data.keys()):
        by_class[get_pos_type(pos)].append(pos)

    col_order = [c for c in ("open", "closed", "other") if by_class[c]]
    n_panels  = len(col_order)

    if layout == "horizontal":
        fig, axes = plt.subplots(1, n_panels, figsize=(14, 4.8), sharey=True)
        if n_panels == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(n_panels, 1, figsize=(4.2, 2.5 * n_panels), sharey=True)
        if n_panels == 1:
            axes = [axes]

    for idx, ax in enumerate(axes):
        pos_class = col_order[idx]
        pos_list  = by_class[pos_class]
        colors    = generate_palette(POS_COLOR[pos_class], len(pos_list))
        _draw_panel(
            ax, pos_list, pos_class, layer_data, all_layers, show_ci, colors,
            layout=layout,
            is_first=(idx == 0),
            is_last_row=(idx == n_panels - 1),
        )

    fig.suptitle(
        "Layer-wise probing F1 by POS class",
        fontsize=17, fontweight="bold", y=1.02,
    )

    plt.tight_layout(
        w_pad=2.8 if layout == "horizontal" else 1.5,
        h_pad=2.0,
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise POS probing F1 plot for 2-column papers.")
    parser.add_argument("--probing_dir", default="probing-results")
    parser.add_argument("--output_dir",  default="outputs")
    parser.add_argument("--layout",      default="horizontal",
                        choices=["horizontal", "vertical"],
                        help="horizontal=1×3 spanning both columns; vertical=3×1 one column")
    parser.add_argument("--show_ci",     action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    entries = discover_layers(args.probing_dir)
    if not entries:
        print(f"No *_results.pkl files found in '{args.probing_dir}'.")
        return

    layers_found = sorted({e["layer"] for e in entries})
    print(f"Found {len(entries)} file(s) across layers: {layers_found}")

    layer_data = collect_layer_data(entries)
    show_ci    = args.show_ci or SHOW_CONFIDENCE_INTERVALS
    out_path   = os.path.join(args.output_dir, f"layer_wise_performances_{args.layout}.png")
    plot_layer_wise_performances(layer_data, out_path, layout=args.layout, show_ci=show_ci)
    print("Done.")


if __name__ == "__main__":
    main()
