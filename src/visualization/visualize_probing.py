#!/usr/bin/env python3
"""
Visualize probing results across all layers found in a probing-results directory.

Outputs saved to outputs/ (or --output_dir):

  Part 1 – per layer:
    {base}_summary.png         F1 + non-zero latents bar charts
    {base}_top_latents.png     Top-k latents per POS grid
    {base}_umap_selected.html  UMAP of top-k selected latents

  Part 2 – cross-layer, per POS:
    {target}_{pos}_all_layers.png  Top-k latents for every layer in one figure
"""

import os
import re
import pickle
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
import plotly.express as px
from tqdm import tqdm

warnings.filterwarnings("ignore")

OPEN_CLASS   = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
CLOSED_CLASS = {"DET", "ADP", "CONJ", "CCONJ", "SCONJ", "PART", "PRON", "AUX", "NUM"}
POS_COLOR    = {"open": "steelblue", "closed": "tomato", "other": "grey"}


def get_pos_type(pos):
    if pos in OPEN_CLASS:   return "open"
    if pos in CLOSED_CLASS: return "closed"
    return "other"


# ── File discovery ────────────────────────────────────────────────────────────

def discover_layers(probing_dir):
    """
    Scan probing_dir for *_results.pkl files.
    File naming: {model_stem}_{layer}_{target}_results.pkl
    The layer field is always numeric, which anchors the regex.
    """
    pattern = re.compile(r"^(.+)_(\d+)_([^_]+)_results\.pkl$")
    entries = []
    for fname in sorted(os.listdir(probing_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        model_stem, layer, target = m.group(1), int(m.group(2)), m.group(3)
        base = fname[: -len("_results.pkl")]
        entries.append({
            "model_stem":   model_stem,
            "layer":        layer,
            "target":       target,
            "base":         base,
            "results_path": os.path.join(probing_dir, fname),
            "scaler_path":  os.path.join(probing_dir, f"{base}_scaler.pkl"),
            "df_path":      os.path.join(probing_dir, f"{base}_df_filtered.parquet"),
        })
    entries.sort(key=lambda e: e["layer"])
    return entries


def load_entry(entry):
    with open(entry["results_path"], "rb") as f:
        results = pickle.load(f)
    df_filtered = pd.read_parquet(entry["df_path"])
    return results, df_filtered


# ── Part 1 helpers ────────────────────────────────────────────────────────────

def plot_summary(results, title_suffix, out_path):
    rows = []
    for pos, r in results.items():
        rows.append({
            "pos":       pos,
            "pos_type":  get_pos_type(pos),
            "cv_f1":     r["cv_f1"],
            "n_nonzero": int((r["coefficients"] != 0).sum()),
        })
    summary = pd.DataFrame(rows).sort_values("cv_f1", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, label in zip(
        axes,
        ["cv_f1", "n_nonzero"],
        ["Cross-validated F1", "Non-zero latents in classifier"],
    ):
        colors = summary["pos_type"].map(POS_COLOR)
        ax.barh(summary["pos"], summary[metric], color=colors)
        ax.set_xlabel(label)
        ax.invert_yaxis()
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=c, label=k)
            for k, c in POS_COLOR.items()
        ]
        ax.legend(handles=handles, fontsize=8)

    plt.suptitle(f"Probing classifier results by POS\n{title_suffix}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_top_latents(results, title_suffix, out_path, top_k=10):
    pos_list = sorted(results.keys())
    n     = len(pos_list)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3), squeeze=False)
    flat = axes.flatten()

    for ax, pos in zip(flat, pos_list):
        top   = results[pos]["top_latents"].head(top_k)
        color = POS_COLOR[get_pos_type(pos)]
        ptype = get_pos_type(pos)
        f1    = results[pos]["cv_f1"]

        if len(top) == 0:
            ax.set_title(f"{pos} (no positive latents)")
            ax.axis("off")
            continue

        ax.barh(range(len(top)), top.values, color=color, alpha=0.8)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([l.replace("act_", "") for l in top.index], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"{pos}  [F1={f1:.2f}]  ({ptype})")
        ax.set_xlabel("coefficient")

    for ax in flat[len(pos_list):]:
        ax.set_visible(False)

    plt.suptitle(f"Top positive latents per POS\n{title_suffix}", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_umap(df_filtered, results, entry, out_path, top_k=20, min_token_freq=50):
    target   = entry["target"]
    layer    = entry["layer"]
    model    = entry["model_stem"]

    act_cols = [c for c in df_filtered.columns if c.startswith("act_")]
    mask     = df_filtered[act_cols].sum(axis=1) > 0
    df       = df_filtered[mask].copy()

    color_col = target if target in df.columns else "upos"
    counts    = df[color_col].value_counts()
    valid     = counts[counts >= min_token_freq].index
    df        = df[df[color_col].isin(valid)]

    selected = sorted({
        lat
        for r in results.values()
        for lat in r["top_latents"].head(top_k).index
        if lat in df.columns
    })
    print(f"  UMAP: {len(selected)} selected latents / {len(act_cols)} total")

    X   = df[selected].values.astype(np.float32)
    emb = umap.UMAP(
        n_neighbors=2, min_dist=0.1, metric="cosine", random_state=42
    ).fit_transform(X)

    hover_cols = {
        "token_id": True, "form": True, "lemma": True,
        "upos": True, "feats": True, "deprel": True, "sent_id": True,
        "umap_x": False, "umap_y": False,
    }
    plot_df = df[
        [c for c in ["token_id", "form", "lemma", "upos", "feats", "deprel", "sent_id", "text"]
         if c in df.columns]
    ].copy()
    plot_df["umap_x"]   = emb[:, 0]
    plot_df["umap_y"]   = emb[:, 1]
    plot_df["pos_type"] = plot_df["upos"].map(
        lambda p: "open" if p in OPEN_CLASS else "closed" if p in CLOSED_CLASS else "other"
    )
    hover_cols["pos_type"] = True
    if "text" in plot_df.columns:
        hover_cols["text"] = False

    fig = px.scatter(
        plot_df,
        x="umap_x", y="umap_y",
        color=color_col,
        symbol="pos_type",
        hover_data=hover_cols,
        title=f"UMAP — {len(selected)} selected latents | Layer {layer} | {model} | {target}",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Light24,
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  Saved → {out_path}")


# ── Part 2 ────────────────────────────────────────────────────────────────────

def plot_cross_layer_per_pos(all_layer_results, output_dir, target, top_k=10):
    """
    For each POS, one PNG with a subplot per layer showing top-k latents.
    all_layer_results: list of (layer, results_dict) sorted by layer.
    """
    all_pos = sorted({pos for _, r in all_layer_results for pos in r})

    for pos in all_pos:
        layers_present = [(layer, r) for layer, r in all_layer_results if pos in r]
        n_layers = len(layers_present)

        ncols = 4
        nrows = (n_layers + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3), squeeze=False)
        flat  = axes.flatten()
        color = POS_COLOR[get_pos_type(pos)]
        ptype = get_pos_type(pos)

        for ax, (layer, results) in zip(flat, layers_present):
            r    = results[pos]
            top  = r["top_latents"].head(top_k)
            f1   = r["cv_f1"]
            n_nz = int((r["coefficients"] != 0).sum())

            if len(top) == 0:
                ax.set_title(f"Layer {layer}\nF1={f1:.2f}  active={n_nz}\n(no positive latents)")
                ax.axis("off")
                continue

            ax.barh(range(len(top)), top.values, color=color, alpha=0.8)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels([l.replace("act_", "") for l in top.index], fontsize=7)
            ax.invert_yaxis()
            ax.set_title(f"Layer {layer}  F1={f1:.2f}  active={n_nz}")
            ax.set_xlabel("coefficient")

        for ax in flat[n_layers:]:
            ax.set_visible(False)

        plt.suptitle(f"{pos} ({ptype}) — top latents across layers [{target}]", y=1.01)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{target}_{pos.lower()}_all_layers.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize probing results across all layers.")
    parser.add_argument("--probing_dir",  default="probing-results", help="Directory with probing result files")
    parser.add_argument("--output_dir",   default="outputs",         help="Directory for output files")
    parser.add_argument("--top_k",        type=int, default=10,      help="Top-k latents to show per POS")
    parser.add_argument("--skip_umap",    action="store_true",        help="Skip UMAP generation (faster)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    entries = discover_layers(args.probing_dir)
    if not entries:
        print(f"No *_results.pkl files found in '{args.probing_dir}'.")
        return

    layers_found = [e["layer"] for e in entries]
    print(f"Found {len(entries)} layer file(s): {layers_found}\n")

    by_target = defaultdict(list)

    for entry in entries:
        layer  = entry["layer"]
        target = entry["target"]
        model  = entry["model_stem"]
        base   = entry["base"]

        print(f"── Layer {layer}  target={target}  model={model} ──────────────────────────")
        results, df_filtered = load_entry(entry)

        title_suffix = f"Layer {layer} | {model} | {target}"

        plot_summary(
            results, title_suffix,
            os.path.join(args.output_dir, f"{base}_summary.png"),
        )
        plot_top_latents(
            results, title_suffix,
            os.path.join(args.output_dir, f"{base}_top_latents.png"),
            top_k=args.top_k,
        )

        if not args.skip_umap:
            plot_umap(
                df_filtered, results, entry,
                os.path.join(args.output_dir, f"{base}_umap_selected.html"),
                top_k=args.top_k,
            )

        by_target[target].append((layer, results))

    print("\n── Cross-layer per-POS plots ───────────────────────────────────────────")
    for target, layer_results in by_target.items():
        layer_results.sort(key=lambda x: x[0])
        print(f"  target={target}  ({len(layer_results)} layer(s))")
        plot_cross_layer_per_pos(layer_results, args.output_dir, target, top_k=args.top_k)

    print("\nDone.")


if __name__ == "__main__":
    main()
