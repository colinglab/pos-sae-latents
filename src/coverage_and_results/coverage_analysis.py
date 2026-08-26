#!/usr/bin/env python3
"""
RQ2 - Coverage and compactness analysis (paper Section 4.2 / 5.2).

Starting from the one-vs-rest binary probing classifiers produced by
probing/probe_onevsrest.py, this script:

  1. Ranks each POS's salient latents by positive coefficient (feature
     salience), and walks down that ranked list to find the smallest number
     of latents k_c^tau whose union covers at least `threshold` (tau) of that
     POS's gold tokens (coverage) - Figure 4 / Table 6.
  2. Saves the resulting per-POS latent sets as a pickle
     (important_act_per_pos_layer{N}[_t{threshold}].pkl), which is the L*_c
     input consumed by probing/probe_multiclass_cv.py and
     probing/probe_multiclass_train_test.py.
  3. Reports how much the selected latent sets overlap across POS tags
     (latent-sharing analysis).
  4. Optionally (--treebank_test_conllu), re-extracts SAE activations for a
     held-out CoNLL-U split restricted to the selected latents and plots the
     per-POS density of "% of L*_c active" (Appendix Figure 13). This step
     needs the LLM + SAE loaded and is skipped unless a path is given.

Usage:
    python coverage_analysis.py \\
        --probing_dir probing-results \\
        --layer_n 30 --threshold 0.95 \\
        --output_dir coverage-results
"""
import os
import argparse
import pickle
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OPEN_CLASS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "INTJ"}
CLOSED_CLASS = {"DET", "ADP", "CONJ", "CCONJ", "SCONJ", "PART", "PRON", "AUX", "NUM"}
OTHER_CLASS = {"PUNCT", "SYM", "X"}
POS_COLOR = {"open": "#EE6352", "closed": "#00A6FB", "other": "#2B4570"}


def get_pos_type(pos):
    if pos in OPEN_CLASS:
        return "open"
    if pos in CLOSED_CLASS:
        return "closed"
    return "other"


def load_probe_results(probing_dir, model_stem, layer_n, target_col):
    base = f"{model_stem}_{layer_n}_{target_col}"
    with open(os.path.join(probing_dir, f"{base}_results.pkl"), "rb") as f:
        clf_results = pickle.load(f)
    df_filtered = pd.read_parquet(os.path.join(probing_dir, f"{base}_df_filtered.parquet"))
    return clf_results, df_filtered


def activation_coverage(df, clf_results, threshold=0.95, act_prefix="act_"):
    """
    For each POS, walk its salience-ranked (positive-coefficient) latents and
    record cumulative token coverage after adding each one. Stops as soon as
    `threshold` coverage is reached (or the ranked list is exhausted).

    Returns dict[pos] -> DataFrame[act, coef, cumulative_coverage, n_acts].
    """
    act_cols = [c for c in df.columns if c.startswith(act_prefix)]
    results = {}

    for pos, r in clf_results.items():
        pos_df = df.loc[df["upos"] == pos, act_cols]
        n = len(pos_df)
        if n == 0:
            continue

        active_acts = r["coefficients"][r["coefficients"] > 0].sort_values(ascending=False)

        covered = pd.Series(False, index=pos_df.index)
        steps = []
        for act_col, coef in active_acts.items():
            if act_col not in pos_df.columns:
                continue
            covered |= (pos_df[act_col] > 0)
            coverage = covered.sum() / n
            steps.append({
                "act": act_col,
                "coef": coef,
                "cumulative_coverage": coverage,
                "n_acts": len(steps) + 1,
            })
            if coverage >= threshold:
                break

        results[pos] = pd.DataFrame(steps)

    return results


def plot_cumulative_coverage(coverage_results, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for pos, df in coverage_results.items():
        ax.plot(df["n_acts"], df["cumulative_coverage"], label=pos,
                color=POS_COLOR[get_pos_type(pos)], linewidth=2, alpha=0.85)
    ax.set_xlabel("Number of latents")
    ax.set_ylabel("Cumulative coverage")
    ax.set_title("Cumulative coverage growth by POS", fontweight="bold")
    handles = [plt.Line2D([0], [0], color=c, linewidth=4, label=k.capitalize())
               for k, c in POS_COLOR.items()]
    ax.legend(handles=handles, title="Class", loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_k95_bar(coverage_results, threshold, out_path):
    """Figure 4: number of salient latents required to reach `threshold` coverage, per POS."""
    counts = {pos: len(df) for pos, df in coverage_results.items()}
    df_counts = pd.DataFrame({"pos": list(counts.keys()), "n": list(counts.values())}) \
        .sort_values("n", ascending=False)
    colors = [POS_COLOR[get_pos_type(p)] for p in df_counts["pos"]]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df_counts["pos"], df_counts["n"], color=colors, width=0.6)
    ax.set_xlabel("UPOS")
    ax.set_ylabel("k")
    ax.set_title(rf"Latents to reach $k_c^{{{int(threshold * 100)}}}$ by POS", fontweight="bold")
    plt.xticks(rotation=90, ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=k.capitalize()) for k, c in POS_COLOR.items()]
    ax.legend(handles=handles, title="Class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")
    return counts


def plot_coverage_over_salience(coverage_results, clf_results, out_path):
    """Table 6: ratio of (# latents to reach coverage threshold) to (# non-zero salient latents)."""
    coverage_vs_active = {}
    for pos, df in coverage_results.items():
        n_active = (clf_results[pos]["coefficients"] != 0).sum()
        coverage_vs_active[pos] = (len(df) / n_active) * 100 if n_active else float("nan")

    colors = [POS_COLOR[get_pos_type(p)] for p in coverage_vs_active]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(coverage_vs_active.keys(), coverage_vs_active.values(), color=colors)
    ax.set_xlabel("UPOS")
    ax.set_ylabel("Coverage on active latents (%)")
    ax.set_title("Coverage over number of salient features per POS")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")
    return coverage_vs_active


def plot_latent_sharing(important_act_per_pos, out_path):
    """How many of a POS's selected latents are also selected for 1, 2, 3, ... other POS tags."""
    latent_sets = {pos: set(latents) for pos, latents in important_act_per_pos.items()}
    latent_frequency = Counter()
    for s in latent_sets.values():
        latent_frequency.update(s)

    sharing_stats = {}
    for pos, s in latent_sets.items():
        sharing_stats[pos] = Counter(latent_frequency[latent] for latent in s)

    sharing_df = pd.DataFrame(sharing_stats).fillna(0).T
    sharing_df = sharing_df[sorted(sharing_df.columns)]
    sharing_df = sharing_df.div(sharing_df.sum(axis=1), axis=0).sort_index()

    ax = sharing_df.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis")
    ax.set_ylabel("Proportion of selected latents")
    ax.set_xlabel("POS")
    ax.set_title("How Shared Are Probe-Selected SAE Latents?")
    ax.legend(title="Number of POS sharing latent", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")
    return sharing_df


def plot_treebank_test_density(important_acts_all, count_active_fn, conllu_path, model, tokenizer, sae,
                                hidden_layer, device, out_path):
    """
    Appendix Figure 13: KDE of "% of L*_c active" per POS, on a held-out
    CoNLL-U split. Requires the LLM + SAE already loaded on `device`.
    """
    # Imported lazily (and via a sys.path tweak, since extract_latents.py
    # lives in a sibling directory) so this module works without the
    # extraction deps unless this optional step is actually requested.
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extraction"))
    from extract_latents import parse_conllu, conllu_to_token_df, extract_latents_aligned

    sentences = parse_conllu(conllu_path)
    token_df = conllu_to_token_df(sentences)
    acts_df = extract_latents_aligned(token_df, model, tokenizer, sae, device, hidden_layer=hidden_layer, window=0)
    final_df = pd.concat([token_df, acts_df], axis=1)
    keep_acts = [c for c in final_df.columns if (not c.startswith("act_") or c in important_acts_all)]
    filtered_df = final_df[keep_acts]
    filtered_df["active_per_pos_percent"] = filtered_df.apply(count_active_fn, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for pos, group in filtered_df.groupby("upos"):
        vals = group["active_per_pos_percent"].dropna()
        if len(vals) >= 2 and vals.nunique() > 1:
            vals.plot(kind="kde", ax=ax, label=pos)
    ax.set_title(f"Distribution of L*_c active latents by POS - {os.path.basename(conllu_path)}")
    ax.set_xlabel("Active latents %")
    ax.legend(title="UPOS")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")
    return filtered_df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probing_dir", type=str, default="probing-results",
                         help="Directory with the *_results.pkl / *_df_filtered.parquet produced by probe_onevsrest.py")
    parser.add_argument("--model_stem", type=str, default="latents_train_meta-llama_Meta-Llama-3-8B")
    parser.add_argument("--layer_n", type=int, default=30)
    parser.add_argument("--target_col", type=str, default="upos")
    parser.add_argument("--threshold", type=float, default=0.95, help="Coverage threshold tau (paper default: 0.95)")
    parser.add_argument("--pkl_suffix", type=str, default="",
                         help="Suffix for the output important_act_per_pos pickle, e.g. '_t0.9'. "
                              "Leave empty for the main-paper default (threshold=0.95) so downstream "
                              "probing scripts find it via their default 'important_act_per_pos_layer{N}.pkl' lookup.")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--treebank_test_conllu", type=str, default=None,
                         help="Optional: path to a held-out CoNLL-U split (e.g. UD_English-GUM/en_gum-ud-test.conllu) "
                              "to additionally produce the Appendix Figure 13 density plot. Requires --sae_id/--model_id.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--sae_id", type=str, default="EleutherAI/sae-llama-3-8b-32x")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading one-vs-rest probe results from {args.probing_dir} (layer {args.layer_n})...")
    clf_results, df_filtered = load_probe_results(args.probing_dir, args.model_stem, args.layer_n, args.target_col)

    print(f"Computing coverage sets at threshold={args.threshold}...")
    coverage_results = activation_coverage(df_filtered, clf_results, threshold=args.threshold)

    important_act_per_pos = {pos: df["act"].tolist() for pos, df in coverage_results.items()}
    important_acts_all = sorted({e for latents in important_act_per_pos.values() for e in latents})
    n_sum = sum(len(v) for v in important_act_per_pos.values())
    print(f"Union of selected latents (L*): {len(important_acts_all)} (sum across POS with overlap: {n_sum})")

    pkl_path = os.path.join(args.output_dir, f"important_act_per_pos_layer{args.layer_n}{args.pkl_suffix}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(important_act_per_pos, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved L*_c per POS -> {pkl_path}")

    plot_cumulative_coverage(coverage_results, os.path.join(args.output_dir, f"cumulative_coverage_layer{args.layer_n}.png"))
    plot_k95_bar(coverage_results, args.threshold, os.path.join(args.output_dir, f"k_threshold_by_pos_layer{args.layer_n}.png"))
    plot_coverage_over_salience(coverage_results, clf_results, os.path.join(args.output_dir, f"coverage_over_salience_layer{args.layer_n}.png"))
    plot_latent_sharing(important_act_per_pos, os.path.join(args.output_dir, f"latent_sharing_layer{args.layer_n}.png"))

    if args.treebank_test_conllu:
        print(f"Extracting activations on {args.treebank_test_conllu} for the Figure 13 density plot...")
        import torch
        from sparsify import Sae
        from transformers import AutoModelForCausalLM, AutoTokenizer

        def count_active(row):
            pos = row["upos"]
            feat_per_pos = important_act_per_pos.get(pos, [])
            if not feat_per_pos:
                return np.nan
            c = sum(1 for feat in feat_per_pos if feat in row.index and row[feat] > 0.0)
            return c / len(feat_per_pos)

        sae = Sae.load_from_hub(args.sae_id, hookpoint=f"layers.{args.layer_n}").to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(args.model_id).to(args.device)

        plot_treebank_test_density(
            set(important_acts_all), count_active, args.treebank_test_conllu,
            model, tokenizer, sae, args.layer_n, args.device,
            os.path.join(args.output_dir, f"density_active_latents_layer{args.layer_n}_test.png"),
        )

    print("Done.")


if __name__ == "__main__":
    main()
