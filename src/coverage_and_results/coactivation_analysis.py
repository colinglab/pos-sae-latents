#!/usr/bin/env python3
"""
RQ3 - Held-out co-activation / distinctiveness analysis (paper Section 4.3 /
5.3, Figure 6, Table 8, Appendix Figure 13's sibling density plots).

For each POS c, predicts "active" on a token whenever at least one latent in
its selected set L*_c (from coverage_analysis.py) fires. Comparing these
per-POS binary predictions against gold UPOS on a held-out split gives:
  - a recall/false-positive-rate matrix M[c, c'] (Figure 6),
  - a per-class distinctiveness score D(c) = M[c,c] / sum_c' M[c,c'] (Table 8),
  - standard multilabel precision/recall/F1 (mirrors Table 7 but computed
    from the additive L*_c indicator instead of the multiclass classifier).

Usage:
    python coactivation_analysis.py \\
        --data_path layer-wise-latents/latents_test_meta-llama_Meta-Llama-3-8B_30.parquet \\
        --important_act_pkl important_act_per_pos_layer30.pkl \\
        --output_dir held-out-results
"""
import os
import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, hamming_loss

warnings.filterwarnings("ignore")


def load_data(path):
    df = pd.read_parquet(path) if ".parquet" in path else pd.read_csv(path)
    act_cols = [c for c in df.columns if c.startswith("act_")]
    print(f"Loaded {len(df):,} rows from '{path}' ({len(act_cols)} activation columns, "
          f"{df['upos'].nunique()} unique UPOS tags)")
    return df


def load_important_acts(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    print(f"Loaded important_act_per_pos with {len(d)} POS entries")
    return d


def add_predictions(df, important_act_per_pos):
    """predicted_{pos} = 1 iff any latent in L*_pos is active; n_active_{pos} = count of those active."""
    df = df.copy()
    for pos, act_cols in important_act_per_pos.items():
        present = [c for c in act_cols if c in df.columns]
        if not present:
            df[f"predicted_{pos}"] = 0
            df[f"n_active_{pos}"] = 0
            continue
        active_mask = (df[present] > 0).astype(int)
        df[f"n_active_{pos}"] = active_mask.sum(axis=1)
        df[f"predicted_{pos}"] = (df[f"n_active_{pos}"] > 0).astype(int)
    return df


def build_true_matrix(df, pos_tags, upos_col="upos"):
    y_true = np.zeros((len(df), len(pos_tags)), dtype=int)
    pos_index = {p: i for i, p in enumerate(pos_tags)}
    for row_i, upos in enumerate(df[upos_col]):
        if upos in pos_index:
            y_true[row_i, pos_index[upos]] = 1
    return y_true


def compute_multilabel_metrics(y_true, y_pred, pos_tags):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    rows = [{"POS": pos, "Precision": p[i], "Recall": r[i], "F1": f[i], "Support": int(s[i])}
            for i, pos in enumerate(pos_tags)]
    rows.append({"POS": "macro avg", "Precision": p_mac, "Recall": r_mac, "F1": f_mac, "Support": int(s.sum())})
    rows.append({"POS": "micro avg", "Precision": p_mic, "Recall": r_mic, "F1": f_mic, "Support": int(s.sum())})
    metrics_df = pd.DataFrame(rows).set_index("POS")
    hl = hamming_loss(y_true, y_pred)
    return metrics_df, hl


def binary_cross_pos_matrix(df, pos_tags, upos_col="upos"):
    """
    Figure 6: M[true_pos, predicted_pos] = P(predicted_pos fires | true POS).
    Diagonal = recall per class; off-diagonal = spurious (false-positive) activation rate.
    """
    pred_cols = [f"predicted_{p}" for p in pos_tags]
    rows = []
    for true_pos in pos_tags:
        mask = df[upos_col] == true_pos
        rows.append(df.loc[mask, pred_cols].mean().tolist() if mask.sum() else [0.0] * len(pos_tags))
    return pd.DataFrame(rows, index=pos_tags, columns=pos_tags)


def distinctiveness_scores(mat_df):
    """Table 8: D(c) = M[c,c] / sum_c' M[c,c'] (chance baseline = 1/|C|)."""
    diag = np.diag(mat_df.values)
    row_sums = mat_df.values.sum(axis=1)
    d = diag / row_sums
    return pd.Series(d, index=mat_df.index, name="D(c)")


def plot_binary_cross_pos_heatmap(mat_df, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat_df.values, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(np.arange(len(mat_df.columns)))
    ax.set_yticks(np.arange(len(mat_df.index)))
    ax.set_xticklabels(mat_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(mat_df.index)
    ax.set_xlabel("Predicted POS ->")
    ax.set_ylabel("True POS ->")
    ax.set_title("P(predicted POS fires | true POS)\n(diagonal = recall, off-diagonal = spurious activation rate)")
    for i in range(mat_df.shape[0]):
        for j in range(mat_df.shape[1]):
            ax.text(j, i, f"{mat_df.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="P(predicted=1 | true POS)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_prediction_count_distribution(df, pos_tags, out_path):
    """How many POS tags are simultaneously predicted active per token."""
    pred_cols = [f"predicted_{p}" for p in pos_tags]
    counts = df[pred_cols].sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("Number of POS tags predicted active")
    ax.set_ylabel("Token count")
    ax.set_title("How many POS predictions fire per token?")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")
    return counts.describe()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", type=str, required=True,
                         help="Held-out latents parquet (e.g. from extraction/extract_latents.py --split test)")
    parser.add_argument("--important_act_pkl", type=str, required=True,
                         help="important_act_per_pos_layer{N}.pkl from coverage_analysis.py")
    parser.add_argument("--upos_col", type=str, default="upos")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.data_path))[0]

    df = load_data(args.data_path)
    important_act_per_pos = load_important_acts(args.important_act_pkl)
    df = add_predictions(df, important_act_per_pos)
    pos_tags = list(important_act_per_pos.keys())

    y_true = build_true_matrix(df, pos_tags, args.upos_col)
    y_pred = df[[f"predicted_{p}" for p in pos_tags]].values

    metrics_df, hl = compute_multilabel_metrics(y_true, y_pred, pos_tags)
    print(f"Hamming loss: {hl:.4f}\n")
    print(metrics_df.to_string(float_format="{:.3f}".format))
    metrics_df.to_csv(os.path.join(args.output_dir, f"{base}_multilabel_metrics.csv"))

    cross_mat = binary_cross_pos_matrix(df, pos_tags, args.upos_col)
    cross_mat.to_csv(os.path.join(args.output_dir, f"{base}_coactivation_matrix.csv"))
    plot_binary_cross_pos_heatmap(cross_mat, os.path.join(args.output_dir, f"{base}_coactivation_heatmap.png"))

    distinct = distinctiveness_scores(cross_mat)
    print(f"\nDistinctiveness D(c): mean={distinct.mean():.3f} std={distinct.std():.3f}")
    distinct.to_csv(os.path.join(args.output_dir, f"{base}_distinctiveness.csv"))

    plot_prediction_count_distribution(df, pos_tags, os.path.join(args.output_dir, f"{base}_prediction_count_dist.png"))

    print("Done.")


if __name__ == "__main__":
    main()
