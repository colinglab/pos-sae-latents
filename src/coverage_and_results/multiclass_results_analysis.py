#!/usr/bin/env python3
"""
RQ2 result plots for the compact-feature multiclass classifier (paper Section
5.2, Figure 5, Figure 12, Table 7). Consumes the *_results.pkl produced by
probing/probe_multiclass_cv.py (or probing/probe_multiclass_train_test.py).

Produces:
  {output_dir}/{base}_top_feature_coefficients.png   Figure 12 (coefficient heatmap)
  {output_dir}/{base}_confusion_matrix.png           Figure 5 (row-normalized confusion matrix)
  {output_dir}/{base}_classification_report.tex      Table 7 (LaTeX table)

Usage:
    python multiclass_results_analysis.py \\
        --results_pkl probing-results/latents_train_..._upos_multiclass_results.pkl \\
        --output_dir results-figures
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
import matplotlib.colors as colors

warnings.filterwarnings("ignore")

OPEN_CLASS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "INTJ"}
CLOSED_CLASS = {"DET", "ADP", "CONJ", "CCONJ", "SCONJ", "PART", "PRON", "AUX", "NUM"}
OTHER_CLASS = {"PUNCT", "SYM", "X"}
POS_COLOR = {"open": "#EE6352", "closed": "#00A6FB", "other": "#2B4570"}


def get_pos_class(pos):
    if pos in OPEN_CLASS:
        return "open"
    if pos in CLOSED_CLASS:
        return "closed"
    return "other"


def plot_top_feature_coefficients(coef_df, out_path, top_n=500):
    """Figure 12: heatmap of the top-`top_n` (by max abs. coefficient) latents x classes."""
    importance = coef_df.abs().max(axis=0)
    top_features = importance.sort_values(ascending=False).head(top_n).index
    df_plot = coef_df[top_features]

    feature_order = df_plot.abs().idxmax(axis=0).sort_values().index
    df_plot = df_plot[feature_order]

    sort_order = df_plot.index.map(lambda pos: (
        0 if get_pos_class(pos) == "open" else 1 if get_pos_class(pos) == "closed" else 2, pos
    ))
    df_plot = df_plot.iloc[np.argsort(sort_order)]

    vmax = np.abs(df_plot.values).max()
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(df_plot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(df_plot.index)))
    ax.set_yticklabels(df_plot.index)
    ax.set_xticks([])
    ax.set_xlabel("Latents")
    ax.set_ylabel("Classes")
    ax.set_title(f"Top {top_n} Feature Coefficients")
    for label in ax.get_yticklabels():
        label.set_color(POS_COLOR[get_pos_class(label.get_text())])
    plt.colorbar(im, ax=ax, label="Coefficient")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_confusion_matrix(cm_df, out_path, row_normalize=True):
    """Figure 5: (row-normalized) confusion matrix of the compact-feature multiclass classifier."""
    if row_normalize:
        row_sums = cm_df.values.sum(axis=1, keepdims=True)
        mat = cm_df.values / row_sums * 100
        cbar_label, title = "% of true class", "Confusion Matrix (row-normalized %)"
        vmin, vmax, cmap = 0, 100, "Blues"
    else:
        mat = cm_df.values
        cbar_label, title = "Count", "Confusion Matrix"
        vmin, vmax, cmap = max(mat.min(), 1e-1), mat.max(), None

    fig, ax = plt.subplots(figsize=(10, 8))
    if row_normalize:
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        im = ax.imshow(mat, norm=colors.LogNorm(vmin=vmin, vmax=vmax))

    ax.set_xticks(np.arange(len(cm_df.columns)))
    ax.set_yticks(np.arange(len(cm_df.index)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            text = f"{val:.1f}" if row_normalize else str(cm_df.values[i, j])
            color = "white" if row_normalize and val > 60 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label=cbar_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def write_classification_report_latex(report_dict, out_path):
    """Table 7: precision/recall/F1/support per class, as a LaTeX table."""
    df_report = pd.DataFrame(report_dict).T
    formatted = df_report.copy()
    for col in ["precision", "recall", "f1-score"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(lambda x: f"{x:.2f}")
    if "support" in formatted.columns:
        formatted["support"] = formatted["support"].astype(int)
    with open(out_path, "w") as f:
        f.write(formatted.to_latex())
    print(f"  Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_pkl", type=str, required=True,
                         help="_results.pkl produced by probe_multiclass_cv.py or probe_multiclass_train_test.py")
    parser.add_argument("--top_n", type=int, default=500, help="Top-N latents shown in the coefficient heatmap")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.results_pkl))[0]

    print(f"Loading {args.results_pkl}...")
    with open(args.results_pkl, "rb") as f:
        results = pickle.load(f)

    coef_df = results["coefficients"]
    plot_top_feature_coefficients(
        coef_df, os.path.join(args.output_dir, f"{base}_top_feature_coefficients.png"), top_n=args.top_n
    )

    # CV runs store "confusion_matrix"/"classification_report"; train/test runs
    # store "test_confusion_matrix"/"test_classification_report".
    cm_df = results.get("confusion_matrix", results.get("test_confusion_matrix"))
    report = results.get("classification_report", results.get("test_classification_report"))

    if cm_df is not None:
        plot_confusion_matrix(cm_df, os.path.join(args.output_dir, f"{base}_confusion_matrix.png"))
    if report is not None:
        write_classification_report_latex(report, os.path.join(args.output_dir, f"{base}_classification_report.tex"))

    print("Done.")


if __name__ == "__main__":
    main()
