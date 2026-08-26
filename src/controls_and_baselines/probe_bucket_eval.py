"""
Shared bucketed-evaluation utilities for the dense/static/majority-baseline
rebuttal experiments. Imported (not duplicated) by probe_dense_train_test.py,
majority_baseline_train_test.py, and compare_baselines.py so that "ambiguous
type", "OOV type", and "open/closed class" mean exactly the same thing in
every one of those reports.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Standard UD open/closed class split. PUNCT/SYM/X are neither.
OPEN_CLASS_UPOS = {"ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"}
CLOSED_CLASS_UPOS = {"ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"}
OTHER_UPOS = {"PUNCT", "SYM", "X"}

BUCKET_ORDER = [
    "overall",
    "open_class",
    "closed_class",
    "other_class",
    "ambiguous_type",
    "unambiguous_type",
    "minority_tag_instance",
    "oov_type",
    "in_vocab_type",
]


def word_type_series(df, word_type_col="form"):
    """Word type = lowercased surface form, consistent with the control-task scripts."""
    return df[word_type_col].astype(str).str.lower()


def compute_type_stats(df_train, target_col, word_type_col="form"):
    """
    Per-type statistics computed from the TRAINING set only (what a real
    lexicon could observe):
      - mode_tag: each type's most frequent tag in training
      - is_ambiguous: whether the type takes >1 distinct tag in training
      - overall_mode_tag: fallback tag for OOV types
    """
    word_type = word_type_series(df_train, word_type_col)
    tmp = pd.DataFrame({"type": word_type.to_numpy(), "tag": df_train[target_col].to_numpy()})
    grouped = tmp.groupby("type")["tag"]
    mode_tag = grouped.agg(lambda s: s.value_counts().idxmax())
    n_unique_tags = grouped.nunique()
    overall_mode_tag = tmp["tag"].value_counts().idxmax()
    return {
        "mode_tag": mode_tag.to_dict(),
        "is_ambiguous": (n_unique_tags > 1).to_dict(),
        "overall_mode_tag": overall_mode_tag,
    }


def lexicon_predict(word_type, type_stats):
    """Majority/lexicon baseline prediction: each type's training-set mode tag, or the
    overall training-set mode tag for types never seen in training (OOV)."""
    mode_tag = type_stats["mode_tag"]
    fallback = type_stats["overall_mode_tag"]
    return np.array([mode_tag.get(t, fallback) for t in word_type])


def class_bucket(tag):
    if tag in OPEN_CLASS_UPOS:
        return "open"
    if tag in CLOSED_CLASS_UPOS:
        return "closed"
    return "other"


def _score(y_true, y_pred):
    if len(y_true) == 0:
        return {"n": 0, "accuracy": float("nan"), "f1_macro": float("nan")}
    # Restrict macro-F1 to classes actually present as ground truth in this
    # bucket - e.g. the open_class bucket structurally never has DET/ADP as
    # y_true, so scoring against the full tagset would spuriously drag the
    # macro average down with irrelevant always-absent classes.
    labels = sorted(set(y_true))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def bucketed_report(y_true, y_pred, word_type, type_stats):
    """
    Returns {bucket_name: {n, accuracy, f1_macro}} for a fixed set of
    evaluation buckets, plus "overall". `word_type` must be the same
    lowercased-form word type used to build `type_stats`.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    word_type = np.asarray(word_type)

    is_ambiguous_map = type_stats["is_ambiguous"]
    mode_tag_map = type_stats["mode_tag"]

    is_oov = np.array([t not in mode_tag_map for t in word_type])
    is_ambiguous = np.array([is_ambiguous_map.get(t, False) for t in word_type])
    classes = np.array([class_bucket(t) for t in y_true])
    is_minority = np.array([
        (not is_oov[i]) and (y_true[i] != mode_tag_map.get(word_type[i]))
        for i in range(len(word_type))
    ])

    report = {"overall": _score(y_true, y_pred)}
    masks = {
        "open_class": classes == "open",
        "closed_class": classes == "closed",
        "other_class": classes == "other",
        "ambiguous_type": is_ambiguous & ~is_oov,
        "unambiguous_type": (~is_ambiguous) & (~is_oov),
        "minority_tag_instance": is_minority,
        "oov_type": is_oov,
        "in_vocab_type": ~is_oov,
    }
    for name, mask in masks.items():
        report[name] = _score(y_true[mask], y_pred[mask])
    return report


def print_bucketed_report(label, report):
    print(f"\nBucketed results ({label}):")
    for name in BUCKET_ORDER:
        scores = report[name]
        print(
            f"  {name:22s} n={scores['n']:6d}  "
            f"acc={scores['accuracy']:.4f}  f1_macro={scores['f1_macro']:.4f}"
        )
