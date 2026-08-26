#!/usr/bin/env bash
# Sweeps the torch-backed one-vs-rest SAE-latent probes over combinations of
# important_act_per_pos file (coverage threshold tau) x C, producing summary
# tables at the end. One-vs-rest analogue of scripts/05_run_hparam_sweep.sh -
# but unlike that script, this one writes into a single non-timestamped
# RUN_DIR (not a fresh timestamped folder each time) and skips any
# (important_act_pkl, C) combination whose _metrics.json already exists
# there, so you can extend an existing sweep (e.g. add C=0.5 to a directory
# that already has 0.01/0.05/0.1) just by re-running this script.
#
# Edit the CONFIG block below, then run:
#   ./06_run_onevsrest_hparam_sweep.sh
set -euo pipefail

# ============================== CONFIG ======================================

INPUT_FILE="layer-wise-latents/latents_train_meta-llama_Meta-Llama-3-8B_30.parquet"
TARGET_COL="upos"

# The dimensions actually being swept. Generate these three with
# coverage_analysis.py --threshold 0.9/0.95/0.99 --pkl_suffix _t0.9/_t0.95/_t0.99.
IMPORTANT_ACT_PKLS=(
    "important_act_per_pos_layer30_t0.9.pkl"
    "important_act_per_pos_layer30_t0.95.pkl"
    "important_act_per_pos_layer30_t0.99.pkl"
)
C_VALUES=(0.01 0.05 0.1 0.5)

# Passed straight through to the probe script.
BACKEND="torch"
MIN_TOKEN_FREQ=50
MIN_LATENT_FREQ=100
MAX_ITER=1000
TOP_K=20
CV_FOLDS=5
DEVICE=""            # e.g. "cuda:0"; empty = auto-detect
TORCH_VERBOSE=0       # 0 = silence per-iteration FISTA logging during the sweep

# Set to an existing directory to resume/extend a previous sweep; results
# accumulate here across repeated runs of this script.
RUN_DIR="onevsrest_sweep_results"

# ============================================================================

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"
mkdir -p "${RUN_DIR}"
echo "Results will be written to ${RUN_DIR}"

# Always pass --device, even when DEVICE="": argparse then sets args.device
# to "", and `self.device or (...)` in TorchL1LogisticRegression.fit treats
# an empty string as falsy, so it still auto-detects. This avoids building an
# optional-args array, which bash 3.2 (macOS's default /bin/bash) mishandles
# under `set -u` when empty ("${arr[@]}" raises "unbound variable").

N_RUN=0
N_SKIPPED=0
N_TOTAL=$(( ${#IMPORTANT_ACT_PKLS[@]} * ${#C_VALUES[@]} ))
INPUT_BASENAME="$(basename "${INPUT_FILE}" .parquet)"

for IAP in "${IMPORTANT_ACT_PKLS[@]}"; do
    IAP_BASENAME="$(basename "${IAP}" .pkl)"
    for C in "${C_VALUES[@]}"; do
        N_RUN=$((N_RUN + 1))
        C_STR="${C//./p}"
        expected_metrics="${RUN_DIR}/${INPUT_BASENAME}_${TARGET_COL}_${IAP_BASENAME}_C${C_STR}_onevsrest_torch_metrics.json"
        if [ -f "${expected_metrics}" ]; then
            echo "[${N_RUN}/${N_TOTAL}] Skipping important_act_pkl=${IAP} C=${C} (already processed)"
            N_SKIPPED=$((N_SKIPPED + 1))
            continue
        fi

        echo ""
        echo "=== [${N_RUN}/${N_TOTAL}] onevsrest: important_act_pkl=${IAP} C=${C} ==="
        python "${SRC_DIR}/robustness/probe_onevsrest_torch.py" \
            --input_file "${INPUT_FILE}" \
            --target_col "${TARGET_COL}" \
            --important_act_pkl "${IAP}" \
            --min_token_freq "${MIN_TOKEN_FREQ}" \
            --min_latent_freq "${MIN_LATENT_FREQ}" \
            --C "${C}" \
            --max_iter "${MAX_ITER}" \
            --top_k "${TOP_K}" \
            --cv "${CV_FOLDS}" \
            --backend "${BACKEND}" \
            --torch_verbose "${TORCH_VERBOSE}" \
            --device "${DEVICE}" \
            --output_dir "${RUN_DIR}"
    done
done

echo ""
echo "=== Sweep complete. Aggregating results ==="
python "${SRC_DIR}/robustness/aggregate_onevsrest_sweep_results.py" --results_dir "${RUN_DIR}"

echo ""
echo "Done. Full results + summary tables in ${RUN_DIR}"
