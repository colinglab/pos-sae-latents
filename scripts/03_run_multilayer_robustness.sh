#!/bin/bash
# Re-runs the compact-feature multiclass probes (CV and train/test) for
# additional SAE layers, in parallel (Appendix A.1, Figure 8). Requires that
# layer-wise-latents/latents_{train,test}_*_{layer}.parquet and
# important_act_per_pos_layer{layer}.pkl (from coverage_analysis.py) already
# exist for each layer.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"

TARGET_COL="upos"
INPUT_DIR="layer-wise-latents"
OUTPUT_DIR="probing-results-multilayer"
LAYERS=(2 15)

mkdir -p "${OUTPUT_DIR}"

pids=()
labels=()

for layer in "${LAYERS[@]}"; do
    important_pkl="important_act_per_pos_layer${layer}.pkl"
    if [ ! -f "${important_pkl}" ]; then
        echo "Skipping layer ${layer}: ${important_pkl} not found"
        continue
    fi

    train_file=$(ls ${INPUT_DIR}/latents_train_*_${layer}.parquet 2>/dev/null | head -n 1)
    if [ -z "${train_file}" ]; then
        echo "Skipping layer ${layer}: no matching ${INPUT_DIR}/latents_train_*_${layer}.parquet"
        continue
    fi
    train_basename=$(basename "${train_file}" .parquet)
    test_file="${INPUT_DIR}/${train_basename/latents_train_/latents_test_}.parquet"

    # --- Cross-validation experiment (probe_multiclass_cv.py) ---
    cv_expected_output="${OUTPUT_DIR}/${train_basename}_${TARGET_COL}_multiclass_df_filtered.parquet"
    if [ -f "${cv_expected_output}" ]; then
        echo "Skipping CV layer ${layer} (already processed)"
    else
        echo "Launching CV layer ${layer} (${train_file})..."
        python "${SRC_DIR}/probing/probe_multiclass_cv.py" \
            --input_file "${train_file}" \
            --target_col "${TARGET_COL}" \
            --layer_n "${layer}" \
            --output_dir "${OUTPUT_DIR}" \
            > "${OUTPUT_DIR}/${train_basename}_${TARGET_COL}_multiclass_cv.log" 2>&1 &
        pids+=($!)
        labels+=("CV layer ${layer}")
    fi

    # --- Train/test experiment (probe_multiclass_train_test.py) ---
    if [ ! -f "${test_file}" ]; then
        echo "Skipping train/test layer ${layer}: no matching test file ${test_file}"
    else
        tt_expected_output="${OUTPUT_DIR}/${train_basename}_${TARGET_COL}_multiclass_test_predictions.parquet"
        if [ -f "${tt_expected_output}" ]; then
            echo "Skipping train/test layer ${layer} (already processed)"
        else
            echo "Launching train/test layer ${layer} (${train_file} / ${test_file})..."
            python "${SRC_DIR}/probing/probe_multiclass_train_test.py" \
                --train_file "${train_file}" \
                --test_file "${test_file}" \
                --target_col "${TARGET_COL}" \
                --layer_n "${layer}" \
                --output_dir "${OUTPUT_DIR}" \
                > "${OUTPUT_DIR}/${train_basename}_${TARGET_COL}_multiclass_traintest.log" 2>&1 &
            pids+=($!)
            labels+=("train/test layer ${layer}")
        fi
    fi
done

if [ "${#pids[@]}" -eq 0 ]; then
    echo "Nothing to run."
    exit 0
fi

echo "Waiting on ${#pids[@]} job(s) in parallel: ${labels[*]}"

failures=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "Done: ${labels[$i]}"
    else
        echo "FAILED: ${labels[$i]} (see corresponding .log in ${OUTPUT_DIR})"
        failures=$((failures + 1))
    fi
done

if [ "${failures}" -gt 0 ]; then
    echo "${failures} job(s) failed."
    exit 1
fi

echo "All jobs completed successfully."
