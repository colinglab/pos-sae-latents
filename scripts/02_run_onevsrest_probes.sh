#!/bin/bash
# RQ1: one-vs-rest binary probing classifiers per POS (paper Section 4.1,
# Figures 2/3). Feeds coverage_analysis.py for RQ2.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"

TARGET_COL="upos"
INPUT_DIR="layer-wise-latents"
OUTPUT_DIR="probing-results"

for input_file in ${INPUT_DIR}/latents_train_*.parquet; do
    basename=$(basename "${input_file}" .parquet)
    expected_output="${OUTPUT_DIR}/${basename}_${TARGET_COL}_df_filtered.parquet"

    if [ -f "${expected_output}" ]; then
        echo "Skipping ${input_file} (already processed)"
        continue
    fi

    echo "Processing ${input_file}..."
    python "${SRC_DIR}/probing/probe_onevsrest.py" \
        --input_file "${input_file}" \
        --target_col "${TARGET_COL}" \
        --output_dir "${OUTPUT_DIR}"
done
