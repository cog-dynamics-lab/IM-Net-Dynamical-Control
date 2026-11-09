#!/usr/bin/env bash
# experiments/03_main_experiments.sh
#
# Section 5.2 / Table 2: run all six rows on Yelp and Amazon.
# Output: results_reproduction/table2_main_results.csv
# Logs:   results_reproduction/training_logs/table2_<dataset>_<strategy>_s<seed>.log
#
# Usage:
#   bash experiments/03_main_experiments.sh
#   EPOCHS=5 SEED=2026 bash experiments/03_main_experiments.sh   # quick smoke test
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_YELP="${DATA_YELP:-$REPO_ROOT/data/yelp_processed_for_meta}"
DATA_AMAZON="${DATA_AMAZON:-$REPO_ROOT/data/amazon_books_processedDataV3}"
EPOCHS="${EPOCHS:-400}"
SEED="${SEED:-2026}"
MODEL="${MODEL:-LightGCN}"
OUT_CSV="${OUT_CSV:-$REPO_ROOT/results_reproduction/table2_main_results.csv}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/results_reproduction/training_logs}"

mkdir -p "$(dirname "$OUT_CSV")" "$LOG_DIR"

export PYTHONPATH="$REPO_ROOT/core:${PYTHONPATH:-}"

# README strategy name -> main_gpu.py --mode
STRATEGIES=(single fixed_weights gradnorm pcgrad mgda meta)

echo "regime,strategy,dataset,recall_at_20,ndcg_at_20" > "$OUT_CSV"

for dataset in yelp amazon; do
    if [[ "$dataset" == "yelp" ]]; then
        data_path="$DATA_YELP"
    else
        data_path="$DATA_AMAZON"
    fi

    for strat in "${STRATEGIES[@]}"; do
        if [[ "$strat" == "single" ]]; then
            regime="Reference Upper Bound"
        else
            regime="Multi-Task Control"
        fi

        log="$LOG_DIR/table2_${dataset}_${strat}_s${SEED}.log"
        echo ">>> [${dataset}/${strat}] -> ${log}"

        python core/main_gpu.py \
            --dataset "$dataset" \
            --data_path "$data_path" \
            --model_name "$MODEL" \
            --mode "$strat" \
            --epochs "$EPOCHS" \
            --seed "$SEED" \
            > "$log" 2>&1 || {
                echo "[WARN] ${dataset}/${strat} failed; see ${log}" >&2
                continue
            }

        line=$(grep -E '\[FINAL_RESULT\]' "$log" | tail -n 1 || true)
        if [[ -z "$line" ]]; then
            echo "[WARN] No FINAL_RESULT for ${dataset}/${strat}" >&2
            continue
        fi

        recall=$(echo "$line" | sed -nE 's/.*Best Recall@20: ([0-9.]+).*/\1/p')
        ndcg=$(echo "$line"   | sed -nE 's/.*Best NDCG@20: ([0-9.]+).*/\1/p')
        echo "${regime},${strat},${dataset},${recall},${ndcg}" >> "$OUT_CSV"
        echo "    Recall@20=${recall}  NDCG@20=${ndcg}"
    done
done

echo
echo "Table 2 saved to: $OUT_CSV"
column -s, -t "$OUT_CSV" 2>/dev/null || cat "$OUT_CSV"