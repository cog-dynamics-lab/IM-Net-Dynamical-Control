#!/usr/bin/env bash
# experiments/05_stress_test.sh
#
# Section 5.4 / Fig. 9: extreme adversarial deadlock stress test.
#   Sweep conflict_scale in {1.0, 2.0, 5.0} for
#       single / fixed_weights / pcgrad / mgda / meta
#   using main_gpu.py --simulation --conflict_scale c
#   and aggregate Best Recall@20 into results_reproduction/fig9_stress_test.csv.
#
# Usage:
#   bash experiments/05_stress_test.sh
#   EPOCHS=5 bash experiments/05_stress_test.sh   # quick smoke test
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_YELP="${DATA_YELP:-$REPO_ROOT/data/yelp_processed_for_meta}"
DATASET="${DATASET:-yelp}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-2026}"
MODEL="${MODEL:-LightGCN}"
OUT_CSV="${OUT_CSV:-$REPO_ROOT/results_reproduction/fig9_stress_test.csv}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/results_reproduction/training_logs/stress_test}"

mkdir -p "$(dirname "$OUT_CSV")" "$LOG_DIR"

export PYTHONPATH="$REPO_ROOT/core:${PYTHONPATH:-}"

SCALES=(1.0 2.0 5.0)
STRATEGIES=(single fixed_weights pcgrad mgda meta)

echo "conflict_scale,strategy,recall_at_20" > "$OUT_CSV"

for scale in "${SCALES[@]}"; do
    for strat in "${STRATEGIES[@]}"; do
        log="$LOG_DIR/${DATASET}_${strat}_c${scale}_s${SEED}.log"
        echo ">>> [${DATASET}/${strat}/c=${scale}] -> ${log}"

        python core/main_gpu.py \
            --dataset "$DATASET" \
            --data_path "$DATA_YELP" \
            --model_name "$MODEL" \
            --mode "$strat" \
            --epochs "$EPOCHS" \
            --seed "$SEED" \
            --simulation \
            --conflict_scale "$scale" \
            > "$log" 2>&1 || {
                echo "[WARN] ${strat}/c=${scale} failed; see ${log}" >&2
                continue
            }

        line=$(grep -E '\[FINAL_RESULT\]' "$log" | tail -n 1 || true)
        if [[ -z "$line" ]]; then
            echo "[WARN] No FINAL_RESULT for ${strat}/c=${scale}" >&2
            continue
        fi
        recall=$(echo "$line" | sed -nE 's/.*Best Recall@20: ([0-9.]+).*/\1/p')
        echo "${scale},${strat},${recall}" >> "$OUT_CSV"
        echo "    Recall@20=${recall}"
    done
done

echo
echo "Fig.9 CSV saved to: $OUT_CSV"
column -s, -t "$OUT_CSV" 2>/dev/null || cat "$OUT_CSV"