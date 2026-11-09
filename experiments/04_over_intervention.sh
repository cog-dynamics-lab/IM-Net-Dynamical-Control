#!/usr/bin/env bash
# experiments/04_over_intervention.sh
#
# Section 5.3 / Fig. 8: over-intervention trap on Yelp.
#   Run main_gpu.py --mode meta --epochs 100 --force_meta_epoch 30,
#   parse per-epoch Recall@20 and Loss from the log,
#   write results_reproduction/fig8_yelp_pivot.csv.
#
# Usage:
#   bash experiments/04_over_intervention.sh
#   EPOCHS=100 FORCE_META_EPOCH=30 bash experiments/04_over_intervention.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_YELP="${DATA_YELP:-$REPO_ROOT/data/yelp_processed_for_meta}"
EPOCHS="${EPOCHS:-100}"
FORCE_META_EPOCH="${FORCE_META_EPOCH:-30}"
SEED="${SEED:-2026}"
MODEL="${MODEL:-LightGCN}"
OUT_CSV="${OUT_CSV:-$REPO_ROOT/results_reproduction/fig8_yelp_pivot.csv}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/results_reproduction/training_logs}"

mkdir -p "$(dirname "$OUT_CSV")" "$LOG_DIR"

export PYTHONPATH="$REPO_ROOT/core:${PYTHONPATH:-}"

log="$LOG_DIR/fig8_yelp_meta_ep${FORCE_META_EPOCH}_s${SEED}.log"
echo ">>> Fig.8 run -> ${log}"

python core/main_gpu.py \
    --dataset yelp \
    --data_path "$DATA_YELP" \
    --model_name "$MODEL" \
    --mode meta \
    --epochs "$EPOCHS" \
    --force_meta_epoch "$FORCE_META_EPOCH" \
    --seed "$SEED" \
    > "$log" 2>&1

# Parse the per-epoch line printed by main_gpu.py:
#   Epoch NNN | Loss: ... | R@20: ... | N@20: ...
# The separate "[Best Updated]" line has no Loss/R@20 fields, so it is not parsed.
{
    echo "epoch,loss,recall_at_20"
    sed -nE 's/^(\[Best Updated\] )?Epoch ([0-9]+) \| Loss: ([0-9.]+) \| R@20: ([0-9.]+).*/\2,\3,\4/p' "$log" \
        | sort -t, -k1,1n
} > "$OUT_CSV"

echo "Fig.8 CSV saved to: $OUT_CSV"
wc -l "$OUT_CSV"