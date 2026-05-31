#!/bin/bash
# Quick script to run evaluation on the new SFT checkpoint
# Usage: ./scripts/run_eval.sh [limit] [output_file]

LIMIT=${1:-30}
OUTPUT=${2:-results_sft_v2.json}

echo "Running evaluation with limit=$LIMIT, output=$OUTPUT"
echo "Using checkpoint: checkpoints/sft_v2"

.venv/bin/python -m src.eval_pipeline \
    --tasks data/tasks.jsonl \
    --sft checkpoints/sft_v2 \
    --limit $LIMIT \
    --out $OUTPUT

echo ""
echo "Generating summary..."
.venv/bin/python scripts/summarize_results.py --results $OUTPUT
