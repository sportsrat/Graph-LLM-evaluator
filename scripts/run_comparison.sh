#!/bin/bash
# Run comparison between raw LLM and tool-based approach

set -e

echo "=========================================="
echo "Raw LLM vs Tool-Based Comparison"
echo "=========================================="
echo ""

# Default values
DATASET="${1:-data/eval_tasks.jsonl}"
LIMIT="${2:-20}"
OUTPUT="${3:-results_comparison.json}"

echo "Dataset: $DATASET"
echo "Limit: $LIMIT samples"
echo "Output: $OUTPUT"
echo ""

# Step 1: Run evaluation with both raw LLM and tool-based
echo "Step 1: Running evaluation..."
python3 -m src.eval_pipeline \
    --tasks "$DATASET" \
    --sft checkpoints/sft_v3 \
    --raw-llm \
    --limit "$LIMIT" \
    --out "$OUTPUT"

echo ""
echo "Step 2: Generating comparison report..."
python3 scripts/compare_raw_vs_tool.py --results "$OUTPUT"

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "Results saved to: $OUTPUT"
echo "=========================================="
