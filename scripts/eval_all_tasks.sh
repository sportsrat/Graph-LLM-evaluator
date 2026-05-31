#!/bin/bash
# Run evaluation ensuring all tasks (shortest_path, diameter, components) are covered

set -e

echo "=========================================="
echo "Evaluation with All Tasks Coverage"
echo "=========================================="
echo ""

# Default values
DATASET="${1:-data/eval_tasks.jsonl}"
LIMIT="${2:-30}"
OUTPUT="${3:-results_all_tasks.json}"

echo "Dataset: $DATASET"
echo "Limit: $LIMIT samples (balanced across all task types)"
echo "Output: $OUTPUT"
echo ""

# Run evaluation with balanced sampling
echo "Running evaluation with balanced task distribution..."
python3 -m src.eval_pipeline \
    --tasks "$DATASET" \
    --sft checkpoints/sft_v3 \
    --raw-llm \
    --balanced \
    --limit "$LIMIT" \
    --out "$OUTPUT"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT"
echo ""
echo "To view comparison:"
echo "  python3 scripts/compare_raw_vs_tool.py --results $OUTPUT"
echo "=========================================="
