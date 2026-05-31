#!/bin/bash
# Run multi-model evaluation ensuring all tasks are covered

set -e

echo "=========================================="
echo "Multi-Model Evaluation with All Tasks"
echo "=========================================="
echo ""

# Default values
DATASET="${1:-data/eval_tasks.jsonl}"
MODELS="${2:-small}"
LIMIT="${3:-30}"
OUTPUT="${4:-results_multi_all_tasks.json}"

echo "Dataset: $DATASET"
echo "Models: $MODELS"
echo "Limit: $LIMIT samples (balanced across all task types)"
echo "Output: $OUTPUT"
echo ""

# Run multi-model evaluation with balanced sampling
echo "Running multi-model evaluation..."
python3 -m src.eval_multi_model \
    --tasks "$DATASET" \
    --models "$MODELS" \
    --tool-based \
    --raw-llm \
    --balanced \
    --limit "$LIMIT" \
    --out "$OUTPUT"

echo ""
echo "Generating comparison report..."
python3 scripts/analyze_multi_model.py --results "$OUTPUT"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT"
echo "=========================================="
