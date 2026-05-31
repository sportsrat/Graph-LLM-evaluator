# Evaluation Dataset

This directory contains the evaluation dataset (`eval_tasks.jsonl`) for testing the graph reasoning models.

## Dataset Overview

The evaluation dataset contains **270 task samples** across 4 graphs and 3 task types:

- **Shortest Path**: 190 samples (diverse path lengths from 1-6 hops)
- **Diameter**: 40 samples (10 per graph)
- **Connected Components**: 40 samples (10 per graph)

### Graph Coverage

- `ego-facebook-small`: 65 samples
- `ego-facebook-medium`: 70 samples
- `ca-grqc-small`: 65 samples
- `soc-advogato-small`: 70 samples

## Dataset Characteristics

1. **No Training Overlap**: Uses seed 9999 (different from training seed 0)
2. **Diverse Difficulty**: Shortest path tasks include a mix of:
   - Direct neighbors (length 1)
   - Medium paths (length 2-3)
   - Long paths (length 4-6)
3. **Unique Samples**: No duplicate node pairs within the same graph
4. **Balanced Distribution**: Equal representation across graphs and task types

## Usage

### Generate the Dataset

```bash
python3 scripts/build_eval_dataset.py
```

Customize the dataset:
```bash
python3 scripts/build_eval_dataset.py \
    --shortest-path-per-graph 100 \
    --diameter-per-graph 20 \
    --components-per-graph 20 \
    --seed 12345 \
    --out data/custom_eval.jsonl
```

### Evaluate Models

Use the evaluation pipeline with the evaluation dataset:

```bash
# Evaluate rule-based baseline
python3 -m src.eval_pipeline --tasks data/eval_tasks.jsonl --limit 50

# Evaluate SFT model
python3 -m src.eval_pipeline \
    --tasks data/eval_tasks.jsonl \
    --sft checkpoints/sft_v3 \
    --limit 100 \
    --out results_eval.json

# Evaluate both SFT and RLVR models
python3 -m src.eval_pipeline \
    --tasks data/eval_tasks.jsonl \
    --sft checkpoints/sft_v3 \
    --rlvr checkpoints/rlvr \
    --limit 270 \
    --out results_full_eval.json
```

## Dataset Format

Each line in `eval_tasks.jsonl` is a JSON object with:

```json
{
  "graph_id": "ego-facebook-small",
  "task": "shortest_path",
  "source": 1,
  "target": 2,
  "prompt": "Find the shortest path between node 1 and node 2 in graph 'ego-facebook-small'.",
  "ground_truth": {
    "path": [1, 2],
    "length": 1
  }
}
```

### Task Types

1. **shortest_path**: Requires `source` and `target` fields
   - Ground truth: `{"path": [...], "length": N}` or `{"path": null, "length": null, "error": "no_path"}`

2. **diameter**: No source/target required
   - Ground truth: `{"diameter": N, "note": "..."}`

3. **components**: No source/target required
   - Ground truth: `{"component_count": N}`

## Quality Assurance

- ✓ No duplicate node pairs within the same graph
- ✓ All samples have valid ground truth
- ✓ Balanced distribution across graphs and tasks
- ✓ Diverse path lengths for comprehensive evaluation
- ✓ Different seed from training data (9999 vs 0)

