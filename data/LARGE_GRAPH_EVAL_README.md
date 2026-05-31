# Large Graph Evaluation Dataset

This directory contains the large graph evaluation dataset (`eval_tasks_large.jsonl`) for testing graph reasoning models on significantly larger graphs.

## Dataset Overview

The large graph evaluation dataset contains **224 task samples** across 4 large graphs:

### Graph Sizes

- **gnp-2k**: ~2,000 nodes, ~10,000 edges
- **gnp-5k**: ~5,000 nodes, ~37,000 edges  
- **gnp-10k**: ~10,000 nodes, ~100,000 edges
- **gnp-20k**: ~20,000 nodes, ~200,000 edges

### Task Distribution

- **Shortest Path**: 200 samples (50 per graph)
- **Diameter**: 12 samples (3 per graph)
- **Connected Components**: 12 samples (3 per graph)

## Why Large Graphs?

Large graphs are essential for evaluating:
1. **Scalability**: How well models handle graphs beyond context window limits
2. **Tool Efficiency**: Whether models can minimize tool calls on larger graphs
3. **Real-world Applicability**: Testing on graph sizes closer to production use cases
4. **Performance**: Measuring latency and computational efficiency

## Usage

### Generate the Dataset

```bash
# Default: 50 shortest path, 3 diameter, 3 components per graph
python3 scripts/build_eval_dataset_large.py

# Customize task counts
python3 scripts/build_eval_dataset_large.py \
    --shortest-path-per-graph 100 \
    --diameter-per-graph 5 \
    --components-per-graph 5 \
    --seed 12345 \
    --out data/custom_large_eval.jsonl

# Use specific graphs only
python3 scripts/build_eval_dataset_large.py \
    --graphs gnp-10k gnp-20k \
    --shortest-path-per-graph 50
```

### Evaluate Models on Large Graphs

```bash
# Quick evaluation (first 50 samples)
python3 -m src.eval_pipeline \
    --tasks data/eval_tasks_large.jsonl \
    --sft checkpoints/sft_v3 \
    --limit 50 \
    --out results_large_eval.json

# Full evaluation (all 224 samples - may take longer)
python3 -m src.eval_pipeline \
    --tasks data/eval_tasks_large.jsonl \
    --sft checkpoints/sft_v3 \
    --rlvr checkpoints/rlvr \
    --limit 224 \
    --out results_large_eval_full.json
```

### Compare Small vs Large Graph Performance

```bash
# Evaluate on small graphs
python3 -m src.eval_pipeline \
    --tasks data/eval_tasks.jsonl \
    --sft checkpoints/sft_v3 \
    --limit 100 \
    --out results_small.json

# Evaluate on large graphs
python3 -m src.eval_pipeline \
    --tasks data/eval_tasks_large.jsonl \
    --sft checkpoints/sft_v3 \
    --limit 100 \
    --out results_large.json

# Compare results
python3 -c "
import json
small = json.load(open('results_small.json'))
large = json.load(open('results_large.json'))
print('Small graphs:', len(small), 'samples')
print('Large graphs:', len(large), 'samples')
"
```

## Dataset Format

Same format as the standard evaluation dataset. Each line is a JSON object:

```json
{
  "graph_id": "gnp-10k",
  "task": "shortest_path",
  "source": 1234,
  "target": 5678,
  "prompt": "Find the shortest path between node 1234 and node 5678 in graph 'gnp-10k'.",
  "ground_truth": {
    "path": [1234, 2345, 3456, 5678],
    "length": 3
  }
}
```

## Performance Considerations

⚠️ **Note**: Large graph evaluation takes significantly longer than small graphs:

- **gnp-2k**: ~2-5 seconds per task
- **gnp-5k**: ~5-15 seconds per task
- **gnp-10k**: ~15-30 seconds per task
- **gnp-20k**: ~30-60+ seconds per task

For quick testing, use `--limit` to evaluate a subset:
```bash
python3 -m src.eval_pipeline --tasks data/eval_tasks_large.jsonl --limit 20
```

## Graph Generation

The large graphs are Erdős–Rényi G(n,p) random graphs generated using:

```bash
# Generate 5k node graph
python3 scripts/generate_graph.py --n 5000 --p 0.003 --out data/gnp_5k.edgelist

# Generate 10k node graph
python3 scripts/generate_graph.py --n 10000 --p 0.002 --out data/gnp_10k.edgelist

# Generate 20k node graph
python3 scripts/generate_graph.py --n 20000 --p 0.001 --out data/gnp_20k.edgelist
```

Graphs are automatically registered in `src/data/graphs.py` when generated.

## Quality Assurance

- ✓ No duplicate node pairs within the same graph
- ✓ All samples have valid ground truth
- ✓ Balanced distribution across graphs
- ✓ Diverse path lengths (1-5+ hops)
- ✓ Different seed from training data (9999 vs 0)

