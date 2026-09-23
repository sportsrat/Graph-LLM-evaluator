# G1-Scale: Tool-Augmented LLM for Graph Reasoning

A system for evaluating and running Large Language Models (LLMs) on graph reasoning tasks using a Graph Traversal Tool (GTT). It enables LLMs to perform shortest path routing, graph diameter calculation, and connected component detection across network graphs.

---

## Quick Start

### 1. Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_device.py
```

### 2. Streamlit Web App
Launch the interactive side-by-side model comparison interface:
```bash
streamlit run streamlit_test_model.py
```


### 3. Command-Line Experiments
```bash
# Shortest path (Instant stub LLM)
python -m src.run_experiment shortest_path --graph ego-facebook-medium --src 0 --dst 14 --backend stub

# Shortest path (TinyLlama 1.1B model)
python -m src.run_experiment shortest_path --graph ego-facebook-medium --src 0 --dst 14 --backend tiny-llama --out results.json

# Graph diameter
python -m src.run_experiment diameter --graph ego-facebook-medium --backend stub

# Connected components
python -m src.run_experiment components --graph ca-grqc-small --backend stub
```

---

## Key Features

- **Tool-Augmented Reasoning**: ReAct agent using NetworkX for exact graph calculations.
- **Multi-Model Support**: TinyLlama, Qwen2 (0.5B/1.5B/7B), Mistral, Phi-2, and Gemma.
- **Interactive UI**: Streamlit web interface for side-by-side GTT vs raw LLM comparison.
- **Training & Evaluation Pipelines**: Supervised Fine-Tuning (SFT), RLVR, and multi-model benchmark evaluation scripts.
- **Windows Optimized**: Automatic CUDA GPU acceleration with CPU fallback and local HuggingFace cache handling.

---

## Project Structure

```
.
├── src/
│   ├── agent.py                 # Tool-use agent wrapper
│   ├── agent_llm.py            # ReAct loop implementation
│   ├── eval.py                 # Metric calculations
│   ├── eval_pipeline.py        # Evaluation pipeline
│   ├── eval_multi_model.py     # Multi-model benchmarking
│   ├── eval_raw_llm.py         # Raw LLM evaluation
│   ├── run_experiment.py       # Experiment runner CLI
│   ├── train_sft.py            # SFT training script
│   ├── train_rl.py             # RLVR training script
│   ├── data/graphs.py          # Graph dataset loading
│   ├── gtt/tool.py             # Graph Traversal Tool (GTT)
│   └── models/
│       ├── llm.py              # LLM backends (Stub, Transformers)
│       └── model_registry.py  # Model registry
├── scripts/
│   ├── check_device.py         # Hardware device diagnostic
│   ├── test_model_interactive.py  # Interactive CLI test script
│   ├── build_tasks.py          # Dataset task generator
│   └── build_sft_dataset.py    # Training dataset builder
├── data/                       # Graph edge lists and dataset JSONL files
├── notebooks/demo.ipynb        # Jupyter notebook demo
└── streamlit_test_model.py    # Streamlit web app
```

---

## Training & Evaluation

### Supervised Fine-Tuning (SFT)
```bash
# 1. Generate tasks & SFT dataset
python scripts/build_tasks.py --per-graph 200 --out data/tasks.jsonl
python scripts/build_sft_dataset.py --tasks data/tasks.jsonl --out data/sft_traces.jsonl

# 2. Train model
python -m src.train_sft --train-path data/sft_traces.jsonl --output-dir checkpoints/sft_v2 --epochs 3
```

### Multi-Model Benchmarking
```bash
python -m src.eval_multi_model --tasks data/eval_tasks.jsonl --models small --tool-based --raw-llm --limit 30 --out results_multi.json
```

---

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ReAct Agent    │ ◄─── LLM Backend (TinyLlama, Qwen, Stub, etc.)
└────────┬────────┘
         │ TOOL_CALL: {"action": "shortest_path", "args": {"source": 0, "target": 14}}
         ▼
┌─────────────────┐
│ Graph Traversal │ ◄─── NetworkX Backend (Exact algorithm execution)
│   Tool (GTT)    │
└─────────────────┘
```

## ss: 
Qwen model : fb dataset
<img width="1911" height="968" alt="image" src="https://github.com/user-attachments/assets/cd2fe0c6-b064-4413-9ee6-5765fd66f22d" />

qwen model: 2k nodes
<img width="1913" height="897" alt="image" src="https://github.com/user-attachments/assets/d98cf692-2f9f-4006-a0b5-477f53794edb" />



