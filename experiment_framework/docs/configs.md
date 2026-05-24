# Config schema, placeholders, inheritance
# ⚙️ Configuration Guide

This document explains how to read, modify, and extend the YAML configuration system used by `train_v5.py`.

---

## 📁 Directory Structure
configs/
├── baseline/ # Reproduction-ready configs for established models
├── hicost_dev/ # HiCoST variant configs (v1, v2, ...)
├── ablation/ # Component removal & controlled variants
├── param_sweep/ # Hyperparameter search templates
└── sensitivity_config.yaml # Advanced sweep runner config



> 💡 **Rule of thumb**: Never edit baseline configs directly. Copy to `ablation/` or use `--override` at runtime.

---

## 🧩 Core Configuration Sections

Every config is split into logical namespaces. The system validates and merges them before training.

| Section | Purpose | Key Fields |
|---------|---------|------------|
| `experiment` | Run metadata & control | `name`, `seed`, `precision`, `debug` |
| `data` | Dataset & evaluation setup | `dataset`, `evaluation_type`, `negative_sampling_strategy`, `train_ratio`, `unseen_ratio` |
| `model` | Architecture & hyperparameters | `name`, `memory_dim`, `use_memory`, `enable_walk`, `n_neighbors`, etc. |
| `training` | Optimization & callbacks | `batch_size`, `max_epochs`, `patience`, `learning_rate`, `monitor`, `mode` |
| `hardware` | Runtime resources | `gpus`, `num_workers`, `pin_memory` |
| `profiling` | FLOPs/latency tracking | `enabled` |
| `logging` | Output paths & checkpointing | `log_dir`, `checkpoint_dir`, `save_top_k` |

---

## 🔗 Placeholder & Variable System

Configs support `${...}` placeholders for dynamic resolution at runtime.

```yaml
experiment:
  name: "${model.name}_${data.dataset}_${experiment.seed}"
data:
  dataset: "wikipedia"