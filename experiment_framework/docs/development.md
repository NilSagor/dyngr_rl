
---

# 🛠️ `experiment_framework/docs/development.md`

```markdown
# 👨‍💻 Development Guide

This document covers architecture, extending the framework, debugging, and contribution standards.

---

## 🏗️ Architecture Overview

train_v5.py
↓
load_config() → resolve_placeholders() → apply_overrides() → build_experiment_config()
↓
get_runner(config['model']['name']) ← RUNNER_REGISTRY lookup
↓
Runner Lifecycle:
create_data_pipeline()
setup_model()
_log_model_status()
_profile_model() [optional]
run() → train/val/test loop → save metrics & checkpoints


All runners inherit from `src/experiments/runner/base_runner.py`.

---

## 🧩 Runner Interface

Override these methods in your custom runner:

| Method | Purpose | Called When |
|--------|---------|-------------|
| `create_data_pipeline()` | Load data, build samplers/loaders | Start of `run()` |
| `setup_model(model, pipeline)` | Inject features, neighbor finders, graph | After model init |
| `_log_model_status(model)` | Print architecture config | Before training |
| `_profile_model(model, pipeline)` | Compute FLOPs/latency | If `profiling.enabled=true` |
| `_get_forward_inputs(batch, model, pipeline)` | Map DataLoader batch → model args | During train/val loops |

> 💡 The base runner handles device placement, checkpointing, metric logging, and early stopping.

---

## ➕ How to Add a New Model (5 Steps)

### 1. Create Model Class
```python
# src/models/my_model/my_model.py
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Define layers...
    
    def forward(self, src, dst, times, **kwargs):
        # Return predictions & loss
        pass

```

### 2 create runner
```python
# src/experiments/runner/my_model_runner.py
from .base_runner import BaseRunner

class MyModelRunner(BaseRunner):
    def create_data_pipeline(self):
        # Return your data pipeline object
        pass

    def setup_model(self, model, pipeline):
        # Inject features, samplers, etc.
        pass

```

## 3. Register in __init__.py

```python
# src/experiments/runner/__init__.py
from .my_model_runner import MyModelRunner

RUNNER_REGISTRY = {
    ...
    "MyModel": MyModelRunner,
}



## 4. Add Config Template

```python
# configs/baseline/my_model.yaml
model:
  name: "MyModel"
  # ... your hyperparams ...
```


# 5. Test
```bash
python experiment_framework/src/experiments/train_v5.py \
  -c configs/baseline/my_model.yaml \
  --override training.max_epochs=1 experiment.debug=true

```
