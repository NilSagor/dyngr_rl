from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
import csv
from datetime import datetime
import torch
import json
import numpy as np


class ExperimentLogger:
    """Handles result logging to CSV."""

    def __init__(
            self, 
            log_dir: str, 
            csv_filename: str = "all_results.csv",
            save_walk_data: bool = True,
            save_memory_traces: bool = True,
            save_ode_trajectories: bool = True,
        ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # self.csv_path = self.log_dir.parent / csv_filename
        self.csv_path = self.log_dir / csv_filename
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

               
        
        
        # Analysis subdirectories
        self.walks_dir = self.log_dir / "walks"
        self.memory_dir = self.log_dir / "memory"
        self.ode_dir = self.log_dir / "ode"
        self.negatives_dir = self.log_dir / "negatives"
        
        if save_walk_data:
            self.walks_dir.mkdir(exist_ok=True)
        if save_memory_traces:
            self.memory_dir.mkdir(exist_ok=True)
        if save_ode_trajectories:
            self.ode_dir.mkdir(exist_ok=True)
        self.negatives_dir.mkdir(exist_ok=True)

    def log(
        self,
        config: Dict,
        test_results: List[Dict],
        training_time: float,
        model: torch.nn.Module,
        count_trainable_only: bool = True,
        # New optional analysis data
        best_val_metric: Optional[float] = None, 
        monitor_name: str = 'val_ap',
        walk_stats: Optional[Dict] = None,           # TAWR walk statistics
        memory_trace: Optional[List[torch.Tensor]] = None,  # Prototype attention over epochs
        ode_trajectory: Optional[Dict] = None,       # ST-ODE dynamics
        negative_stats: Optional[Dict] = None,       # Hard negative mining stats
        cooccurrence_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """Log experiment results to CSV."""
        if not test_results:
            raise ValueError("test_results must be non-empty")
        
        
        
        # Safely extract metrics, warn if missing
        first_result = test_results[0]
        metrics = {
            'test_accuracy': first_result.get('test_accuracy'),
            'test_ap': first_result.get('test_ap'),
            'test_auc': first_result.get('test_auc'),
            'test_loss': first_result.get('test_loss'),
        }
        for name, value in metrics.items():
            if value is None:
                logger.warning(f"Metric '{name}' not found in test results")

        # Parameter count
        if count_trainable_only:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in model.parameters())

        row = {
            'model': config['model']['name'],
            'dataset': config['data']['dataset'],
            'evaluation_type': config['data']['evaluation_type'],
            'negative_sampling_strategy': config['data']['negative_sampling_strategy'],
            'seed': config['experiment']['seed'],
            'best_val_metric': best_val_metric,
            'monitor_metric': monitor_name,
            'test_accuracy': metrics['test_accuracy'],
            'test_ap': metrics['test_ap'],
            'test_auc': metrics['test_auc'],
            'test_loss': metrics['test_loss'],
            'training_time': training_time,
            'num_parameters': num_params,
            'timestamp': datetime.now().isoformat(),
        }


        # ── Inject architecture & training params into the row ──
        model_cfg = config.get('model', {})
        train_cfg = config.get('training', {})

        # Architecture flags (all boolean/enum values)
        for key in ['use_memory', 'enable_walk', 'enable_restart', 'enable_neighbor_cooc',
                    'use_explicit_co_gnn', 'n_layers', 'n_heads', 'dropout',
                    'walk_length', 'num_walks', 'n_neighbors']:
            row[key] = model_cfg.get(key)
        
        # Training hyperparams
        for key in ['learning_rate', 'weight_decay', 'batch_size']:
            row[key] = train_cfg.get(key)       
        
        row['best_val_metric'] = best_val_metric
        row['monitor_metric'] = monitor_name

        file_exists = self.csv_path.exists()
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.info(f"Results saved to {self.csv_path}")

        if walk_stats:
            self._save_walk_analysis(walk_stats, config)
        
        if memory_trace:
            self._save_memory_trace(memory_trace, config)
            
        if ode_trajectory:
            self._save_ode_dynamics(ode_trajectory, config)
            
        if negative_stats:
            self._save_negative_mining_stats(negative_stats, config)
            
        if cooccurrence_matrix is not None:
            np.save(self.log_dir / "cooccurrence_matrix.npy", cooccurrence_matrix)

    def _save_walk_analysis(self, walk_stats: Dict, config: Dict) -> None:
        """Save TAWR walk statistics: length distributions, restart probabilities."""
        walk_data = {
            'model': config['model']['name'],
            'num_prototypes': config['model'].get('num_prototypes'),
            'num_walks': config['model'].get('num_walks'),
            'walk_length': config['model'].get('walk_length'),
            'length_distribution': walk_stats.get('length_distribution', []),
            'restart_probs': walk_stats.get('restart_probabilities', []),
            'learned_temperatures': walk_stats.get('temperatures', []),
            'timestamp': datetime.now().isoformat(),
        }
        with open(self.walks_dir / f"walk_stats_{config['experiment']['seed']}.json", 'w') as f:
            json.dump(walk_data, f, indent=2)
        logger.info(f"Walk analysis saved to {self.walks_dir}")
    
    
    def _save_memory_trace(self, memory_trace: List[torch.Tensor], config: Dict) -> None:
        """Save prototype attention evolution over training."""
        # Stack list of tensors [num_epochs, num_prototypes, feature_dim]
        trace_tensor = torch.stack(memory_trace) if isinstance(memory_trace, list) else memory_trace
        torch.save({
            'attention_weights': trace_tensor,
            'num_prototypes': config['model'].get('num_prototypes'),
            'epochs': len(memory_trace),
        }, self.memory_dir / f"memory_trace_{config['experiment']['seed']}.pt")
        logger.info(f"Memory trace saved: shape {trace_tensor.shape}")
    
    
    def _save_ode_dynamics(self, ode_trajectory: Dict, config: Dict) -> None:
        """Save ST-ODE dynamics: Dirichlet energy, smoothness metrics over time."""
        ode_data = {
            'dirichlet_energy': ode_trajectory.get('dirichlet_energy', []),
            'node_embeddings_over_time': ode_trajectory.get('embeddings', []),
            'time_points': ode_trajectory.get('time_points', []),
            'ode_interval': config['model'].get('ode_update_interval'),
            'solver_type': config['model'].get('ode_solver', 'dopri5'),
            'timestamp': datetime.now().isoformat(),
        }
        torch.save(ode_data, self.ode_dir / f"ode_dynamics_{config['experiment']['seed']}.pt")
        logger.info(f"ODE dynamics saved")
    
    def _save_negative_mining_stats(self, negative_stats: Dict, config: Dict) -> None:
        """Save hard negative mining statistics vs random negatives."""
        neg_data = {
            'hard_negative_ratio': negative_stats.get('hard_ratio', []),
            'negative_loss_contribution': negative_stats.get('loss_contrib', []),
            'mining_strategy': config['data'].get('negative_sampling_strategy'),
            'num_negatives': config['data'].get('num_negatives'),
            'epochs': len(negative_stats.get('hard_ratio', [])),
            'timestamp': datetime.now().isoformat(),
        }
        with open(self.negatives_dir / f"negative_stats_{config['experiment']['seed']}.json", 'w') as f:
            json.dump(neg_data, f, indent=2)