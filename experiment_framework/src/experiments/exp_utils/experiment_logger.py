from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
import csv
from datetime import datetime
import torch



class ExperimentLogger:
    """Handles result logging to CSV."""

    def __init__(self, log_dir: str, csv_filename: str = "all_results.csv"):
        self.log_dir = Path(log_dir)
        self.csv_path = self.log_dir.parent / csv_filename
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        config: Dict,
        test_results: List[Dict],
        training_time: float,
        model: torch.nn.Module,
        count_trainable_only: bool = True,
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
            'test_accuracy': metrics['test_accuracy'],
            'test_ap': metrics['test_ap'],
            'test_auc': metrics['test_auc'],
            'test_loss': metrics['test_loss'],
            'training_time': training_time,
            'num_parameters': num_params,
            'timestamp': datetime.now().isoformat(),
        }

        file_exists = self.csv_path.exists()
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.info(f"Results saved to {self.csv_path}")