from abc import ABC, abstractmethod
from typing import Dict

from exp_utils.trainer_setup import TrainerSetup

from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger
import torch

from src.utils.general_utils import set_seed
from src.experiments.exp_utils.model_factory import ModelFactory
from src.experiments.exp_utils.experiment_logger import ExperimentLogger
from src.experiments.exp_utils.trainer_setup import TrainerSetup
from src.experiments.exp_utils.analysis_callback import AnalysisCollector
from src.experiments.exp_utils.clear_callback import ClearCacheCallback
from lightning.pytorch.callbacks import EarlyStopping


class BaseRunner(ABC):
    """Abstract base class for model-specific training runners."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.start_time: Optional[datetime] = None
        self.analysis_collector: Optional[AnalysisCollector] = None
        # self.clear_callback: Optional[ClearCacheCallback] = None

    @abstractmethod
    def create_data_pipeline(self):
        """Create and return the data pipeline for this model.

        The pipeline must provide:
            - loaders: Dict[str, DataLoader] with keys 'train', 'val', 'test'
            - get_features() -> Dict with node/edge feature info
            - neighbor_finder attribute
            - (optional) other attributes needed by setup_model
        """
        pass

    @abstractmethod
    def setup_model(self, model: torch.nn.Module, pipeline) -> None:
        """Set up model-specific components from the pipeline.

        Args:
            model: The instantiated LightningModule.
            pipeline: The data pipeline object returned by create_data_pipeline().
        """
        pass

    def _log_model_status(self, model: torch.nn.Module) -> None:
        """Log model component status (overridable by subclasses)."""
        pass

    def _collect_additional_artifacts(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Collect model-specific analysis artifacts for logging."""
        return {}

    def _profile_model(self, model: torch.nn.Module, pipeline) -> None:
        """Optional profiling hook called after model setup if debug is enabled.
        
        Subclasses can override to compute FLOPs, parameter counts, etc.
        """
        pass
    
    
    
    def run(self) -> Dict[str, Any]:
        """Execute the full training and evaluation pipeline."""
        self.start_time = datetime.now()
        seed = self.config['experiment']['seed']
        set_seed(seed)

        logger.info(f"{'='*50}")
        logger.info(f"Starting training: {self.config['experiment']['name']}")
        logger.info(f"Model: {self.config['model']['name']}")
        logger.info(f"Seed: {seed}")
        logger.info(f"{'='*50}")

        # 1. Data pipeline
        pipeline = self.create_data_pipeline()
        logger.info(f"Data pipeline ready: {pipeline.num_nodes} nodes")

        # 2. Model creation
        features = pipeline.get_features()
        model = ModelFactory.create(self.config, features)

        # 3. Model-specific setup
        self.setup_model(model, pipeline)
        self._log_model_status(model)

        # Validate model is ready (optional)
        if hasattr(model, 'neighbor_finder'):
            assert model.neighbor_finder is not None, "Neighbor finder not set"

        # 4. Optional profiling (before training)
        if self.config['experiment'].get('debug', False):
            self._profile_model(model, pipeline)
        
        # 4. Analysis collector
        self.analysis_collector = AnalysisCollector()

        # 5. Trainer
        trainer = TrainerSetup.create(self.config, callbacks=[self.analysis_collector, ClearCacheCallback()])
        logger.info(f"Trainer max_epochs: {trainer.max_epochs}")

        # 6. Training
        logger.info("Starting training...")
        trainer.fit(
            model=model,
            train_dataloaders=pipeline.loaders['train'],
            val_dataloaders=pipeline.loaders['val'],
        )

        # 7. Testing with best checkpoint
        logger.info("Running evaluation with best checkpoint...")
        test_results = trainer.test(
            model=model,
            dataloaders=pipeline.loaders['test'],
            ckpt_path='best' if trainer.checkpoint_callback.best_model_path else None
        )

        # 8. Collect artifacts
        additional_artifacts = self._collect_additional_artifacts(model)

        # 9. Log results
        training_time = (datetime.now() - self.start_time).total_seconds()
        exp_logger = ExperimentLogger(self.config['logging']['log_dir'])

        exp_logger.log(
            config=self.config,
            test_results=test_results,
            training_time=training_time,
            model=model,
            walk_stats=self.analysis_collector.walk_stats,
            memory_trace=self.analysis_collector.memory_trace,
            ode_trajectory=self.analysis_collector.ode_trajectory,
            negative_stats=self.analysis_collector.negative_stats,
            cooccurrence_matrix=additional_artifacts.get('cooccurrence_matrix'),
        )

        # 10. Save final checkpoint
        final_path = Path(self.config['logging']['checkpoint_dir']) / 'final_model.ckpt'
        final_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(final_path))
        logger.info(f"Final model saved to {final_path}")

        logger.info(f"Training complete in {training_time:.1f}s")

        return {
            'test_results': test_results,
            'training_time': training_time,
            'model': model,
        }