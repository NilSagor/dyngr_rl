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
# from lightning.pytorch.callbacks import EarlyStopping


def _make_run_tag(config: Dict) -> str:
        """Create a human-readable short tag that identifies the experiment variant."""
        model = config['model']['name']
        dataset = config['data']['dataset']
        eval_type = config['data']['evaluation_type']
        neg_strat = config['data']['negative_sampling_strategy']
        seed = config['experiment']['seed']

        # Collect architecture flags that matter for ablations
        flags = []
        if not config['model'].get('use_explicit_co_gnn', False):
            flags.append('no-co-gnn')
        if not config['model'].get('enable_walk', False):
            flags.append('no-walk')
        if not config['model'].get('enable_neighbor_cooc', False):
            flags.append('no-cooc')
        # Add learning rate if it's not the default (optional)
        lr = config['training'].get('learning_rate', 1e-4)
        if lr != 1e-4:
            flags.append(f'lr{lr}')
        # ... add any other discriminative parameters

        tag_parts = [model, dataset, eval_type, neg_strat] + flags + [f'seed{seed}']
        return '_'.join(tag_parts)




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
        self.start_time = datetime.now()
        seed = self.config['experiment']['seed']
        set_seed(seed)

        # ── Create unique run directory ────────────────────────
        run_tag = _make_run_tag(self.config)
        run_id = f"{run_tag}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path("experiment_framework/outputs") / run_id
        
        
        # run_id = f"{self.config['experiment']['name']}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        # run_dir = Path("experiment_framework/outputs") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run directory: {run_dir}")

        # 1. Data pipeline
        pipeline = self.create_data_pipeline()
        logger.info(f"Data pipeline ready: {getattr(pipeline, 'num_nodes', '?')} nodes")

        # 2. Model creation
        features = pipeline.get_features()
        model = ModelFactory.create(self.config, features)

        # 3. Model-specific setup + inject training hyperparams
        self.setup_model(model, pipeline)
        tcfg = self.config['training']
        model.learning_rate = tcfg['learning_rate']
        model.weight_decay = tcfg.get('weight_decay', 0.0)

        if self.config['experiment'].get('compile', False):
            if torch.cuda.is_available() and hasattr(torch, 'compile'):
                logger.info("Enabling torch.compile ...")
                model = torch.compile(model, mode="reduce-overhead")

        self._log_model_status(model)

        # 4. Optional profiling
        if self.config['experiment'].get('debug', False):
            self._profile_model(model, pipeline)

        # 5. Analysis collector
        self.analysis_collector = AnalysisCollector()

        if self.config['training'].get('dev_fast_run', False):
            logger.info("🏃 Fast dev run enabled - single batch train/val/test")
        
        


        
        # 6. Trainer – now writes to the run directory        

        if self.config['training'].get('dev_fast_run', False):
            trainer = TrainerSetup.create(
                self.config,
                callbacks=[self.analysis_collector, ClearCacheCallback()],
                default_root_dir=str(run_dir),
                fast_dev_run=True,
            )
        else:
            trainer = TrainerSetup.create(
                self.config,
                callbacks=[self.analysis_collector, ClearCacheCallback()],
                default_root_dir=str(run_dir),
            )



        # 7. Training
        trainer.fit(model, train_dataloaders=pipeline.loaders['train'],
                    val_dataloaders=pipeline.loaders['val'])

        best_ckpt_metric = None
        if not self.config['training'].get('dev_fast_run', False):
            ckpt_cb = trainer.checkpoint_callback
            if ckpt_cb and ckpt_cb.best_model_score is not None:
                best_ckpt_metric = ckpt_cb.best_model_score.item()
        
        
        # 8. Testing
        test_results = trainer.test(model, dataloaders=pipeline.loaders['test'],
                                    ckpt_path='best' if trainer.checkpoint_callback.best_model_path else None)

        # 8. Collect artifacts
        additional_artifacts = self._collect_additional_artifacts(model)
        
        # 10. Log results – log_dir points inside the run directory
        exp_logger = ExperimentLogger(str(run_dir / "metrics"))
        exp_logger.log(
            config=self.config,
            best_val_metric=best_ckpt_metric,
            monitor_name=self.config['training'].get('monitor', 'val_ap'),
            test_results=test_results,
            training_time=(datetime.now() - self.start_time).total_seconds(),
            model=model,
            walk_stats=self.analysis_collector.walk_stats,
            memory_trace=self.analysis_collector.memory_trace,
            ode_trajectory=self.analysis_collector.ode_trajectory,
            negative_stats=self.analysis_collector.negative_stats,
            cooccurrence_matrix=additional_artifacts.get('cooccurrence_matrix'),
        )

        # 10. Save final checkpoint – directly into run_dir
        final_path = run_dir / "final_model.ckpt"
        trainer.save_checkpoint(str(final_path))
        logger.info(f"Final model saved to {final_path}")

        logger.info(f"Training complete in {(datetime.now() - self.start_time).total_seconds():.1f}s")
        return {'test_results': test_results,
                'training_time': (datetime.now() - self.start_time).total_seconds(),
                'model': model}