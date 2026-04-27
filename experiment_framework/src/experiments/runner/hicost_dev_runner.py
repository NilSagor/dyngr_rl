from .hicost_base_runner import HiCoSTBaseRunner
from loguru import logger
import torch

class HiCoSTDevRunner(HiCoSTBaseRunner):
    """Runner for all HiCoST dev variants (v1, v2, ...)."""

    def _log_model_status(self, model: torch.nn.Module) -> None:
        # logger.info(f"=== HiCoSTDev ({model.cfg.version}) Model Status ===")
        logger.info(f"=== HiCoSTDev ({self.config['model']['name']}) Model Status ===")
        logger.info(f"Use memory: {model.use_memory}")
        logger.info(f"Enable walk: {model.enable_walk}")
        logger.info(f"Enable restart: {model.enable_restart}")
        logger.info(f"Enable co-occurrence: {model.neighbor_cooc}")
        logger.info(f"Explicit Co-GNN enabled: {model.use_explicit_co_gnn}")
        if model.use_explicit_co_gnn and hasattr(model, 'co_gnn'):
            logger.info(f"Co-GNN: {model.co_gnn}")