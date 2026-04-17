
from loguru import logger
from src.experiments.runner.tgn_runner import TGNRunner


class HiCoSTRunner(TGNRunner):
    """Runner for HiCoST variants (v1, v2, v3). Inherits pipeline setup from TGN."""

    def _log_model_status(self, model) -> None:
        super()._log_model_status(model)
        logger.info(f"=== HiCoST Specific Components ===")
        if hasattr(model, 'hparams'):
            logger.info(f"SAM prototypes enabled: {model.hparams.get('use_prototype_attention', True)}")
            logger.info(f"HCT hierarchical enabled: {model.hparams.get('use_hct_hierarchical', True)}")
            logger.info(f"MRP gating enabled: {model.hparams.get('use_gated_refinement', True)}")
            logger.info(f"Multi-scale walks enabled: {model.hparams.get('use_multi_scale_walks', True)}")
            logger.info(f"Spectral Temporal ODE: {model.hparams.get('use_st_ode', True)}")
            logger.info(f"Hard Negative Mining: {model.hparams.get('use_hard_negative_mining', True)}")
            logger.info(f"Walk sampler initialized: {model.walk_sampler is not None}")
        logger.info("=" * 40)

    def _collect_additional_artifacts(self, model) -> Dict[str, Any]:
        artifacts = {}
        if hasattr(model, 'get_cooccurrence'):
            try:
                artifacts['cooccurrence_matrix'] = model.get_cooccurrence()
            except Exception as e:
                logger.debug(f"Could not collect co-occurrence: {e}")
        return artifacts