# src/experiments/sensitivity_analyzer.py
import time
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from copy import deepcopy
from loguru import logger
import gc


from dataclasses import dataclass, asdict, field

from src.datasets.continue_temporal.data_con_pipeline import DataPipeline
from src.experiments.exp_utils.model_factory import ModelFactory
from src.experiments.exp_utils.trainer_setup import TrainerSetup
from src.experiments.exp_utils.analysis_callback import AnalysisCollector
from src.utils.config_utils import _clean_config_none_values, _ensure_dict

@dataclass
class SensitivityResult:    
    param_name: str
    param_value: Any
    metrics: Dict[str, Any] = field(default_factory=dict) 
    training_time: float = 0.0                              
    config_snapshot: Dict[str, Any] = field(default_factory=dict) 
    timestamp: Optional[str] = None
    seed: Optional[int] = None
    error: Optional[str] = None 

    def to_dict(self):
        d = asdict(self)
        if self.metrics:
            d.update(self.metrics)
        return d


class SensitivityAnalyzerV2:
    def __init__(self, base_config: Dict, output_dir: str = "results/sensitivity"):
        self.base_config = _clean_config_none_values(deepcopy(base_config))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []
        
         # master summary path 
        self.summary_path = self.output_dir / "sensitivity_summary.csv"
        
        # Load existing results if the file already exists 
        if self.summary_path.exists():
            try:
                existing_df = pd.read_csv(self.summary_path)
                self.results = existing_df.to_dict('records')
                logger.info(f"Loaded {len(self.results)} existing results from {self.summary_path}")
            except Exception as e:
                logger.warning(f"Could not load existing summary: {e}")
       

    
    def _save_master_summary(self):
        """Saves all accumulated results to a single CSV file."""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)      
        
        df.to_csv(self.summary_path, index=False)       
                
    def run_study(
            self, 
            param_path: str, 
            values: List[Any], 
            seeds: List[int] = [42],
            config_filter: Optional[List[str]] = None,
            coupled_params: Optional[Dict] = None
        ):
        """Run sensitivity study for a single parameter."""
        logger.info(f"Starting Study: {param_path} over {values}")
        
        for val in values:
            if config_filter and str(val) not in config_filter:
                logger.debug(f"Skipping {val} (not in filter: {config_filter})")
                continue
            
            for seed in seeds:
                config = deepcopy(self.base_config)
                self._set_nested_param(config, param_path, val)
                if coupled_params:
                    for target, template in coupled_params.items():
                        resolved_value = self._resolve_param_reference(config, template)
                        self._set_nested_param(config, target, resolved_value)
                
                
                
                # config['experiment']['seed'] = seed #None object problem
                exp_cfg = self._safe_ensure_dict(config, 'experiment', {
                    'name': f"sensitivity_{param_path.replace('.', '_')}",
                    'seed': seed
                })
                exp_cfg['seed'] = seed  # Ensure seed is set even if in defaults
                
                # === 5. Ensure logging section ===
                self._safe_ensure_dict(config, 'logging', {
                    'name': exp_cfg['name'],
                    'log_dir': 'logs'
                })
                
                try:
                    start_time = time.time()
                    metrics = self._run_experiment(config)
                    training_time = time.time() - start_time                    
                    result = SensitivityResult(
                        param_name=param_path,
                        param_value=val,
                        metrics=metrics,
                        training_time=training_time,
                        config_snapshot=config.get('model', {}),
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        seed=seed
                    )
                    self.results.append(result.to_dict())
                    # log with metrics value
                    ap = result.metrics.get('test_ap', float('nan'))
                    logger.info(f"{param_path}={val}, seed={seed}: AP={ap:.4f}")                  
                    
                        
                except Exception as e:
                    logger.error(f"Failed {param_path}={val}, seed={seed}: {e}")
                    self.results.append(SensitivityResult(
                        param_name=param_path,
                        param_value=val,
                        seed=seed,
                        training_time=-1,
                        error=str(e)
                    ).to_dict())

                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                self._save_master_summary()
        
        return pd.DataFrame(self.results)
    
    def _run_experiment(self, config: Dict) -> Dict:
        """Run single experiment and return metrics, with verbose error tracing."""
        try:
            logger.debug(f"[1/7] Setting seed: {config['experiment']['seed']}")
            torch.manual_seed(config['experiment']['seed'])
            
            logger.debug("[2/7] Building data pipeline")
            pipeline = (DataPipeline(config)
                    .load()
                    .build_neighbor_finder()
                    .build_samplers()
                    .build_datasets()
                    .build_loaders())
            
            logger.debug("[3/7] Getting features")
            features = pipeline.get_features()
            logger.debug(f"  features keys: {list(features.keys()) if features else 'None'}")
            if features:
                for k, v in features.items():
                    if v is None:
                        logger.warning(f"  ⚠️  features['{k}'] is None")
            
            logger.debug("[4/7] Creating model")
            model = ModelFactory.create(config, features)
            
            logger.debug("[5/7] Setting model inputs")
            node_feat = features.get('node_features')
            edge_feat = features.get('edge_features')
            logger.debug(f"  node_features: {type(node_feat)}, shape: {node_feat.shape if hasattr(node_feat, 'shape') else 'N/A'}")
            logger.debug(f"  edge_features: {type(edge_feat)}, shape: {edge_feat.shape if hasattr(edge_feat, 'shape') else 'N/A'}")
            model.set_raw_features(node_feat, edge_feat)
            model.set_neighbor_finder(pipeline.neighbor_finder)
            
            if hasattr(pipeline.neighbor_finder, 'edge_index'):
                logger.debug("[6/7] Setting graph")
                model.set_graph(pipeline.neighbor_finder.edge_index, pipeline.neighbor_finder.edge_time)
            
            logger.debug("[7/7] Training and evaluating")
            analysis_cb = AnalysisCollector()

            if torch.cuda.is_available():
                logger.info(f"GPU memory before training: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(0)/1024**3:.2f} GB reserved")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            trainer = TrainerSetup.create(config, callbacks=[analysis_cb])
            trainer.fit(model, pipeline.loaders['train'], pipeline.loaders['val'])

            if torch.cuda.is_available():
                model = model.to('cuda')
                logger.info(f"GPU memory after model load: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
                        
            results = trainer.test(model, pipeline.loaders['test'], ckpt_path='best')
            return results[0] if results else {}
            
        except Exception as e:
            import traceback
            logger.error(f"_run_experiment failed at step: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {
                'test_ap': float('nan'),
                'test_auc': float('nan'), 
                'test_accuracy': float('nan'),
                'error': str(e)
            }
        finally:
             if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    
    # def _run_experiment(self, config: Dict) -> Dict:
    #     """Run single experiment and return metrics."""
    #     try:        
    #         torch.manual_seed(config['experiment']['seed'])
            
    #         # Data setup
    #         pipeline = (DataPipeline(config)
    #                 .load()
    #                 .build_neighbor_finder()
    #                 .build_samplers()
    #                 .build_datasets()
    #                 .build_loaders())
            
    #         features = pipeline.get_features()
            
    #         # Model setup
    #         model = ModelFactory.create(config, features)
    #         model.set_raw_features(features['node_features'], features['edge_features'])
    #         model.set_neighbor_finder(pipeline.neighbor_finder)
            
    #         if hasattr(pipeline.neighbor_finder, 'edge_index'):
    #             model.set_graph(pipeline.neighbor_finder.edge_index, pipeline.neighbor_finder.edge_time)
            
    #         # Training
    #         analysis_cb = AnalysisCollector()
    #         trainer = TrainerSetup.create(config, callbacks=[analysis_cb])
    #         trainer.fit(model, pipeline.loaders['train'], pipeline.loaders['val'])
            
    #         results = trainer.test(model, pipeline.loaders['test'], ckpt_path='best')
    #         return results[0] if results else {}
        
    #     except Exception as e:
    #         # Return empty metrics dict instead of raising
    #         logger.error(f"_run_experiment failed: {e}")
    #         return {
    #             'test_ap': float('nan'),
    #             'test_auc': float('nan'), 
    #             'test_accuracy': float('nan'),
    #             'error': str(e)
    #         }            
    
    
    def _set_param(self, config: Dict, path: str, value: Any):
        """Set nested dict parameter by dot-path."""
        keys = path.split('.')
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value

    

    def _resolve_param_reference(self, config: Dict, ref: str) -> Any:
        """Resolve ${path.to.param} references to actual config values."""
        if not isinstance(ref, str) or not ref.startswith("${") or not ref.endswith("}"):
            return ref
        
        path = ref[2:-1]  # Remove ${ and }
        value = self._get_nested_param(config, path)
        if value is None:
            logger.warning(f"Could not resolve reference: {ref}")
            return ref  # Return literal string if not found
        return value

    def _get_nested_param(self, config: Dict, path: str) -> Optional[Any]:
        """Get nested dict parameter by dot-path."""
        keys = path.split('.')
        curr = config
        for k in keys:
            if not isinstance(curr, dict) or k not in curr:
                return None
            curr = curr[k]
        return curr
    
    def _set_nested_param(self, config: Dict, path: str, value: Any) -> None:
        """Set nested dict parameter, resolving ${...} references in value."""
        # Resolve any references in the value first
        if isinstance(value, str) and value.startswith("${"):
            value = self._resolve_param_reference(config, value)
        
        keys = path.split('.')
        curr = config
        for k in keys[:-1]:
            if k not in curr or curr[k] is None:
                curr[k] = {}
            curr = curr[k]
            if not isinstance(curr, dict):
                raise ValueError(f"Cannot set '{path}': '{k}' is {type(curr).__name__}, not dict")
        curr[keys[-1]] = value

    def _safe_ensure_dict(self, config: Dict, key: str, defaults: Optional[Dict] = None) -> Dict:
        """Ensure config[key] is a dict, with optional default values."""
        if key not in config or config[key] is None:
            config[key] = {}
        if not isinstance(config[key], dict):
            raise ValueError(f"config['{key}'] should be dict, got {type(config[key]).__name__}")
        
        if defaults:
            for k, v in defaults.items():
                if k not in config[key]:
                    config[key][k] = v
        return config[key]
    
