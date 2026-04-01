# src/experiments/sensitivity_analyzer.py
import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional
from copy import deepcopy
from loguru import logger
import traceback

from src.datasets.continue_temporal.data_con_pipeline import DataPipeline
from src.experiments.exp_utils.model_factory import ModelFactory
from src.experiments.exp_utils.trainer_setup import TrainerSetup
from src.experiments.exp_utils.analysis_callback import AnalysisCollector


class SensitivityResult:
    def __init__(self, param_name: str, param_value: Any, metrics: Dict, training_time: float, config_snapshot: Dict):
        self.param_name = param_name
        self.param_value = param_value
        self.test_ap = metrics.get('test_ap', 0.0)
        self.test_auc = metrics.get('test_auc', 0.0)
        self.test_accuracy = metrics.get('test_accuracy', 0.0)
        self.training_time = training_time
        self.config_snapshot = config_snapshot


class SensitivityAnalyzer:
    def __init__(self, base_config: Dict, output_dir: str = "results/sensitivity"):
        self.base_config = deepcopy(base_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []
        
         # Define the master summary path immediately
        self.summary_path = self.output_dir / "sensitivity_summary.csv"
        
        # Load existing results if the file already exists (prevents data loss on restart)
        if self.summary_path.exists():
            try:
                existing_df = pd.read_csv(self.summary_path)
                self.results = existing_df.to_dict('records')
                logger.info(f"Loaded {len(self.results)} existing results from {self.summary_path}")
            except Exception as e:
                logger.warning(f"Could not load existing summary: {e}")
        
        
        self._validate_and_fix_config()
        
        self._raw_data_cache = None


    def _validate_and_fix_config(self):
        """Ensure config has all required sections and keys with defaults."""
        
        defaults = {
            'model': {'name': 'HiCoSTv3', 'hidden_dim': 256, 'memory_dim': 256},
            'training': {'batch_size': 256, 'learning_rate': 1e-4, 'max_epochs': 50},
            'data': {'dataset': 'wikipedia'},
            'experiment': {
                'seed': 42, 
                'device': 'auto', 
                'precision': 32,
                'name': 'sensitivity_study' 
            },
            'logging': {
                'log_dir': 'logs',
                'checkpoint_dir': 'checkpoints',
                'monitor': 'val_ap',
                'mode': 'max',
                'save_top_k': 1,
            },
            'hardware': {'gpus': 1 if torch.cuda.is_available() else 0, 'num_workers': 4}
        }
        
        for section, section_defaults in defaults.items():
            if section not in self.base_config:
                self.base_config[section] = {}
            
            for key, value in section_defaults.items():
                if key not in self.base_config[section]:
                    self.base_config[section][key] = value
        
        logger.info(f"Config validated: model={self.base_config['model']['name']}, "
                   f"experiment={self.base_config['experiment']['name']}")

    def _ensure_config_structure(self, config: Dict):
        """Ensure all required sections exist in a config copy."""
        required = {
            'model': {},
            'training': {},
            'data': {},
            'experiment': {'name': 'sensitivity_run', 'seed': 42},
            'logging': {},
            'hardware': {}
        }
        for section, defaults in required.items():
            if section not in config:
                config[section] = {}
            for key, val in defaults.items():
                if key not in config[section]:
                    config[section][key] = val

    def run_study(self, param_path: str, values: List[Any], seeds: List[int] = [42], 
                  fixed_overrides: Optional[Dict] = None, condition: Optional[str] = None,
                  coupled_params: Optional[Dict] = None, lr_scaling: Optional[str] = None):
        """Run sensitivity study for a single parameter."""
        logger.info(f"Starting Study: {param_path} over {values}")
        
        # Track initial count to know what we added in this run
        initial_count = len(self.results)
        
        for val in values:
            if condition:
                temp_config = deepcopy(self.base_config)
                self._set_nested_param(temp_config, param_path, val)
                try:
                    if not eval(condition, {"__builtins__": {}}, {"config": temp_config}):
                        continue
                except:
                    continue

            for seed in seeds:
                try:
                    config = deepcopy(self.base_config)
                    
                    # Ensure structure before modifications
                    self._ensure_config_structure(config)
                    
                    # Set main parameter
                    self._set_nested_param(config, param_path, val)

                                        
                    # Apply overrides
                    if fixed_overrides and val in fixed_overrides:
                        for k, v in fixed_overrides[val].items():
                            path = f"model.{k}" if '.' not in k else k
                            self._set_nested_param(config, path, v)
                    
                    # Apply coupled params
                    if coupled_params:
                        for target, template in coupled_params.items():
                            if isinstance(template, str) and template.startswith("${"):
                                ref_val = self._get_nested_param(config, template[2:-1])
                                if ref_val is not None:
                                    self._set_nested_param(config, target, ref_val)
                    
                    # LR scaling
                    if lr_scaling and param_path == "training.batch_size":
                        base_lr = self.base_config['training']['learning_rate']
                        scale = np.sqrt(val / 256) if lr_scaling == 'sqrt' else val / 256
                        config['training']['learning_rate'] = base_lr * scale
                    
                    config['experiment']['seed'] = seed
                    
                    # Ensure experiment name exists
                    if 'name' not in config['experiment']:
                        config['experiment']['name'] = f"sensitivity_{param_path.replace('.', '_')}_{val}"
                    
                    # Run Experiment
                    start_time = time.time()
                    metrics = self._run_single_experiment(config, seed)
                    training_time = time.time() - start_time
                    
                    res = SensitivityResult(param_path, val, metrics, training_time, config['model'])
                    # self.results.append(res.__dict__)
                    
                    result_dict = res.__dict__
                    
                    result_dict['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    self.results.append(result_dict)
                    
                    self._save_master_summary()
                    
                    current_study_df = pd.DataFrame([r for r in self.results if r.get('param_name') == param_path])
                    study_csv = self.output_dir / f"{param_path.replace('.', '_')}_results.csv"
                    current_study_df.to_csv(study_csv, index=False)
                    logger.info(f"{param_path}={val}, seed={seed}: AP={metrics.get('test_ap', 0):.4f} | Saved to {study_csv}")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()                 
                    
            
                        
                except Exception as e:
                    # logger.error(f" Failed {param_path}={val}, seed={seed}: {e}")
                    # import traceback
                    # logger.error(traceback.format_exc())
                    # self.results.append({
                    #     'param_name': param_path, 'param_value': val, 'seed': seed,
                    #     'test_ap': np.nan, 'training_time': -1, 'error': str(e)
                    # })

                    error_dict = {
                        'param_name': param_path, 'param_value': val, 'seed': seed,
                        'test_ap': np.nan, 'test_auc': np.nan, 'test_accuracy': np.nan,
                        'training_time': -1, 'error': str(e), 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self.results.append(error_dict)
                    self._save_master_summary() # Save errors too so we know they happened
                    logger.error(f"Failed {param_path}={val}, seed={seed}: {e}")
        
        logger.info(f"Study complete. Total results saved: {len(self.results)}")
        return pd.DataFrame(self.results)

    def _save_master_summary(self):
        """Saves all accumulated results to a single CSV file."""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # Ensure consistent column order
        standard_cols = ['timestamp', 'param_name', 'param_value', 'seed', 'test_ap', 'test_auc', 'test_accuracy', 'training_time', 'error']
        available_cols = [c for c in standard_cols if c in df.columns]
        extra_cols = [c for c in df.columns if c not in standard_cols]
        
        df = df[available_cols + extra_cols]
        
        df.to_csv(self.summary_path, index=False)
    
    
    def _set_nested_param(self, config: Dict, path: str, value: Any):
        keys = path.split('.')
        curr = config
        for k in keys[:-1]:
            if k not in curr:
                curr[k] = {}
            curr = curr[k]
        curr[keys[-1]] = value

    def _get_nested_param(self, config: Dict, path: str):
        keys = path.split('.')
        curr = config
        for k in keys:
            if k not in curr:
                return None
            curr = curr[k]
        return curr

    def _get_raw_data(self, config: Dict):
        # if self._raw_data_cache is None:
        #     pipeline = (DataPipeline(self.base_config)
        #         .load()
        #         .build_neighbor_finder()
        #         .build_samplers()
        #         .build_datasets())
        #     self._raw_data_cache = pipeline
        # return self._raw_data_cache
        """
        Get raw data pipeline. 
        We do NOT cache the full pipeline because config (batch_size) changes per run.
        We only cache the heavy data loading part if possible, but for safety, we rebuild.
        """
        # Create a fresh pipeline for THIS specific config to ensure batch_size/lr are correct
        pipeline = (DataPipeline(config)
            .load()
            .build_neighbor_finder()
            .build_samplers()
            .build_datasets())
        
        # Do NOT cache the whole pipeline object as it holds state. 
        # Just return it. The deep copy in _run_single_experiment handles isolation.
        return pipeline

    def _run_single_experiment(self, config: Dict, seed: int) -> Dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        raw_pipeline = self._get_raw_data(config)
        
        pipeline = copy.deepcopy(raw_pipeline)

        # FORCE UPDATE batch size in pipeline if it doesn't re-read config
        if hasattr(pipeline, 'config'):            
            pipeline.config['training']['batch_size'] = config['training']['batch_size']

        pipeline.build_loaders()
        
        features = pipeline.get_features()

        # Inject unique checkpoint dir to avoid collisions
        unique_id = f"{config['experiment'].get('name', 'sens')}_{seed}"
        orig_checkpoint_dir = config['logging']['checkpoint_dir']
        config['logging']['checkpoint_dir'] = os.path.join(orig_checkpoint_dir, unique_id)


        model = ModelFactory.create(config, features)
        model.set_raw_features(features['node_features'], features['edge_features'])
        model.set_neighbor_finder(pipeline.neighbor_finder)
        
        if hasattr(pipeline.neighbor_finder, 'edge_index'):
            model.set_graph(pipeline.neighbor_finder.edge_index, pipeline.neighbor_finder.edge_time)
        
        analysis_cb = AnalysisCollector()
        trainer = TrainerSetup.create(config, callbacks=[analysis_cb])
        
        trainer.fit(model, pipeline.loaders['train'], pipeline.loaders['val'])
        
        # Restore checkpoint dir for next run (optional, since we use deepcopy of config)
        config['logging']['checkpoint_dir'] = orig_checkpoint_dir
        
        results = trainer.test(model, pipeline.loaders['test'], ckpt_path='best')
        
        return results[0] if results else {}

    def _save_and_plot(self, df: pd.DataFrame, param_name: str):
        if df.empty:
            return
        
        csv_path = self.output_dir / f"{param_name.replace('.', '_')}_results.csv"
        df.to_csv(csv_path, index=False)
        
        plot_df = df.dropna(subset=['test_ap'])
        if plot_df.empty:
            return
        
        try:
            plot_df['param_num'] = pd.to_numeric(plot_df['param_value'])
            agg = plot_df.groupby('param_num')['test_ap'].agg(['mean', 'std']).reset_index()
            x_col = 'param_num'
        except:
            agg = plot_df.groupby('param_value')['test_ap'].agg(['mean', 'std']).reset_index()
            x_col = 'param_value'
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(agg[x_col], agg['mean'], yerr=agg['std'], marker='o', capsize=5)
        plt.fill_between(agg[x_col], agg['mean'] - agg['std'], agg['mean'] + agg['std'], alpha=0.2)
        plt.title(f"Sensitivity: {param_name}")
        plt.xlabel(param_name)
        plt.ylabel("Test AP")
        plt.grid(True, alpha=0.3)
        
        plot_path = self.output_dir / f"{param_name.replace('.', '_')}_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot saved: {plot_path}")

    def run_study_from_spec(self, name: str, spec: Dict, seeds: List[int], config_filter: Optional[List[str]] = None):
        """Wrapper to handle special study types defined in YAML."""
        
        if 'configs' in spec:
            logger.info(f"Running explicit config study: {name}")
            for cfg_option in spec['configs']:
                run_name = cfg_option.get('name', f"{name}_{len(self.results)}")
                
                # Apply filter if specified
                if config_filter:
                    # Check if any filter keyword matches the config name (case-insensitive)
                    if not any(f.lower() in run_name.lower() for f in config_filter):
                        logger.debug(f"Skipping config '{run_name}' (does not match filter: {config_filter})")
                        continue
                    logger.info(f"Including config: {run_name}")
                
                self._run_custom_config_study(run_name, cfg_option, seeds)
            return

        if 'values' not in spec:
            logger.warning(f"Study {name} has no 'values' or 'configs'. Skipping.")
            return

        self.run_study(
            param_path=spec['parameter'],
            values=spec['values'],
            seeds=seeds,
            fixed_overrides=spec.get('fixed_overrides'),
            condition=spec.get('condition'),
            coupled_params=spec.get('coupled_params'),
            lr_scaling=spec.get('lr_scaling')
        )

    def _run_custom_config_study(self, name: str, config_overrides: Dict, seeds: List[int]):
        """Special runner for explicit config dictionaries."""
        for seed in seeds:
            try:
                config = deepcopy(self.base_config)
                
                for k, v in config_overrides.items():
                    if k == 'name':
                        continue
                    if k in ['num_walks_short', 'num_walks_long', 'num_walks_tawr']:
                        config['model'][k] = v
                    elif k in ['batch_size']:
                        config['training'][k] = v
                
                config['experiment']['seed'] = seed
                if 'name' not in config['experiment']:
                    config['experiment']['name'] = f"sensitivity_{name}"
                
                metrics = self._run_single_experiment(config, seed)
                res = SensitivityResult(name, str(config_overrides), metrics, 0, config.get('model', {}))
                result_dict = res.__dict__
                result_dict['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.results.append(result_dict)
                self._save_master_summary() 
                
            except Exception as e:
                logger.error(f"Failed {name} seed {seed}: {e}")
                logger.error(traceback.format_exc())
                error_dict = {
                    'param_name': name, 
                    'param_value': str(config_overrides),
                    'seed': seed,
                    'test_ap': np.nan, 
                    'test_auc': np.nan, 
                    'test_accuracy': np.nan,
                    'training_time': -1, 
                    'error': str(e),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self.results.append(error_dict)
                self._save_master_summary()
                
                
    
    def _validate_and_fix_config(self):
        """Ensure config has all required sections and keys."""
        if 'experiment' not in self.base_config:
            self.base_config['experiment'] = {}
        if 'name' not in self.base_config['experiment']:
            self.base_config['experiment']['name'] = 'sensitivity_study'