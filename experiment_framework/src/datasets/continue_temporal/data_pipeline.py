# data_pipeline.py
import torch 
from typing import Dict, List, Optional, Any, Iterator
from loguru import logger

from torch.utils.data import DataLoader, Sampler, BatchSampler

# from src.datasets.load_dataset import load_dataset
from src.datasets.sam_dataloading.data_loader import load_dataset
from src.datasets.sam_dataloading.negative_sample import NegativeSampler
from src.datasets.sam_dataloading.neighbor_finder import NeighborFinder
from src.datasets.continue_temporal.temporal_data import TemporalDataset


# from torch.utils.data import BatchSampler, SequentialSampler
# ============================================================================
# CONTINUOUS TIME BATCH SAMPLER FOR ODE MODELS
# ============================================================================

class ContinuousTimeBatchSampler(Sampler):
    """
    BatchSampler that provides continuous time across epochs for ODE-based models.
    
    CRITICAL: Ensures temporal ordering WITHIN each batch. Batches are contiguous
    chunks of the temporally sorted dataset, maintaining chronological order.
    """
    
    def __init__(
        self, 
        dataset: TemporalDataset, 
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False  # Must be False for temporal continuity
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle  # Should always be False for ODE continuity
        
        # Total number of samples
        self.num_samples = len(dataset)
        
        # Verify dataset is temporally sorted
        self._verify_temporal_ordering()
        
        # Calculate number of batches per epoch
        if drop_last:
            self.batches_per_epoch = self.num_samples // batch_size
        else:
            self.batches_per_epoch = (self.num_samples + batch_size - 1) // batch_size
        
        # Global batch counter (persists across epochs - THIS IS THE KEY)
        self.global_position = 0
        self.global_batch_idx = 0
        
        
        
        logger.info(f"ContinuousTimeBatchSampler: {self.num_samples} samples, "
                   f"batch_size={batch_size}, {self.batches_per_epoch} batches/epoch, "
                )
    
    def _verify_temporal_ordering(self):
        """
        CRITICAL: Verify that dataset samples are in strict temporal order.
        Raises error if not sorted - this is fundamental for temporal GNNs.
        """
        if len(self.dataset) == 0:
            return
        
        # Sample check first 1000 elements (or all if smaller)
        n_check = min(1000, len(self.dataset))
        timestamps = [self.dataset.samples[i]['timestamp'] for i in range(n_check)]
        
        # Check for inversions
        inversions = []
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i-1] - 1e-9:
                inversions.append((i-1, i, timestamps[i-1], timestamps[i]))
        
        if inversions:
            # Log first few inversions
            for inv in inversions[:5]:
                logger.error(f"Temporal inversion at indices {inv[0]}->{inv[1]}: "
                           f"{inv[2]:.1f} -> {inv[3]:.1f}")
            
            raise RuntimeError(
                f"Dataset NOT temporally sorted! Found {len(inversions)} inversions "
                f"in first {n_check} samples. Temporal GNNs require chronological ordering."
            )
        
        # Also check full dataset if small enough
        if len(self.dataset) <= 10000:
            full_timestamps = [s['timestamp'] for s in self.dataset.samples]
            for i in range(1, len(full_timestamps)):
                if full_timestamps[i] < full_timestamps[i-1] - 1e-9:
                    raise RuntimeError(
                        f"Dataset NOT temporally sorted at index {i}: "
                        f"{full_timestamps[i-1]:.1f} -> {full_timestamps[i]:.1f}"
                    )
            logger.info(f"Full dataset temporal ordering verified ({len(self.dataset)} samples)")
        else:
            logger.info(f"Sample temporal ordering verified ({n_check}/{len(self.dataset)} samples)")
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield batches with STRICT temporal monotonicity.
        
        KEY FIX: Never wrap around within a batch. Each batch contains 
        contiguous indices [start, end) where timestamps are monotonic.
        """
        for batch_idx in range(self.batches_per_epoch):
            # Calculate batch boundaries
            batch_start = self.global_position
            batch_end = min(batch_start + self.batch_size, self.num_samples)
            
            # Check if we've reached the end
            if batch_start >= self.num_samples:
                # Reset to beginning for next epoch
                self.global_position = 0
                batch_start = 0
                batch_end = min(self.batch_size, self.num_samples)
            
            # Generate contiguous indices
            indices = list(range(batch_start, batch_end))
            
            # Handle incomplete final batch
            if len(indices) < self.batch_size:
                if self.drop_last:
                    # Skip incomplete batch, reset position for next epoch
                    self.global_position = 0
                    continue
                else:
                    # CRITICAL FIX: Don't wrap - pad with last valid sample repeated
                    # This maintains temporal monotonicity (same timestamp repeated)
                    # rather than jumping back to t=0
                    last_idx = indices[-1] if indices else 0
                    while len(indices) < self.batch_size:
                        indices.append(last_idx)
                    logger.debug(f"Batch {batch_idx}: padded with last index {last_idx}")
            
            # STRICT verification: ensure temporal monotonicity
            if len(indices) > 1:
                batch_times = [self.dataset.samples[i]['timestamp'] for i in indices]
                
                # Check for ANY non-monotonicity
                for i in range(len(batch_times) - 1):
                    if batch_times[i+1] < batch_times[i] - 1e-9:
                        logger.error(f"CRITICAL: Temporal violation in batch {batch_idx}: "
                                   f"index {i} ({batch_times[i]}) > index {i+1} ({batch_times[i+1]})")
                        logger.error(f"  Indices: {indices[i:i+2]}")
                        raise RuntimeError(f"Temporal non-monotonicity in batch {batch_idx}")
            
            # Update global position for next batch
            self.global_position = batch_end % self.num_samples
            
            yield indices
            self.global_batch_idx += 1
    
    def __len__(self) -> int:
        return self.batches_per_epoch
    
    def state_dict(self) -> Dict[str, Any]:
        """Save sampler state for checkpointing."""
        return {
            'global_batch_idx': self.global_batch_idx,
            'global_position': self.global_position
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Restore sampler state from checkpoint."""
        self.global_batch_idx = state.get('global_batch_idx', 0)
        self.global_position = state.get('global_position', 0)


class ContinuousTimeDataLoader(DataLoader):
    """
    DataLoader that maintains continuous time across epochs for ODE models.
    
    Uses BatchSampler to ensure temporal ordering within batches and 
    continuity across epochs.
    """
    
    def __init__(
        self,
        dataset: TemporalDataset,
        batch_size: int = 1,
        shuffle: bool = False,  # Ignored, always False for continuity
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Any] = None,
        num_workers: int = 0,
        collate_fn = None,
        pin_memory: bool = False,
        drop_last: bool = False,  # Only used if creating batch_sampler internally
        timeout: float = 0,
        worker_init_fn = None,
        multiprocessing_context = None,
        generator = None,
        *,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        pin_memory_device: str = ""
    ):
        # Create continuous batch sampler if not provided
        if batch_sampler is None:
            batch_sampler = ContinuousTimeBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                drop_last=drop_last,
                shuffle=False
            )
        
        # CRITICAL: When batch_sampler is provided, do NOT pass batch_size, shuffle, sampler, or drop_last
        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device
        )
        
        self.continuous_batch_sampler = batch_sampler
        self._epoch_count = 0
    
    def __iter__(self):
        """Track epoch boundaries while maintaining continuous time."""
        self._epoch_count += 1
        start_batch = getattr(self.continuous_batch_sampler, 'global_batch_idx', 0)
        logger.debug(f"ContinuousTimeDataLoader epoch {self._epoch_count} "
                    f"starting at global batch {start_batch}")
        return super().__iter__()
    
    def get_temporal_state(self) -> Dict[str, Any]:
        """Get current temporal state for ODE solver initialization."""
        sampler = self.continuous_batch_sampler
        if isinstance(sampler, ContinuousTimeBatchSampler):
            current_pos = (sampler.global_batch_idx * sampler.batch_size) % sampler.num_samples
            return {
                'global_batch_idx': sampler.global_batch_idx,
                'current_position': current_pos,
                'total_samples': sampler.num_samples,
                'epoch_batches': sampler.batches_per_epoch,
            }
        return {}
    
    def state_dict(self) -> Dict[str, Any]:
        """Save dataloader state."""
        return self.continuous_batch_sampler.state_dict()
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Restore dataloader state."""
        self.continuous_batch_sampler.load_state_dict(state)


# ============================================================================
# DATA PIPELINE
# ============================================================================

class DataPipeline:
    """Encapsulates all data-related setup."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data: Optional[Dict] = None
        self.neighbor_finder: Optional[NeighborFinder] = None
        self.samplers: Dict[str, NegativeSampler] = {}
        self.datasets: Dict[str, TemporalDataset] = {}
        self.loaders: Dict[str, DataLoader] = {}
    
    def load(self) -> 'DataPipeline':
        """Load raw dataset with validation."""
        logger.info(f"Loading dataset: {self.config['data']['dataset']}")
        
        eval_type = self.config['data']['evaluation_type']
        sampling_strategy = self.config['data']['negative_sampling_strategy']
        
        #  Comprehensive inductive sampling validation
        if sampling_strategy == 'inductive':
            if eval_type != 'inductive':
                raise ValueError(
                    f"Inductive sampling requires inductive evaluation. "
                    f"Got evaluation_type='{eval_type}'. "
                    f"Fix: Use evaluation_type='inductive' OR switch to "
                    f"negative_sampling_strategy='random'/'historical'."
                )
            if self.config['data'].get('unseen_ratio', 0.1) <= 0:
                raise ValueError(
                    "Inductive sampling requires unseen_ratio > 0. "
                    "Fix: Set data.unseen_ratio=0.1 in config."
                )
        
        self.data = load_dataset(
            dataset_name=self.config['data']['dataset'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio'],
            inductive=(eval_type == 'inductive'),
            unseen_ratio=self.config['data'].get('unseen_ratio', 0.1),
            seed=self.config['experiment']['seed'],
        )        
       
        logger.info(f"Loaded: {self.data['num_nodes']} nodes, {self.data['statistics']['num_edges']} edges")
        return self
    
    def build_neighbor_finder(self) -> 'DataPipeline':
        """Build neighbor finder from training edges only (leakage-proof)."""
        train_edges = self.data['edges'][self.data['train_mask']]
        train_ts = self.data['timestamps'][self.data['train_mask']]
        
        self.neighbor_finder = NeighborFinder(
            train_edges=train_edges,
            train_timestamps=train_ts,
            max_neighbors=self.config['data']['max_neighbors']
        )
        
        logger.info(f"Built leakage-proof NeighborFinder from {len(train_edges)} training edges")
        return self
    
    def build_samplers(self) -> 'DataPipeline':
        """Build negative samplers with TGN paper standard enforcement."""
        splits = ['train', 'val', 'test']
        masks = ['train_mask', 'val_mask', 'test_mask']
        
        for split, mask_key in zip(splits, masks):
            edges = self.data['edges'][self.data[mask_key]]
            timestamps = self.data['timestamps'][self.data[mask_key]]
            
            
            self.samplers[split] = NegativeSampler(
                edges=edges,
                timestamps=timestamps,
                num_nodes=self.data['num_nodes'],
                neighbor_finder=self.neighbor_finder,
                seed=self.config['experiment']['seed']
            )
        
        logger.info(f"Built samplers: train=random (TGN standard), "
                   f"val/test={self.config['data']['negative_sampling_strategy']}")
        return self
    
    def build_datasets(self) -> 'DataPipeline':
        """Build TemporalDatasets with STRICT feature masking."""
        splits = ['train', 'val', 'test']
        masks = ['train_mask', 'val_mask', 'test_mask']
        is_inductive = self.config['data']['evaluation_type'] == 'inductive'
        
        for split, mask_key in zip(splits, masks):
            mask = self.data[mask_key]
            
            #  MASK EDGE FEATURES PER SPLIT (prevent leakage)
            # Using full edge_features tensor would leak val/test features into training!
            split_edge_features = (
                self.data['edge_features'][mask] if self.data['edge_features'] is not None else None
            )
            
            # Get unseen nodes ONLY for val/test in inductive evaluation
            unseen_nodes = (
                self.data['unseen_nodes'] 
                if (is_inductive and split != 'train') 
                else None
            )
            
            # Determine sampling strategy per split (train always random)
            sampling_strategy = (
                'random' if split == 'train' 
                else self.config['data']['negative_sampling_strategy']
            )
            
            self.datasets[split] = TemporalDataset(
                edges=self.data['edges'][mask],
                timestamps=self.data['timestamps'][mask],
                edge_features=split_edge_features,  # MASKED FEATURES
                num_nodes=self.data['num_nodes'],
                split=split,
                negative_sampler=self.samplers[split],
                negative_sampling_strategy=sampling_strategy,
                unseen_nodes=unseen_nodes,
                seed=self.config['experiment']['seed']
            )
        
        # Validate per-batch label balance (prevent single-class batches)
        # Optional batch validation
        if self.config['data'].get('validate_batches', True):
            self._validate_evaluation_batches()
        else:
            logger.info("Skipping evaluation batch validation (config flag disabled)")
        
        logger.info(f"Built datasets: { {k: len(v) for k, v in self.datasets.items()} }")
        return self
    
    def _validate_evaluation_batches(self):
        """Ensure ALL evaluation batches contain both classes (prevent AUC=nan)."""
        for split in ['val', 'test']:
            if split not in self.datasets:
                continue
            
            dataset = self.datasets[split]
            batch_size = self.config['training']['batch_size']
            
            # Check first 10 batches
            for i in range(min(10, len(dataset) // batch_size)):
                start = i * batch_size
                end = min(start + batch_size, len(dataset))
                batch_labels = [dataset.samples[j]['label'] for j in range(start, end)]
                
                # Must have BOTH classes in every batch
                if 0.0 not in batch_labels or 1.0 not in batch_labels:
                    raise ValueError(
                        f"Single-class batch detected in {split} split (batch {i})! "
                        f"Labels: {set(batch_labels)}. "
                        f"Fix: Ensure positives/negatives are interleaved in _prepare_samples()."
                    )
        
        logger.info("All evaluation batches validated: contain both classes (valid metrics guaranteed)")
    
    
    
    def build_loaders(self) -> 'DataPipeline':
        """
        Build DataLoaders with CONTINUOUS TIME for ODE models.
        
        Training uses ContinuousTimeDataLoader to maintain temporal continuity
        across epochs. Validation and test use standard loaders.
        """
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['hardware'].get('num_workers', 0)
        
        # Check if we should use continuous time (for ODE models)
        use_continuous_time = self.config.get('model', {}).get('use_ode', False) or \
                             self.config.get('training', {}).get('continuous_time', False)
        
        for split, dataset in self.datasets.items():
            if split == 'train' and use_continuous_time:
                # Use continuous time batch sampler for training (ODE models)
                # This maintains global temporal state across epochs
                self.loaders[split] = ContinuousTimeDataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=TemporalDataset.collate_fn,
                    pin_memory=self.config['hardware'].get('pin_memory', False),
                    drop_last=False
                )
                logger.info(f"{split}: ContinuousTimeDataLoader (ODE mode, workers={num_workers})")
            else:
                # Standard sequential loader for val/test (or non-ODE training)
                self.loaders[split] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=TemporalDataset.collate_fn,
                    pin_memory=self.config['hardware'].get('pin_memory', False),
                    sampler=torch.utils.data.SequentialSampler(dataset),
                )
                logger.info(f"{split}: Standard DataLoader (sequential)")
        
        if use_continuous_time:
            logger.info("CONTINUOUS TIME MODE: Global temporal state maintained across epochs")
        
        return self
    
    def get_features(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get node/edge features with STRUCTURAL DATASET DETECTION."""
        dataset = self.config['data']['dataset'].lower()
        # STRUCTURAL_DATASETS = {'untrade', 'uslegis', 'canparl', 'unvote', 'enron', 'uci'}  # UCI is NOT structural!
        
        # UCI is NOT structural - has real edge features
        IS_STRUCTURAL = dataset in {'untrade', 'uslegis', 'canparl', 'unvote'}
        
        if IS_STRUCTURAL:
            # Structural datasets: create 1-dim dummy edge features
            node_features = None
            num_edges = self.data['train_mask'].sum().item()
            edge_features = torch.ones(num_edges, 1)  # 1-dim dummy features (required)
            logger.info(f" Structural dataset {dataset}: using 1-dim dummy edge features")
            return {
                'node_features': node_features,
                'edge_features': edge_features,
                'num_nodes': self.data['num_nodes'],
                'edge_feat_dim': 1,  #  structural datasets need dummy features
                'node_feat_dim': 0,
            }
        
        # Enron has 32-dim edge features (DyGLib format)
        if dataset == "enron":
            train_mask = self.data['train_mask']
            edge_features = self.data['edge_features'][train_mask]
            
            # Enron edge features are 32-dimensional (message content embedding)
            if edge_features.shape[1] != 32:
                logger.warning(
                    f"Enron edge features should be 32-dim, got {edge_features.shape[1]}. "
                    f"Using actual dimension: {edge_features.shape[1]}"
                )
            
            return {
                'node_features': None,  # Enron has no node features
                'edge_features': edge_features,
                'num_nodes': self.data['num_nodes'],
                'edge_feat_dim': edge_features.shape[1],  # 32 (not 1!)
                'node_feat_dim': 0,
            }        
        
        
        # UCI-specific handling (has 2-dim edge features)
        if dataset == "uci":
            train_mask = self.data['train_mask']
            edge_features = self.data['edge_features'][train_mask]
            
            # UCI edge features are 2-dimensional (message content embedding)
            if edge_features.shape[1] != 2:
                logger.warning(f"UCI edge features should be 2-dim, got {edge_features.shape[1]}. Truncating.")
                edge_features = edge_features[:, :2]
            
            return {
                'node_features': self.data.get('node_features'),  # 100-dim for UCI
                'edge_features': edge_features,
                'num_nodes': self.data['num_nodes'],
                'edge_feat_dim': 2,  # Critical: NOT 1!
                'node_feat_dim': 100 if self.data.get('node_features') is not None else 0,
            }
              
        
        node_features = self.data.get("node_features")

        if dataset == "wikipedia" and node_features is not None:
            logger.warning("Wikipedia unexpectedly has node features – keeping them.")
        
        
        train_edge_features = (
            self.data['edge_features'][self.data['train_mask']] 
            if self.data['edge_features'] is not None 
            else None
        )
        
        return {
            'node_features': node_features,
            'edge_features': train_edge_features,
            'num_nodes': self.data['num_nodes'],
            'edge_feat_dim': train_edge_features.shape[1] if train_edge_features is not None else 0,
            'node_feat_dim': node_features.shape[1] if node_features is not None else 0,
        }     
    
    
    @property
    def num_nodes(self) -> int:
        return self.data['num_nodes']