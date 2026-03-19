# data_pipeline.py
import torch 
from typing import Dict, List, Optional, Any, Iterator
from loguru import logger

from torch.utils.data import DataLoader, Sampler


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


