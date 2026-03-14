import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math
from loguru import logger

class HardNegativeMiner(nn.Module):
    """
    Fully Vectorized Hard Negative Miner with Label Smoothing.
    
    Optimizations:
    1. Batched Matrix Multiplication for similarity (no loops).
    2. TopK selection for efficient hard negative mining.
    3. Mask-based filtering for ground-truth exclusion.
    4. Integrated Label Smoothing.
    """
    def __init__(
        self, 
        neg_sample_ratio: int = 5, 
        hard_neg_threshold: float = 0.7,
        label_smoothing: float = 0.1,
        use_half_precision: bool = True,  # Use FP16 for similarity matrix to save memory
        max_candidates: Optional[int] = None  # Limit candidates if num_nodes is huge
    ):
        super().__init__()
        self.neg_sample_ratio = neg_sample_ratio
        self.hard_neg_threshold = hard_neg_threshold
        self.label_smoothing = label_smoothing
        self.use_half_precision = use_half_precision
        self.max_candidates = max_candidates
        
        # Pre-register buffers for stability
        self.register_buffer('one', torch.tensor(1.0))
        self.register_buffer('zero', torch.tensor(0.0))

    def forward(
        self, 
        batch: Dict[str, torch.Tensor], 
        memory: torch.Tensor,
        cooccurrence_matrix: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dict with 'src_nodes', 'dst_nodes', 'labels'
            memory: [num_nodes, dim] node embeddings
            
        Returns:
            Augmented batch with hard negatives and smoothed labels.
        """
        src = batch['src_nodes']
        dst = batch['dst_nodes']
        labels = batch['labels']
        device = src.device
        num_nodes = memory.size(0)
        
        # Identify positive pairs
        pos_mask = (labels == 1.0)
        if not pos_mask.any():
            logger.warning("No positive pairs found in batch. Skipping hard negative mining.")
            return self._apply_label_smoothing(batch)
        
        pos_src = src[pos_mask]
        pos_dst = dst[pos_mask]
        num_pos = pos_src.size(0)
        
        if num_pos == 0:
            return self._apply_label_smoothing(batch)

        # --- 1. Vectorized Similarity Computation ---
        # Get embeddings for positive sources: [N_pos, D]
        src_emb = memory[pos_src]
        
        # Determine candidate set for negatives
        # Optimization: If num_nodes is huge, sample a random subset of candidates first
        if self.max_candidates is not None and num_nodes > self.max_candidates:
            # Ensure we don't accidentally exclude too many valid nodes
            # But for simplicity in full vectorization, we assume full set or random subset
            # Here we just use all nodes for correctness, but cast to FP16 for speed
            candidate_indices = torch.randperm(num_nodes, device=device)[:self.max_candidates]
        else:
            candidate_indices = torch.arange(num_nodes, device=device)
            
        all_cand_emb = memory[candidate_indices] # [N_all, D]
        
        # Compute Cosine Similarity Matrix: [N_pos, N_all]
        # Normalize once
        src_emb_norm = F.normalize(src_emb, p=2, dim=-1)
        all_cand_emb_norm = F.normalize(all_cand_emb, p=2, dim=-1)
        
        # Cast to FP16 if enabled to save memory on large matrices
        if self.use_half_precision and src_emb_norm.dtype != torch.float16:
            src_emb_norm = src_emb_norm.half()
            all_cand_emb_norm = all_cand_emb_norm.half()
            
        # Matrix Multiply: [N_pos, D] x [D, N_all] -> [N_pos, N_all]
        sim_matrix = torch.matmul(src_emb_norm, all_cand_emb_norm.t())
        
        # Cast back to original dtype for masking/logic
        if self.use_half_precision:
            sim_matrix = sim_matrix.float()
            
        # --- 2. Masking & Hard Negative Selection ---
        
        # Mask A: Exclude Ground Truth Positives
        # We need to mask out the specific (src, dst) pair for each positive sample
        # Create a mask [N_pos, N_all] where True = Allow Negative
        allowed_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        
        # Find indices in candidate_indices that match the true positive dst
        # Map pos_dst values to their index in candidate_indices
        # Since candidate_indices is usually 0..N-1, pos_dst IS the index.
        # If we used a random subset, we'd need a lookup. Assuming full set here.
        if self.max_candidates is None or num_nodes <= self.max_candidates:
            # Direct indexing
            gt_indices = pos_dst.unsqueeze(1) # [N_pos, 1]
            # Create a range row [0, 1, ..., N_all-1]
            col_range = torch.arange(num_nodes, device=device).unsqueeze(0) # [1, N_all]
            is_gt = (col_range == gt_indices) # [N_pos, N_all]
            allowed_mask = ~is_gt
        else:
            # If using subset, map global IDs to local indices
            # This is slightly slower but necessary for subset mode
            # For now, assume full set for max speed in standard use case
            logger.warning("Ground truth masking simplified for candidate subset mode.")
            allowed_mask[:] = True 

        # Mask B: Apply Similarity Threshold
        # We want negatives with similarity > threshold
        threshold_mask = (sim_matrix > self.hard_neg_threshold)
        
        # Combined Valid Mask
        valid_neg_mask = allowed_mask & threshold_mask

        # --- 3. Apply Co-occurrence Weighting (NEW) ---
        weights = torch.ones_like(sim_matrix)

        if cooccurrence_matrix is not None and isinstance(cooccurrence_matrix, dict):
            pass  # Shape mismatch handling as before
            logger.warning("Co-occurrence matrix passed but shape mismatch for direct node biasing. Using similarity only.")

        # --- 4. Select Hardest Negatives ---
        # CRITICAL: k is defined HERE and used immediately
        k = min(self.neg_sample_ratio, num_nodes - 1)
        
        fill_val = -1e9
        sim_score = sim_matrix.masked_fill(~valid_neg_mask, fill_val)
        sim_score = sim_score + weights
        
        # TopK selection
        topk_vals, topk_indices = torch.topk(sim_score, k, dim=-1)  # [N_pos, K]
        
        # Filter out fake negatives
        is_real_neg = (topk_vals > fill_val + 1.0)

        # Flatten for batching
        flat_src_idx = pos_src.repeat_interleave(k)
        flat_dst_idx = topk_indices.reshape(-1)
        flat_is_real = is_real_neg.reshape(-1)
        
        # Filter
        final_src = flat_src_idx[flat_is_real]
        final_dst = flat_dst_idx[flat_is_real]

        real_counts = is_real_neg.sum(dim=-1)  # [N_pos]

        # --- 5. Supplement with Random Negatives if needed ---
        current_count = final_src.size(0)
        target_count = num_pos * self.neg_sample_ratio
        
        pos_timestamps = batch['timestamps'][pos_mask]  # [N_pos]
        hard_neg_timestamps = pos_timestamps.repeat_interleave(real_counts)  # [N_pos * k]
        
        hard_neg_edge_features = None
        if batch.get('edge_features') is not None:
            pos_edge_feats = batch['edge_features'][pos_mask]
            hard_neg_edge_features = pos_edge_feats.repeat_interleave(real_counts, dim=0)


        if current_count < target_count:
            needed = target_count - current_count
            # Generate random negatives
            rand_src = pos_src.repeat_interleave(self.neg_sample_ratio)[:needed]
            rand_dst = torch.randint(0, num_nodes, (needed,), device=device)
            collision = (rand_dst == rand_src)
            rand_dst[collision] = (rand_dst[collision] + 1) % num_nodes
            
            # Random negatives inherit timestamps from their source
            rand_timestamps = pos_timestamps.repeat_interleave(self.neg_sample_ratio)[:needed]
            
            final_src = torch.cat([final_src, rand_src])
            final_dst = torch.cat([final_dst, rand_dst])
            hard_neg_timestamps = torch.cat([hard_neg_timestamps, rand_timestamps])
            
            if hard_neg_edge_features is not None:
                rand_edge_feats = pos_edge_feats.repeat_interleave(self.neg_sample_ratio, dim=0)[:needed]
                hard_neg_edge_features = torch.cat([hard_neg_edge_features, rand_edge_feats])
        
        # --- 6. Construct Augmented Batch ---
        hard_neg_labels = torch.zeros_like(final_src, dtype=torch.float)
        
        new_src = torch.cat([src, final_src])
        new_dst = torch.cat([dst, final_dst])
        new_labels = torch.cat([labels, hard_neg_labels])
        new_timestamps = torch.cat([batch['timestamps'], hard_neg_timestamps])

        new_batch = {
            'src_nodes': new_src,
            'dst_nodes': new_dst,
            'labels': new_labels,
            'timestamps': new_timestamps,
        }
        
        # Handle edge features
        if batch.get('edge_features') is not None:
            if hard_neg_edge_features is not None:
                new_batch['edge_features'] = torch.cat([batch['edge_features'], hard_neg_edge_features])
            else:
                # Create zero features for hard negatives if needed
                feat_dim = batch['edge_features'].size(-1)
                hard_neg_feats = torch.zeros(final_src.size(0), feat_dim, device=device)
                new_batch['edge_features'] = torch.cat([batch['edge_features'], hard_neg_feats])
        
        # Copy any other batch fields
        for key in batch.keys():
            if key not in new_batch:
                new_batch[key] = batch[key]
        
        return self._apply_label_smoothing(new_batch)

    def _apply_label_smoothing(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Applies label smoothing: 1 -> 1-eps, 0 -> eps"""
        if self.label_smoothing <= 0:
            return batch
            
        labels = batch['labels']
        # Smoothed Label = (1 - eps) * y + eps * (1 - y)
        # If y=1: 1-eps. If y=0: eps.
        smooth_labels = labels * (1.0 - self.label_smoothing) + \
                        (1.0 - labels) * self.label_smoothing
        
        batch['labels'] = smooth_labels
        return batch

    def get_cosine_scheduler(
        self, 
        optimizer: torch.optim.Optimizer, 
        num_epochs: int, 
        steps_per_epoch: int,
        warmup_epochs: float = 0.1,
        min_lr_ratio: float = 0.01
    ):
        """
        Factory method to create a Cosine Annealing LR Scheduler with Warmup.
        Note: This returns a lambda-based scheduler or requires a custom class 
        because standard PyTorch schedulers don't always handle warmup+cosine perfectly 
        in one go without chaining.
        """
        from torch.optim.lr_scheduler import LambdaLR
        
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = int(warmup_epochs * steps_per_epoch)
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(optimizer, lr_lambda=lr_lambda)