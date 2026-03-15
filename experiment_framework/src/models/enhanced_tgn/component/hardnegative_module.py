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
        use_half_precision: bool = True,
        max_candidates: Optional[int] = None
    ):
        super().__init__()
        self.neg_sample_ratio = neg_sample_ratio
        self.hard_neg_threshold = hard_neg_threshold
        self.label_smoothing = label_smoothing
        self.use_half_precision = use_half_precision
        self.max_candidates = max_candidates
        
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
            batch: Dict with 'src_nodes', 'dst_nodes', 'labels', 'timestamps'
            memory: [num_nodes, dim] node embeddings
            cooccurrence_matrix: Not used in this version (kept for API compatibility)
        """
        src = batch['src_nodes']
        dst = batch['dst_nodes']
        labels = batch['labels']
        device = src.device
        num_nodes = memory.size(0)
        
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
        src_emb = memory[pos_src]
        
        if self.max_candidates is not None and num_nodes > self.max_candidates:
            candidate_indices = torch.randperm(num_nodes, device=device)[:self.max_candidates]
        else:
            candidate_indices = torch.arange(num_nodes, device=device)
            
        all_cand_emb = memory[candidate_indices]
        
        src_emb_norm = F.normalize(src_emb, p=2, dim=-1)
        all_cand_emb_norm = F.normalize(all_cand_emb, p=2, dim=-1)
        
        if self.use_half_precision and src_emb_norm.dtype != torch.float16:
            src_emb_norm = src_emb_norm.half()
            all_cand_emb_norm = all_cand_emb_norm.half()
            
        sim_matrix = torch.matmul(src_emb_norm, all_cand_emb_norm.t())
        
        if self.use_half_precision:
            sim_matrix = sim_matrix.float()
            
        # --- 2. Masking & Hard Negative Selection ---
        allowed_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        
        if self.max_candidates is None or num_nodes <= self.max_candidates:
            gt_indices = pos_dst.unsqueeze(1)
            col_range = torch.arange(num_nodes, device=device).unsqueeze(0)
            is_gt = (col_range == gt_indices)
            allowed_mask = ~is_gt
        else:
            # For subset, we need a mapping. This is simplified; we rely on random negatives later.
            pass

        threshold_mask = (sim_matrix > self.hard_neg_threshold)
        valid_neg_mask = allowed_mask & threshold_mask

        k = min(self.neg_sample_ratio, num_nodes - 1)
        
        fill_val = -1e9
        sim_score = sim_matrix.masked_fill(~valid_neg_mask, fill_val)
        
        topk_vals, topk_indices = torch.topk(sim_score, k, dim=-1)
        
        is_real_neg = (topk_vals > fill_val + 1.0)

        flat_src_idx = pos_src.repeat_interleave(k)
        flat_dst_idx = topk_indices.reshape(-1)
        flat_is_real = is_real_neg.reshape(-1)
        
        final_src = flat_src_idx[flat_is_real]
        final_dst = flat_dst_idx[flat_is_real]

        real_counts = is_real_neg.sum(dim=-1)

        current_count = final_src.size(0)
        target_count = num_pos * self.neg_sample_ratio
        
        pos_timestamps = batch['timestamps'][pos_mask]
        hard_neg_timestamps = pos_timestamps.repeat_interleave(real_counts)
        
        hard_neg_edge_features = None
        if batch.get('edge_features') is not None:
            pos_edge_feats = batch['edge_features'][pos_mask]
            hard_neg_edge_features = pos_edge_feats.repeat_interleave(real_counts, dim=0)

        if current_count < target_count:
            needed = target_count - current_count
            rand_src = pos_src.repeat_interleave(self.neg_sample_ratio)[:needed]
            rand_dst = torch.randint(0, num_nodes, (needed,), device=device)
            collision = (rand_dst == rand_src)
            rand_dst[collision] = (rand_dst[collision] + 1) % num_nodes
            
            rand_timestamps = pos_timestamps.repeat_interleave(self.neg_sample_ratio)[:needed]
            
            final_src = torch.cat([final_src, rand_src])
            final_dst = torch.cat([final_dst, rand_dst])
            hard_neg_timestamps = torch.cat([hard_neg_timestamps, rand_timestamps])
            
            if hard_neg_edge_features is not None:
                rand_edge_feats = pos_edge_feats.repeat_interleave(self.neg_sample_ratio, dim=0)[:needed]
                hard_neg_edge_features = torch.cat([hard_neg_edge_features, rand_edge_feats])
        
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
        
        if batch.get('edge_features') is not None:
            if hard_neg_edge_features is not None:
                new_batch['edge_features'] = torch.cat([batch['edge_features'], hard_neg_edge_features])
            else:
                feat_dim = batch['edge_features'].size(-1)
                hard_neg_feats = torch.zeros(final_src.size(0), feat_dim, device=device)
                new_batch['edge_features'] = torch.cat([batch['edge_features'], hard_neg_feats])
        
        for key in batch.keys():
            if key not in new_batch:
                new_batch[key] = batch[key]
        
        return self._apply_label_smoothing(new_batch)

    def _apply_label_smoothing(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.label_smoothing <= 0:
            return batch
            
        labels = batch['labels']
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
        from torch.optim.lr_scheduler import LambdaLR
        
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = int(warmup_epochs * steps_per_epoch)
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(optimizer, lr_lambda=lr_lambda)