import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional


def temporal_train_val_test_split(
    edges: torch.Tensor,
    timestamps: torch.Tensor,
    edge_features: Optional[torch.Tensor] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    respect_chronological_order: bool = True
) -> Dict[str, Any]:
    """Split temporal graph data into train/validation/test sets.
    
    This function creates a single split that can be used for both transductive 
    and inductive evaluation, rather than creating separate datasets.
    
    Args:
        edges: Edge indices [num_edges, 2]
        timestamps: Edge timestamps [num_edges]
        edge_features: Optional edge features [num_edges, feature_dim]
        train_ratio: Proportion of edges for training
        val_ratio: Proportion of edges for validation
        test_ratio: Proportion of edges for testing
        respect_chronological_order: Whether to maintain temporal order in splits
        
    Returns:
        Dictionary containing:
        - train_edges, val_edges, test_edges
        - train_timestamps, val_timestamps, test_timestamps
        - train_edge_features, val_edge_features, test_edge_features (if provided)
        - split_indices: indices for each split
    """
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    num_edges = len(edges)
    
    if respect_chronological_order:
        # Sort edges by timestamp to maintain chronological order
        sorted_indices = torch.argsort(timestamps)
        edges = edges[sorted_indices]
        timestamps = timestamps[sorted_indices]
        if edge_features is not None:
            edge_features = edge_features[sorted_indices]
    
    # Calculate split points
    train_end = int(num_edges * train_ratio)
    val_end = train_end + int(num_edges * val_ratio)
    
    # Create splits
    splits = {
        'train_edges': edges[:train_end],
        'train_timestamps': timestamps[:train_end],
        'val_edges': edges[train_end:val_end],
        'val_timestamps': timestamps[train_end:val_end],
        'test_edges': edges[val_end:],
        'test_timestamps': timestamps[val_end:],
        'split_indices': {
            'train': torch.arange(0, train_end),
            'val': torch.arange(train_end, val_end),
            'test': torch.arange(val_end, num_edges)
        }
    }
    
    if edge_features is not None:
        splits['train_edge_features'] = edge_features[:train_end]
        splits['val_edge_features'] = edge_features[train_end:val_end]
        splits['test_edge_features'] = edge_features[val_end:]
    
    return splits


def create_inductive_split(
    edges: torch.Tensor,
    timestamps: torch.Tensor,
    edge_features: Optional[torch.Tensor] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    unseen_node_ratio: float = 0.2
) -> Dict[str, Any]:
    """Create inductive split where test set contains unseen nodes.
    
    This creates a true inductive setting for evaluation.
    
    Args:
        edges: Edge indices [num_edges, 2]
        timestamps: Edge timestamps [num_edges]
        edge_features: Optional edge features
        train_ratio: Proportion of edges for training
        val_ratio: Proportion of edges for validation
        test_ratio: Proportion of edges for testing
        unseen_node_ratio: Proportion of nodes to hold out for testing
        
    Returns:
        Dictionary with inductive splits
    """
    
    # Sort chronologically
    sorted_idx = torch.argsort(timestamps)
    edges = edges[sorted_idx]
    timestamps = timestamps[sorted_idx]
    if edge_features is not None:
        edge_features = edge_features[sorted_idx]
    
    num_edges = len(edges)
    train_end = int(num_edges * train_ratio)
    val_end = train_end + int(num_edges * val_ratio)
    
    # Get nodes from training period only
    train_nodes = torch.unique(edges[:train_end].flatten())
    
    # Randomly hold out some training nodes as unseen
    num_unseen = int(len(train_nodes) * unseen_node_ratio)
    perm = torch.randperm(len(train_nodes))
    unseen_nodes = train_nodes[perm[:num_unseen]]
    seen_nodes = train_nodes[perm[num_unseen:]]
    
    # Create initial chronological splits
    train_edges = edges[:train_end]
    train_timestamps = timestamps[:train_end]
    val_edges = edges[train_end:val_end]
    val_timestamps = timestamps[train_end:val_end]
    test_edges = edges[val_end:]
    test_timestamps = timestamps[val_end:]
    
    if edge_features is not None:
        train_edge_features = edge_features[:train_end]
        val_edge_features = edge_features[train_end:val_end]
        test_edge_features = edge_features[val_end:]
    
  # Remove any edges involving unseen nodes from train/val
    def filter_edges(edges, timestamps, edge_features, unseen_nodes):
        mask = ~(torch.isin(edges[:, 0], unseen_nodes) | torch.isin(edges[:, 1], unseen_nodes))
        filtered_edges = edges[mask]
        filtered_timestamps = timestamps[mask]
        filtered_features = edge_features[mask] if edge_features is not None else None
        return filtered_edges, filtered_timestamps, filtered_features

    train_edges, train_timestamps, train_edge_features = filter_edges(
        train_edges, train_timestamps, 
        train_edge_features if edge_features is not None else None,
        unseen_nodes
    )
    
    val_edges, val_timestamps, val_edge_features = filter_edges(
        val_edges, val_timestamps,
        val_edge_features if edge_features is not None else None,
        unseen_nodes
    )  
    # Create splits
    # Test set remains unchanged (contains both seen and unseen node edges)
    
    splits = {
        'train_edges': train_edges,
        'train_timestamps': train_timestamps,
        'val_edges': val_edges,
        'val_timestamps': val_timestamps,
        'test_edges': test_edges,
        'test_timestamps': test_timestamps,
        'seen_nodes': seen_nodes,
        'unseen_nodes': unseen_nodes
    }
    
    if edge_features is not None:
        splits['train_edge_features'] = train_edge_features
        splits['val_edge_features'] = val_edge_features
        splits['test_edge_features'] = test_edge_features
    
    return splits


def prepare_evaluation_data(
    dataset_dict: Dict[str, Any],
    evaluation_type: str = 'transductive'
) -> Dict[str, Any]:
    """Prepare data for either transductive or inductive evaluation.
    
    This function creates the appropriate data structure for the specified
    evaluation type without duplicating datasets.
    
    Args:
        dataset_dict: Dataset dictionary from split functions
        evaluation_type: 'transductive' or 'inductive'
        
    Returns:
        Dictionary with evaluation-specific data preparation
    """
    
    if evaluation_type == 'transductive':
        # All nodes are visible in all splits
        return {
            'train_data': {
                'edges': dataset_dict['train_edges'],
                'timestamps': dataset_dict['train_timestamps'],
                'edge_features': dataset_dict.get('train_edge_features'),
                'nodes': torch.unique(dataset_dict['train_edges'].flatten())
            },
            'val_data': {
                'edges': dataset_dict['val_edges'],
                'timestamps': dataset_dict['val_timestamps'],
                'edge_features': dataset_dict.get('val_edge_features'),
                'nodes': torch.unique(dataset_dict['val_edges'].flatten())
            },
            'test_data': {
                'edges': dataset_dict['test_edges'],
                'timestamps': dataset_dict['test_timestamps'],
                'edge_features': dataset_dict.get('test_edge_features'),
                'nodes': torch.unique(dataset_dict['test_edges'].flatten())
            },
            'all_nodes': torch.unique(torch.cat([
                dataset_dict['train_edges'].flatten(),
                dataset_dict['val_edges'].flatten(),
                dataset_dict['test_edges'].flatten()
            ])),
            'evaluation_type': 'transductive'
        }
    
    elif evaluation_type == 'inductive':
        # Use inductive split if available, otherwise create it
        if 'seen_nodes' in dataset_dict:
            return {
                'train_data': {
                    'edges': dataset_dict['train_edges'],
                    'timestamps': dataset_dict['train_timestamps'],
                    'edge_features': dataset_dict.get('train_edge_features'),
                    'nodes': dataset_dict['seen_nodes']
                },
                'val_data': {
                    'edges': dataset_dict['val_edges'],
                    'timestamps': dataset_dict['val_timestamps'],
                    'edge_features': dataset_dict.get('val_edge_features'),
                    'nodes': dataset_dict['seen_nodes']
                },
                'test_data': {
                    'edges': dataset_dict['test_edges'],
                    'timestamps': dataset_dict['test_timestamps'],
                    'edge_features': dataset_dict.get('test_edge_features'),
                    'nodes': torch.unique(dataset_dict['test_edges'].flatten())
                },
                'seen_nodes': dataset_dict['seen_nodes'],
                'unseen_nodes': dataset_dict.get('unseen_nodes', torch.tensor([])),
                'evaluation_type': 'inductive'
            }
        else:
            # Create inductive split on the fly
            inductive_splits = create_inductive_split(
                dataset_dict['train_edges'],
                dataset_dict['train_timestamps'],
                dataset_dict.get('train_edge_features')
            )
            return prepare_evaluation_data(inductive_splits, 'inductive')
    
    else:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")


def compute_temporal_statistics(edges: torch.Tensor, timestamps: torch.Tensor) -> Dict[str, Any]:
    """Compute statistics for temporal graph dataset.
    
    Args:
        edges: Edge indices [num_edges, 2]
        timestamps: Edge timestamps [num_edges]
        
    Returns:
        Dictionary with temporal statistics
    """
    
    num_edges = len(edges)
    num_nodes = int(edges.max().item() + 1)
    
    # Time statistics
    time_span = timestamps.max().item() - timestamps.min().item()
    avg_inter_arrival = time_span / num_edges
    
    # Edge statistics
    edge_density = num_edges / (num_nodes * (num_nodes - 1))
    
    # Node degree statistics
    node_degrees = torch.zeros(num_nodes)
    for edge in edges:
        node_degrees[edge[0]] += 1
        node_degrees[edge[1]] += 1
    
    avg_degree = node_degrees.mean().item()
    max_degree = node_degrees.max().item()
    min_degree = node_degrees.min().item()
    
    # Temporal edge statistics
    edges_per_time = num_edges / time_span if time_span > 0 else 0
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'time_span': time_span,
        'avg_inter_arrival_time': avg_inter_arrival,
        'edge_density': edge_density,
        'avg_node_degree': avg_degree,
        'max_node_degree': max_degree,
        'min_node_degree': min_degree,
        'edges_per_unit_time': edges_per_time
    }


def validate_temporal_order(edges: torch.Tensor, timestamps: torch.Tensor) -> bool:
    """Validate that edges are in chronological order.
    
    Args:
        edges: Edge indices [num_edges, 2]
        timestamps: Edge timestamps [num_edges]
        
    Returns:
        True if edges are chronologically ordered
    """
    
    return torch.all(timestamps[:-1] <= timestamps[1:])


def remove_duplicate_edges(edges: torch.Tensor, timestamps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove duplicate edges keeping the earliest timestamp.
    
    Args:
        edges: Edge indices [num_edges, 2]
        timestamps: Edge timestamps [num_edges]
        
    Returns:
        Tuple of (unique_edges, unique_timestamps)
    """
    
    # Create edge tuples for hashing
    edge_tuples = torch.cat([edges, timestamps.unsqueeze(-1)], dim=-1)
    
    # Find unique edges (considering undirected)
    edge_tuples_sorted, _ = torch.sort(edges, dim=1)  # Make undirected
    edge_tuples_final = torch.cat([edge_tuples_sorted, timestamps.unsqueeze(-1)], dim=-1)
    
    # Get unique indices
    _, unique_indices = torch.unique(edge_tuples_final, dim=0, return_inverse=True)
    
    # Take first occurrence (earliest timestamp)
    final_indices = []
    seen = set()
    for i, idx in enumerate(unique_indices):
        if idx.item() not in seen:
            final_indices.append(i)
            seen.add(idx.item())
    
    final_indices = torch.tensor(final_indices, dtype=torch.long)
    
    return edges[final_indices], timestamps[final_indices]