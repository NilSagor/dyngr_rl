from collections import defaultdict
import torch 
import torch.nn as nn
import numpy as np


class MessageAggregator(nn.Module):
    """Base class: aggregate multiple messages per node into one."""
    def __init__(self, device):
        super().__init__()
        self.device = device

    def aggregate(self, node_ids, messages, timestamps):
        """
        Args:
            node_ids: list or numpy array of unique node IDs (int)
            messages: list of torch.Tensor, each of shape [num_messages_for_node, message_dim]
            timestamps: list of torch.Tensor, each of shape [num_messages_for_node]
        Returns:
            to_update: list of node IDs that have at least one message
            agg_messages: torch.Tensor of shape [len(to_update), message_dim]
            agg_timestamps: torch.Tensor of shape [len(to_update)]
        """
        raise NotImplementedError



class LastMessageAggregator(MessageAggregator):
    """Keep only the most recent message per node."""
    def aggregate(self, node_ids, messages, timestamps):
        to_update = []
        agg_msgs = []
        agg_ts = []
        for i, node_id in enumerate(node_ids):
            if len(messages[i]) > 0:
                to_update.append(node_id)
                # most recent message = highest timestamp
                last_idx = torch.argmax(timestamps[i])
                agg_msgs.append(messages[i][last_idx])
                agg_ts.append(timestamps[i][last_idx])
        if to_update:
            agg_msgs = torch.stack(agg_msgs)
            agg_ts = torch.stack(agg_ts)
        return to_update, agg_msgs, agg_ts




class MeanMessageAggregator(MessageAggregator):
    def aggregate(self, node_ids, messages, timestamps):
        to_update = []
        agg_msgs = []
        agg_ts = []
        for i, node_id in enumerate(node_ids):
            if len(messages[i]) > 0:
                to_update.append(node_id)
                agg_msgs.append(torch.mean(messages[i], dim=0))
                agg_ts.append(timestamps[i][-1])   # keep last interaction time
        if to_update:
            agg_msgs = torch.stack(agg_msgs)
            agg_ts = torch.stack(agg_ts)
        return to_update, agg_msgs, agg_ts

def get_message_aggregator(aggregator_type, device):
    if aggregator_type == "last":
        return LastMessageAggregator(device)
    elif aggregator_type == "mean":
        return MeanMessageAggregator(device)
    else:
        raise ValueError(f"Unknown aggregator: {aggregator_type}") 