from collections import defaultdict
import torch 
import torch.nn as nn
import numpy as np


# class MessageAggregator(nn.Module):
#     def __init__(self, device):
#         super(MessageAggregator, self).__init__()
#         self.device = device

#     def aggregate(self, node_ids, messages):
#         pass 
#     def group_by_id(self, node_ids, messages, timestamps):
#         node_id_to_messages = defaultdict(list)
#         for i, node_id in enumerate(node_ids):
#             node_id_to_messages[node_id].append(messages[i], timestamps[i])
#         return node_id_to_messages

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



# class LastMessageAggregator(MessageAggregator):
#     def __init__(self, device):
#         super(LastMessageAggregator, self).__init__(device)

#     def aggregate(self, node_ids, messages, timestamps):

#         """Only keep the last message for each node"""    
#         unique_node_ids = np.unique(node_ids)
#         unique_messages = []
#         unique_timestamps = []
        
#         to_update_node_ids = []
        
#         for node_id in unique_node_ids:
#             if len(messages[node_id]) > 0:
#                 to_update_node_ids.append(node_id)
#                 unique_messages.append(messages[node_id][-1][0])
#                 unique_timestamps.append(timestamps[node_id][-1][1])
        
#         unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
#         unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

#         return to_update_node_ids, unique_messages, unique_timestamps

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


# class MeanMessageAggregator(MessageAggregator):
#     def __init__(self, device):
#         super(MeanMessageAggregator, self).__init__(device)

#     def aggregate(self, node_ids, messages):
#         """Only keep the last message for each node"""
#         unique_node_ids = np.unique(node_ids)
#         unique_messages = []
#         unique_timestamps = []

#         to_update_node_ids = []
#         n_messages = 0

#         for node_id in unique_node_ids:
#             if len(messages[node_id]) > 0:
#                 n_messages += len(messages[node_id])
#                 to_update_node_ids.append(node_id)
#                 unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))
#                 unique_timestamps.append(messages[node_id][-1][1])

#         unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
#         unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

#         return to_update_node_ids, unique_messages, unique_timestamps


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