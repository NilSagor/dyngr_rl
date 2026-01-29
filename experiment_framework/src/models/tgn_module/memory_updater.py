import torch 
import torch.nn as nn


class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        pass

class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dim, memory_dim, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dim)
    self.message_dim = message_dim
    self.device = device

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
        return

    # Get current last update times
    last_update = self.memory.get_last_update(unique_node_ids)
    
    # Skip updates that would go backwards in time
    valid_mask = last_update <= timestamps
    if not valid_mask.all():
        # Log warning for debugging
        print(f"Warning: Skipping {(~valid_mask).sum().item()} memory updates in the past")
        unique_node_ids = unique_node_ids[valid_mask]
        unique_messages = unique_messages[valid_mask]
        timestamps = timestamps[valid_mask]
        
        if len(unique_node_ids) == 0:
            return

    # Update memory for valid entries
    memory = self.memory.get_memory(unique_node_ids)
    updated_memory = self.memory_updater(unique_messages, memory)
    self.memory.set_memory(unique_node_ids, updated_memory)
    self.memory.last_update[unique_node_ids] = timestamps

def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    updated_memory = self.memory.memory.data.clone()
    updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps

    return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dim, memory_dim, device):
    super(GRUMemoryUpdater, self).__init__(memory, message_dim, memory_dim, device)

    self.memory_updater = nn.GRUCell(input_size=message_dim,
                                     hidden_size=memory_dim)

class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dim, memory_dim, device):
    super(RNNMemoryUpdater, self).__init__(memory, message_dim, memory_dim, device)

    self.memory_updater = nn.RNNCell(input_size=message_dim,
                                     hidden_size=memory_dim)

def get_memory_updater(module_type, memory, message_dim, memory_dim, device):
    if module_type == "gru":
        return GRUMemoryUpdater(memory, message_dim, memory_dim, device)
    elif module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dim, memory_dim, device)
        