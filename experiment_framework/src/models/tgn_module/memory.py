import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

  def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
               device="cpu", combination_method='sum'):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.memory_dimension = memory_dimension
    self.input_dimension = input_dimension
    self.message_dimension = message_dimension
    self.device = device

    self.combination_method = combination_method

    

    self.__init_memory__()

  def __init_memory__(self):
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    # Get current device from existing memory or use specified device
    current_device = self.memory.device if hasattr(self, 'memory') else self.device
    
    # Create tensors on correct device
    # memory_tensor = torch.zeros((self.n_nodes, self.memory_dimension), device=current_device)
    memory_tensor = torch.randn(self.n_nodes, self.memory_dimension, device=current_device) * 0.01
    last_update_tensor = torch.zeros(self.n_nodes, device=current_device)
    
    # Register as parameters
    self.memory = nn.Parameter(memory_tensor, requires_grad=True)
    # self.memory = nn.Parameter(torch.randn(self.n_nodes, self.memory_dimension) * 0.01,
    #                        requires_grad=False)
    self.last_update = nn.Parameter(last_update_tensor, requires_grad=False)
    
    self.messages = defaultdict(list)

  def store_raw_messages(self, nodes, node_id_to_messages):
    for node in nodes:
      self.messages[node].extend(node_id_to_messages[node])

  def get_memory(self, node_idxs):
    # Ensure memory is on the same device as the indexing tensor
    if self.memory.device != node_idxs.device:
        self.memory = self.memory.to(node_idxs.device)
    return self.memory[node_idxs, :]

  def set_memory(self, node_idxs, values):
    self.memory[node_idxs, :] = values

  def get_last_update(self, node_idxs):
    if self.last_update.device != node_idxs.device:
        self.last_update = self.last_update.to(node_idxs.device)
    return self.last_update[node_idxs]

  def backup_memory(self):
    messages_clone = {}
    for k, v in self.messages.items():
      messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

    return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

  def restore_memory(self, memory_backup):
    self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    self.messages = defaultdict(list)
    for k, v in memory_backup[2].items():
      self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

  def detach_memory(self):
    self.memory.detach_()

    # Detach all stored messages
    for k, v in self.messages.items():
      new_node_messages = []
      for message in v:
        new_node_messages.append((message[0].detach(), message[1]))

      self.messages[k] = new_node_messages

  def clear_messages(self, nodes):
    for node in nodes:
        if node in self.messages:
            del self.messages[node]

  def to(self, device):
    """Move memory to specified device."""
    if hasattr(self, 'memory'):
        self.memory = self.memory.to(device)
        self.last_update = self.last_update.to(device)
        self.device = device
    return self