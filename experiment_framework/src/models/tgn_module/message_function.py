import torch.nn as nn

class MessageFunction(nn.Module):
    def forward(self, raw_messages):
        return NotImplementedError
    
class MLPMessageFunction(MessageFunction):
    def __init__(self, raw_message_dimension, message_dimension):
        super(MLPMessageFunction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_message_dimension, raw_message_dimension//2),
            nn.ReLU(),
            nn.Linear(raw_message_dimension//2, message_dimension),
        )

    def forward(self, raw_messages):
        messages = self.mlp(raw_messages)
        return messages
    

class IdentityMessageFunction(MessageFunction):
    def forward(self, raw_messages):
        return raw_messages

def get_message_function(module_type, raw_message_dimension, message_dimension):
    if module_type == "mlp":
        return MLPMessageFunction(raw_message_dimension, message_dimension)
    elif module_type == "identity":
        return IdentityMessageFunction()
    else:
        raise ValueError(f"Unknown message function type: {module_type}")