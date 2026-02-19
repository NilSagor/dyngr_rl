import torch 
import torch.nn as nn



class PrototypeMemory(nn.Module):
    """
        Learnable prototype vectors for a single node.
        Each node has K prototypes representing stable "modes" or "roles".
    """    
    def __init__(self, 
                 num_prototypes:int, 
                 prototype_dim:int, 
                 node_id:int, 
                 initialization:str='xavier'):
        super(PrototypeMemory, self).__init__()
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.node_id = node_id

        # 
        self.prototypes = nn.Parameter(
            torch.empty(num_prototypes, prototype_dim)
        )

        # initialize prototypes
        if initialization == 'xavier':
            nn.init.xavier_uniform_(self.prototypes)
        elif initialization == "normal":
            nn.init.normal_(self.prototypes, mean=0.0, std=0.1)
        elif initialization == "uniform":
            nn.init.uniform_(self.prototypes, -0.1, 0.1)
        else:
            raise ValueError(f"Unknown initialization: {initialization}")
        

    def forward(self)->torch.Tensor:
        """Return prototype vectors"""
        return self.prototypes
    
    def get_prototype(self, idx:int)->torch.Tensor:
        """Get a specific prototype"""
        return self.prototypes[idx]
    


