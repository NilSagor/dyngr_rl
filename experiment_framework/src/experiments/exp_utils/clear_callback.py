import torch 
from lightning.pytorch.callbacks import Callback

class ClearCacheCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()