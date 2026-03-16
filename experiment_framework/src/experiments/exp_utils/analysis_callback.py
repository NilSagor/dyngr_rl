# ============================================================================
# CALLBACK FOR ANALYSIS DATA COLLECTION
# ============================================================================

from lightning.pytorch.callbacks import Callback

from loguru import logger 


class AnalysisCollector(Callback):
    """Collect analysis data during training epochs."""
    
    def __init__(self):
        self.memory_trace = []
        self.ode_trajectory = {'dirichlet_energy': [], 'embeddings': [], 'time_points': []}
        self.negative_stats = {'hard_ratio': [], 'loss_contrib': []}
        self.walk_stats = None
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Collect data at end of each epoch."""
        # Collect prototype attention
        if hasattr(pl_module, 'get_prototype_attention'):
            try:
                attn = pl_module.get_prototype_attention().detach().cpu()
                self.memory_trace.append(attn)
            except Exception as e:
                logger.debug(f"Could not collect memory trace: {e}")
        
        # Collect ODE dynamics
        if hasattr(pl_module, 'get_ode_trajectory'):
            try:
                traj = pl_module.get_ode_trajectory()
                self.ode_trajectory['dirichlet_energy'].append(traj.get('energy'))
                self.ode_trajectory['time_points'].append(traj.get('time'))
            except Exception as e:
                logger.debug(f"Could not collect ODE trajectory: {e}")
        
        # Collect negative mining stats
        if hasattr(pl_module, 'get_negative_stats'):
            try:
                stats = pl_module.get_negative_stats()
                self.negative_stats['hard_ratio'].append(stats.get('hard_ratio'))
                self.negative_stats['loss_contrib'].append(stats.get('loss_contrib'))
            except Exception as e:
                logger.debug(f"Could not collect negative stats: {e}")
    
    def on_train_end(self, trainer, pl_module):
        """Collect final walk statistics."""
        if hasattr(pl_module, 'get_walk_statistics'):
            try:
                self.walk_stats = pl_module.get_walk_statistics()
            except Exception as e:
                logger.debug(f"Could not collect walk stats: {e}")