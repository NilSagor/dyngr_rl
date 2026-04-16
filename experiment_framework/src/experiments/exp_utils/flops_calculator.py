from fvcore.nn import FlopCountAnalysis, parameter_count_table
from typing import Dict
import torch


class FLOPsCalculator:
    """Calculate FLOPs for HiCoSTv3 properly."""
    
    @staticmethod
    def calculate_model_flops(model, sample_input: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Args:
            model: Your HiCoSTv3 model
            sample_input: Dict with keys like 'src', 'dst', 'time', etc.
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Move sample to device
        sample_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in sample_input.items()}
        
        # Count FLOPs
        flops = FlopCountAnalysis(model, sample_input)
        
        # Detailed breakdown
        flops_by_module = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                mod_flops = flops.by_module().get(name, 0)
                if mod_flops > 0:
                    flops_by_module[name] = mod_flops
                    
        total_flops = flops.total()
        
        return {
            'total_flops': total_flops,
            'total_gflops': total_flops / 1e9,
            'by_module': flops_by_module,
            'by_operator': flops.by_operator(),
            'parameter_count': parameter_count_table(model)
        }
    
    @staticmethod
    def print_summary(model, sample_input: Dict[str, torch.Tensor]):
        """Print formatted FLOPs summary."""
        stats = FLOPsCalculator.calculate_model_flops(model, sample_input)
        
        print("\n" + "="*70)
        print("FLOPs ANALYSIS")
        print("="*70)
        print(f"Total FLOPs: {stats['total_flops']:,.0f} ({stats['total_gflops']:.2f} GFLOPs)")
        print(f"Total Params: {sum(p.numel() for p in model.parameters()):,}")
        print("-"*70)
        print("By Module (top 10):")
        sorted_modules = sorted(stats['by_module'].items(), key=lambda x: x[1], reverse=True)[:10]
        for name, flops in sorted_modules:
            print(f"  {name:50s}: {flops/1e9:>8.2f} GFLOPs")
        print("="*70)
        
        return stats