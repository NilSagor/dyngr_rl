# experiments/runner/__init__.py
from .base_runner import BaseRunner
from .tawrmac_runner import TAWRMACRunner
from .tgn_runner import TGNRunner
from .hicost_runner import HiCoSTRunner
from .hicostdev1_runner import HiCoSTdev1Runner
from .hicostdev2_runner import HiCoSTdev2Runner
# from .freedyg_runner import FreeDyGRunner   # if exists


# Create a registry mapping model names to runner classes
RUNNER_REGISTRY = {
    "TAWRMACv1": TAWRMACRunner,
    "TGN": TGNRunner,
    "TGNv2": TGNRunner,
    "TGNv3": TGNRunner,
    "TGNv4": TGNRunner,
    "TGNv5": TGNRunner,
    "TGNv6": TGNRunner,
    "TGNv7": TGNRunner,
    "DyGFormer": TGNRunner,           # DyGFormer uses same runner as TGN
    # "FreeDyG": FreeDyGRunner,
    "HiCoST": HiCoSTRunner,
    "HiCoSTv2": HiCoSTRunner,
    "HiCoSTv3": HiCoSTRunner,
    "HiCoSTv4": HiCoSTRunner,
    "HiCoSTdev1": HiCoSTdev1Runner,
    "HiCoSTdev2": HiCoSTdev2Runner,
}


def get_runner(model_name: str):
    try:
        return RUNNER_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"No runner registered for model: {model_name}")