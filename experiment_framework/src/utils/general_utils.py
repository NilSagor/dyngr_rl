import os 
import random
import torch 
import numpy as np
import lightning as L
import yaml
from datetime import datetime

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_ids=None):
    if gpu_ids is None or gpu_ids == 0:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDa not available, using cpu")
        device = torch.device('cpu')
    return device

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_experiment_dir(base_dir, experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    return exp_dir
