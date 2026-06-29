import os
import random
import numpy as np

import torch

def set_all_seeds(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    print(f"Random seeds set to: {seed}")


def resolve_camera_data_root(data_root, enable_moving_camera):
    camera_root = "moving" if enable_moving_camera else "static"

    if not os.path.isabs(data_root):
        data_root = os.path.abspath(data_root)

    return os.path.join(data_root, camera_root)
