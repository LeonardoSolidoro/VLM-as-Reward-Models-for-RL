import os
import random
import numpy as np

def set_all_seeds(seed = 42):

    random.seed(seed)
    np.random.seed(seed)

    print(f"Random seeds set to: {seed}")


def resolve_camera_data_root(data_root, enable_moving_camera):
    camera_root = "moving" if enable_moving_camera else "static"

    if not os.path.isabs(data_root):
        data_root = os.path.abspath(data_root)

    return os.path.join(data_root, camera_root)