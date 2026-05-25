import random
import numpy as np

def set_all_seeds(seed = 42):

    random.seed(seed)
    np.random.seed(seed)

    print(f"Random seeds set to: {seed}")