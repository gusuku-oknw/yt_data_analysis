import random
import numpy as np
import torch


def set_seed(seed):
    """
    再現性を確保するためのランダムシードを設定する。

    Parameters:
        seed (int): ランダムシードの値。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")
