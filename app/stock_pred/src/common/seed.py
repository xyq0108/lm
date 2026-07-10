"""随机种子管理"""
import random
import os

import numpy as np


def set_seed(seed: int) -> None:
    """全局设置随机种子，保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
