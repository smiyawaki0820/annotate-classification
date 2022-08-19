# -*- coding: utf-8 -*-

from importlib import import_module
from pathlib import Path
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_module(path):
    module_path, module_class = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, module_class)
