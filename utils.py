import torch
import torch.nn.functional as F
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Dict(dict):
    def __init__(self, dct=None):
        super().__init__()
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = Dict(value)
            self[key] = value

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
