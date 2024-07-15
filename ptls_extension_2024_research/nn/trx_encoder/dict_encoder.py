from typing import Dict
import pickle

import numpy as np
import torch
from ptls.nn.trx_encoder.encoders import BaseEncoder 


def unpickle_dict(fpath: str):
    with open(fpath, 'rb') as f:
        return pickle.load(f)

class DictEncoder(BaseEncoder):
    def __init__(self, embs_dict: Dict[str, np.ndarray]):
        super().__init__()
        self.embs_dict = embs_dict

    def forward(self, x):
        """
        x is supposed to be a np.ndarray of strings or other hashable
        """
        embs_lst = [torch.tensor(self.embs_dict[el], dtype = torch.float, requires_grad=False) for el in x]
        return torch.stack(embs_lst).to('cuda')  # !!! FIX ME

    @property
    def output_size(self) -> int:
        return next(iter(self.embs_dict.values())).shape[0]
