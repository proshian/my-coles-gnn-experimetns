from typing import List, Tuple, Dict
from functools import reduce
from operator import iadd
from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.padded_batch import PaddedBatch 
import torch
import numpy as np  # For type hinting.

FeatureDictType = Dict[str, np.ndarray]


class DatasetWithId(torch.utils.data.Dataset):
    def __init__(self, data: torch.utils.data.Dataset):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], idx

    @staticmethod
    def collate_fn(batch: List[Tuple[FeatureDictType, int]]
                   ) -> Tuple[PaddedBatch, torch.Tensor]:
        feature_dicts, client_ids = zip(*batch)
        padded_batch = collate_feature_dict(feature_dicts)
        return padded_batch, torch.LongTensor(client_ids)