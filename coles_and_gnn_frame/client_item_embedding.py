import numpy as np
import torch
import torch.nn as nn


class BaseClientItemEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, client_ids: torch.Tensor, item_ids: torch.Tensor):
        """
        client_ids: torch.Tensor, shape: (batch_size,)
        item_ids: torch.Tensor, shape: (batch_size, seq_len)
        """
        raise NotImplementedError()

    @property
    def output_size(self) -> int:
        raise NotImplementedError()


class DummyGNNClientItemEncoder(BaseClientItemEncoder):
    def __init__(self, output_size = 10):
        super().__init__()
        self.__output_size = output_size

    def forward(self, client_ids: torch.Tensor, item_ids: torch.Tensor):
        """
        client_ids: torch.Tensor, shape: (batch_size,)
        item_ids: torch.Tensor, shape: (batch_size, seq_len)
        """
        batch_size, seq_len = item_ids.size()
        return torch.zeros(batch_size, seq_len, self.output_size)

    @property
    def output_size(self):
        return self.__output_size
