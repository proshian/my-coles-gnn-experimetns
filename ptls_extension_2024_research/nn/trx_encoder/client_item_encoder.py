from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import dgl

from ptls_extension_2024_research.graphs.static_models.gnn import GraphSAGE, GAT


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
        return torch.zeros(batch_size, seq_len, self.output_size, device=item_ids.device)

    @property
    def output_size(self):
        return self.__output_size


class StaticGNNTrainableClientItemEncoder(BaseClientItemEncoder):
    def __init__(self,
                 n_users: int,
                 n_items: int,
                 output_size: int=10,
                 embedding_dim: int=64,
                 train_client_embeddings: bool=True,
                 train_item_embeddings: bool=True,
                 graph_file_path: Optional[str]=None,
                 client_id2graph_id_path: str='',
                 item_id2graph_id_path: str='',
                 graph_id2client_id_path: str = '',
                 graph_id2item_id_path: str = '',
                 client_feats_path: Optional[str]=None,
                 item_feats_path: Optional[str]=None,
                 gnn_name: str='graphsage',
                 **gnn_kwags):
        super().__init__()
        self.__output_size = output_size
        self.n_users = n_users
        self.n_items = n_items
        self.train_client_embeddings = train_client_embeddings
        self.train_item_embeddings = train_item_embeddings

        self.client_feats = self._init_feats(train_client_embeddings, n_users, embedding_dim, client_feats_path)
        self.item_feats = self._init_feats(train_item_embeddings, n_items, embedding_dim, item_feats_path)

        self.gnn = self._init_gnn(gnn_name, in_feats=embedding_dim, h_feats=output_size, **gnn_kwags)
        g_list, _ = dgl.load_graphs(graph_file_path, [0])
        self.g = g_list[0]

        self.client_id2graph_id = torch.load(client_id2graph_id_path)
        self.item_id2graph_id = torch.load(item_id2graph_id_path)
        self.graph_id2client_id = torch.load(graph_id2client_id_path)
        self.graph_id2item_id = torch.load(graph_id2item_id_path)


    def _init_gnn(self, gnn_name, in_feats, **gnn_kwags):
        if gnn_name == 'GraphSAGE':
            return GraphSAGE(in_feats=in_feats, **gnn_kwags)
        if gnn_name == 'GAT':
            return GAT(in_feats=in_feats, **gnn_kwags)
        raise Exception(f'No such graph model {gnn_name}')

    def _init_feats(self, train_embeddings, n_size, embedding_dim, feat_path):
        if train_embeddings:
            return nn.Embedding(n_size, embedding_dim)
        if feat_path is not None:
            return torch.load(feat_path)
        raise Exception('Problem with feats')

    def get_node_embeddings(self):
        if self.train_user_embeddings:
            user_feats = self.user_embeddings.weight.data
        else:
            user_feats = self.user_feats
        if self.train_item_embeddings:
            item_feats = self.item_embeddings.weight.data
        else:
            item_feats = self.item_feats
        assert user_feats is not None and item_feats is not None
        user_feats = self
        item_feats = torch.index_select(item_feats, dim=0, index=item_order_in_graph)
        node_feats = torch.cat([user_feats, item_feats])
        return self.gnn(self.g, node_feats).sigmoid()

    def forward(self, client_ids: torch.Tensor, item_ids: torch.Tensor):
        """
        client_ids: torch.Tensor, shape: (batch_size,)
        item_ids: torch.Tensor, shape: (batch_size, seq_len)
        """
        batch_size, seq_len = item_ids.size()
        return torch.zeros(batch_size, seq_len, self.output_size, device=item_ids.device)

    @property
    def output_size(self):
        return self.__output_size


class GNNPretrainedClientItemEncoder(BaseClientItemEncoder):
    def __init__(self, client_embs_path: str, item_embs_path: str,
                 client_id2graph_id_path: str, item_id2graph_id_path: str):
        super().__init__()
        self.client_embeddings = torch.load(client_embs_path)
        self.client_id2graph_id = torch.load(client_id2graph_id_path)

        self.item_embeddings = torch.load(item_embs_path)
        self.item_id2graph_id = torch.load(item_id2graph_id_path)
        self.__output_size = self.client_embeddings.shape[0]

    def forward(self, client_ids: torch.Tensor, item_ids: torch.Tensor):
        """
        client_ids: torch.Tensor, shape: (batch_size,)
        item_ids: torch.Tensor, shape: (batch_size, seq_len)
        """
        graph_item_ids = self.item_id2graph_id[item_ids]
        return self.item_embeddings[graph_item_ids]

    @property
    def output_size(self):
        return self.__output_size