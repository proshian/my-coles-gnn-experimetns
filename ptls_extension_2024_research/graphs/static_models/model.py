from typing import Dict, List

import torch
import torch.nn as nn

from ptls_extension_2024_research.graphs.static_models.gnn import GraphSAGE


class MainGNNModel(nn.Module):
    def __init__(self, n_users, n_items, embeddings_dim, train_user_embeddings=True, train_item_embeddings=True,
                 gnn_name='graphsage', **gnn_kwags):
        if train_user_embeddings:
            self.user_embeddings = nn.Embedding(n_users, embeddings_dim)
        if train_item_embeddings:
            self.item_embeddings = nn.Embedding(n_items, embeddings_dim)
        self.gnn = self._init_gnn(gnn_name, in_feats=embeddings_dim, **gnn_kwags)

    def _init_gnn(self, gnn_name, in_feats, **gnn_kwags):
        if gnn_name == 'graphsage':
            return GraphSAGE(in_feats=in_feats, **gnn_kwags)
        raise Exception(f'No such graph model {gnn_name}')

    def forward(self, g, user_feats, item_feats, user_order_in_graph: torch.Tensor, item_order_in_graph: torch.Tensor):
        if self.train_user_embeddings:
            user_feats = self.user_embeddings.weight.data
        if self.train_item_embeddings:
            item_feats = self.item_embeddings.weight.data
        assert user_feats is not None and item_feats is not None
        user_feats = torch.index_select(user_feats, dim=0, index=user_order_in_graph)
        item_feats = torch.index_select(item_feats, dim=0, index=item_order_in_graph)
        node_feats = torch.cat([user_feats, item_feats])
        return self.gnn(g, node_feats).sigmoid()
