from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ptls_extension_2024_research.graphs.graph import ClientItemGraph
from ptls_extension_2024_research.graphs.static_models.gnn import GraphSAGE, GAT
from ptls_extension_2024_research.graphs.utils import MLPPredictor, construct_negative_graph
from ptls_extension_2024_research.frames.gnn.gnn_module import ColesBatchToSubgraphConverter, GnnLinkPredictor


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
        return torch.zeros(batch_size, seq_len, self.__output_size, device=item_ids.device)

    @property
    def output_size(self):
        return self.__output_size






class StaticGNNTrainableClientItemEncoder(BaseClientItemEncoder):
    def __init__(self,
                 data_adapter: ColesBatchToSubgraphConverter,
                 gnn_link_predictor: GnnLinkPredictor,) -> None:
        self.gnn_link_predictor = gnn_link_predictor
        self.data_adapter = data_adapter

    def forward(self, client_ids: torch.Tensor, item_ids: torch.Tensor):
        """
        client_ids: torch.Tensor, shape: (batch_size,)
        item_ids: torch.Tensor, shape: (batch_size, seq_len)
        """
        data_adapter_result = self.data_adapter(client_ids, item_ids)
        subgraph, coles_item_ids2subgraph_item_ids = data_adapter_result['subgraph'], data_adapter_result['coles_item_ids2subgraph_item_ids']
        subgraph_node_embeddings = self.gnn_link_predictor(subgraph)
        item_embeddings = subgraph_node_embeddings[coles_item_ids2subgraph_item_ids] # []
        return item_embeddings
    
    @property
    def output_size(self):
        return self.gnn_link_predictor.__output_size


    























# class OLD_StaticGNNTrainableClientItemEncoder(BaseClientItemEncoder):
#     def __init__(self,
#                  n_users: int,
#                  n_items: int,
#                  output_size: int=10,
#                  embedding_dim: int=64,
#                  train_client_embeddings: bool=True,
#                  train_item_embeddings: bool=True,
#                  graph_file_path: Optional[str]=None,
#                  client_id2graph_id_path: str='',
#                  item_id2graph_id_path: str='',
#                  graph_id2client_id_path: str = '',
#                  graph_id2item_id_path: str = '',
#                  client_feats_path: Optional[str]=None,
#                  item_feats_path: Optional[str]=None,
#                  neg_items_per_pos: int=1,
#                  lp_criterion_name: str='BCELoss',
#                  link_predictor_name: str='MLP',
#                  link_predictor_add_sigmoid: bool=True,
#                  gnn_name: str='graphsage',
#                  **gnn_kwags):
#         super().__init__()
#         self.__output_size = output_size
#         self.n_users = n_users
#         self.n_items = n_items
#         self.train_client_embeddings = train_client_embeddings
#         self.train_item_embeddings = train_item_embeddings

#         self.client_feats = self._init_feats(train_client_embeddings, n_users, embedding_dim, client_feats_path)
#         self.item_feats = self._init_feats(train_item_embeddings, n_items, embedding_dim, item_feats_path)

#         self.gnn = self._init_gnn(gnn_name, in_feats=embedding_dim, h_feats=output_size, **gnn_kwags)
#         self.client_item_g: ClientItemGraph = ClientItemGraph.from_graph_file(graph_file_path)
#         self.link_predictor = self._init_link_predictor(link_predictor_name, output_size, link_predictor_add_sigmoid)

#         self.client_id2graph_id = torch.load(client_id2graph_id_path)
#         self.item_id2graph_id = torch.load(item_id2graph_id_path)

#         # loss
#         self.neg_items_per_pos = neg_items_per_pos
#         self.lp_criterion = getattr(nn, lp_criterion_name)()


#     def _init_gnn(self, gnn_name, in_feats, **gnn_kwags):
#         if gnn_name == 'GraphSAGE':
#             return GraphSAGE(in_feats=in_feats, **gnn_kwags)
#         if gnn_name == 'GAT':
#             return GAT(in_feats=in_feats, **gnn_kwags)
#         raise Exception(f'No such graph model {gnn_name}')

#     def _init_link_predictor(self, link_predictor_name, output_size, link_predictor_add_sigmoid):
#         if link_predictor_name == 'MLP':
#             return MLPPredictor(output_size, link_predictor_add_sigmoid)
#         raise Exception(f'No such link predictor {link_predictor_name}')

#     def _init_feats(self, train_embeddings, n_size, embedding_dim, feat_path):
#         if train_embeddings:
#             return nn.Embedding(n_size, embedding_dim)
#         if feat_path is not None:
#             return torch.load(feat_path)
#         raise Exception('Problem with feats')

#     def get_node_embeddings(self, client_ids: torch.Tensor, item_ids: torch.Tensor, calc_loss):
#         """
#         client_ids и item_ids - графовые id, а не coles'овые
#         """
#         if self.train_user_embeddings:
#             user_feats = self.user_embeddings.weight.data
#         else:
#             user_feats = self.user_feats
#         if self.train_item_embeddings:
#             item_feats = self.item_embeddings.weight.data
#         else:
#             item_feats = self.item_feats
#         assert user_feats is not None and item_feats is not None


#         node_feats = torch.cat([user_feats, item_feats])
#         subgraph = self.client_item_g.create_subgraph(client_ids, item_ids)
#         subgraph_node_embeddings = self.gnn(subgraph, node_feats[subgraph.ndata['_ID']])
#         original_embeddings = torch.zeros(self.client_item_g.g.number_of_nodes(), self.__output_size)
#         original_embeddings[subgraph.ndata['_ID']] = subgraph_node_embeddings
#         loss = None
#         if calc_loss:
#             loss = self.calc_loss(subgraph, subgraph_node_embeddings)
#         return original_embeddings, loss

#     def forward(self, client_ids: torch.Tensor, item_ids: torch.Tensor, calc_loss=False):
#         """
#         client_ids: torch.Tensor, shape: (batch_size,)
#         item_ids: torch.Tensor, shape: (batch_size, seq_len)
#         """
#         graph_item_ids = self.item_id2graph_id[item_ids]
#         graph_client_ids = self.client_id2graph_id[client_ids]
#         cur_node_embeddings, loss = self.get_node_embeddings(graph_client_ids, graph_item_ids, calc_loss)
#         item_embeddings = cur_node_embeddings[graph_item_ids]
#         return item_embeddings, loss

#     def get_user_item_ids(self, client_ids: torch.Tensor, item_ids: torch.Tensor, calc_loss=False):
#         graph_client_ids = self.client_id2graph_id[client_ids]
#         graph_item_ids = self.item_id2graph_id[item_ids]
#         cur_node_embeddings, loss = self.get_node_embeddings(graph_client_ids, graph_item_ids, calc_loss)
#         return cur_node_embeddings[graph_client_ids], cur_node_embeddings[graph_item_ids], loss

#     def calc_loss(self, g, node_embeddings):
#         pos_scores = self.link_predictor(g, node_embeddings)
#         neg_g = construct_negative_graph(g, self.neg_items_per_pos)
#         neg_scores = self.link_predictor(neg_g, node_embeddings)
#         scores = torch.cat([pos_scores, neg_scores])
#         labels = torch.cat([torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])])
#         loss = self.lp_criterion(scores, labels)
#         return loss

#     @property
#     def output_size(self):
#         return self.__output_size


# class GNNPretrainedClientItemEncoder(BaseClientItemEncoder):
#     def __init__(self, client_embs_path: str, item_embs_path: str,
#                  client_id2graph_id_path: str, item_id2graph_id_path: str):
#         super().__init__()
#         self.client_embeddings = torch.load(client_embs_path)
#         self.client_id2graph_id = torch.load(client_id2graph_id_path)

#         self.item_embeddings = torch.load(item_embs_path)
#         self.item_id2graph_id = torch.load(item_id2graph_id_path)
#         self.__output_size = self.client_embeddings.shape[0]

#     def forward(self, client_ids: torch.Tensor, item_ids: torch.Tensor):
#         """
#         client_ids: torch.Tensor, shape: (batch_size,)
#         item_ids: torch.Tensor, shape: (batch_size, seq_len)
#         """
#         graph_item_ids = self.item_id2graph_id[item_ids]
#         loss = None
#         return self.item_embeddings[graph_item_ids], loss

#     def get_user_item_ids(self, client_ids: torch.Tensor, item_ids: torch.Tensor):
#         graph_client_ids = self.client_id2graph_id[client_ids]
#         graph_item_ids = self.item_id2graph_id[item_ids]
#         loss = None
#         return self.cur_node_embeddings[graph_client_ids], self.cur_node_embeddings[graph_item_ids], loss

#     @property
#     def output_size(self):
#         return self.__output_size
