import pytorch_lightning as pl
from ptls.frames.abs_module import ABSModule
from ptls.frames.coles.losses import ContrastiveLoss
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from ptls.nn.head import Head
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls_extension_2024_research import TrxEncoder_WithCIEmbeddings
from ptls_extension_2024_research.nn.trx_encoder.client_item_encoder import GNNClientItemEncoder


import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional

from ptls_extension_2024_research.graphs.graph import ClientItemGraph
from ptls_extension_2024_research.graphs.utils import MLPPredictor, construct_negative_graph




class ColesGnnIdAdapetr:
    def __init__(self, item_id2graph_id, client_id2graph_id):
        self.item_id2graph_id = item_id2graph_id
        self.client_id2graph_id = client_id2graph_id

    def __call__(self, batch):
        client_ids, item_ids = batch
        graph_item_ids = self.item_id2graph_id[item_ids]
        graph_client_ids = self.client_id2graph_id[client_ids]
        return graph_client_ids, graph_item_ids



class GnnLinkPredictor(nn.Module):
    """
    GNN with all components needed for link prediction
    """
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
                 link_predictor_name: str='MLP',
                 link_predictor_add_sigmoid: bool=True,
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
        self.client_item_g: ClientItemGraph = ClientItemGraph.from_graph_file(graph_file_path)
        self.link_predictor = self._init_link_predictor(link_predictor_name, output_size, link_predictor_add_sigmoid)

        self.client_id2graph_id = torch.load(client_id2graph_id_path)
        self.item_id2graph_id = torch.load(item_id2graph_id_path)



    def _init_gnn(self, gnn_name, in_feats, **gnn_kwags):
        if gnn_name == 'GraphSAGE':
            return GraphSAGE(in_feats=in_feats, **gnn_kwags)
        if gnn_name == 'GAT':
            return GAT(in_feats=in_feats, **gnn_kwags)
        raise Exception(f'No such graph model {gnn_name}')
    

    def _init_link_predictor(self, link_predictor_name, output_size, link_predictor_add_sigmoid):
        if link_predictor_name == 'MLP':
            return MLPPredictor(output_size, link_predictor_add_sigmoid)
        raise Exception(f'No such link predictor {link_predictor_name}')

    def _init_feats(self, train_embeddings, n_size, embedding_dim, feat_path):
        if train_embeddings:
            return nn.Embedding(n_size, embedding_dim)
        if feat_path is not None:
            return torch.load(feat_path)
        raise Exception('Problem with feats')
    

    def forward(self, batch):
        item_ids, user_ids = batch
        node_feats = torch.cat([self.client_feats.weight.data, self.item_feats.weight.data])
        subgraph = self.client_item_graph.get_subgraph(item_ids, user_ids)
        subgraph_node_embeddings = self.gnn(subgraph, node_feats[subgraph.ndata['_ID']])
        return subgraph, subgraph_node_embeddings





class GnnModule(pl.LightningModule):
    def __init__(self, 
                 gnn_link_predictor: GnnLinkPredictor,
                 optimizer_partial,
                 lr_scheduler_partial,
                 neg_items_per_pos: int=1,
                 lp_criterion_name: str='BCELoss',):
        self.gnn_link_predictor = gnn_link_predictor
        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial

        # loss
        self.neg_items_per_pos = neg_items_per_pos
        self.lp_criterion = getattr(nn, lp_criterion_name)()


    def calc_loss(self, g, node_embeddings):
        pos_scores = self.gnn_link_predictor.link_predictor(g, node_embeddings)
        neg_g = construct_negative_graph(g, self.neg_items_per_pos)
        neg_scores = self.gnn_link_predictor.link_predictor(neg_g, node_embeddings)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])])
        loss = self.lp_criterion(scores, labels)
        return loss


    def training_step(self, batch):
        subgraph, subgraph_node_embeddings = self.gnn_link_predictor(batch)
        return self.calc_loss(subgraph, subgraph_node_embeddings)
    

    def validation_step(self, batch):
        subgraph, subgraph_node_embeddings = self.forward(batch)
        val_loss = self.calc_loss(subgraph, subgraph_node_embeddings)
        return val_loss
    

    def on_validation_epoch_end(self):
        # TODO: Add val_loss logging as a metric
        pass
    

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self.parameters())
        scheduler = self._lr_scheduler_partial(optimizer)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                'scheduler': scheduler,
                'monitor': self.metric_name,
            }
        return [optimizer], [scheduler]

