import pytorch_lightning as pl
import torch
import torch.nn as nn

from ptls_extension_2024_research.graphs.graph import ClientItemGraph
from ptls_extension_2024_research.graphs.utils import MLPPredictor, RandEdgeSampler
from ptls_extension_2024_research.graphs.static_models.gnn import GraphSAGE, GAT




class ColesBatchToSubgraphConverter:
    def __init__(self, graph_file_path, item_id2graph_id_path, client_id2graph_id_path):
        self.client_item_g: ClientItemGraph = ClientItemGraph.from_graph_file(graph_file_path)
        self.item_id2graph_id = torch.load(item_id2graph_id_path)
        self.client_id2graph_id = torch.load(client_id2graph_id_path)

    def get_coles_item_ids2subgraph_item_ids(self, 
                                             subgraph_ids_to_graph_ids, 
                                             item_ids) -> torch.Tensor:
        graph_ids_to_subgraph_ids = {graph_id: subgraph_id 
                                     for subgraph_id, graph_id 
                                     in enumerate(subgraph_ids_to_graph_ids)}

        coles_item_ids2subgraph_item_ids = [graph_ids_to_subgraph_ids[self.item_id2graph_id[item_id]] 
                                            for item_id in item_ids]
        
        coles_item_ids2subgraph_item_ids = torch.LongTensor(coles_item_ids2subgraph_item_ids, 
                                                            device=item_ids.device, requires_grad=False)
        
        return coles_item_ids2subgraph_item_ids


    def __call__(self, client_ids, item_ids):
        """
        client_ids: torch.Tensor, shape: (batch_size,)
        item_ids: torch.Tensor, shape: (batch_size, seq_len)
        """
        graph_item_ids = self.item_id2graph_id[item_ids]
        graph_client_ids = self.client_id2graph_id[client_ids]
        subgraph = self.client_item_g.get_subgraph(graph_item_ids, graph_client_ids)
        subgraph_ids_to_graph_ids = subgraph.ndata['_ID']

        coles_item_ds2subgraph_item_ids = self.get_coles_item_ids2subgraph_item_ids(
            subgraph_ids_to_graph_ids, item_ids
        )
        
        result = {
            'subgraph': subgraph,
            'coles_item_ids2subgraph_item_ids': coles_item_ds2subgraph_item_ids
        }

        return result

    



class ColesBatchToSubgraphConverterFull(ColesBatchToSubgraphConverter):
    """
    A special case of ColesBatchToSubgraphConverter where
    a full graph is used as a subgraph contatining client_ids and item_ids;
    And it's guaranteed that g.ndata['_ID'] = range(n_nodes)
    """
    def __init__(self, graph_file_path, item_id2graph_id_path, client_id2graph_id_path):
        super().__init__(graph_file_path, item_id2graph_id_path, client_id2graph_id_path)

    def __call__(self, client_ids, item_ids):
        """
        client_ids: torch.Tensor, shape: (batch_size,)
        item_ids: torch.Tensor, shape: (batch_size, seq_len)
        """
        
        result = {
            'subgraph': self.client_item_g,
            'coles_item_ids2subgraph_item_ids': self.item_id2graph_id_path
        }

        return result





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
                 link_predictor_name: str='MLP',
                 link_predictor_add_sigmoid: bool=True,
                 gnn_name: str='GraphSAGE',
                 gnn_kwargs_dict = None):
        super().__init__()

        if gnn_kwargs_dict is None:
            gnn_kwargs_dict = {}

        self._output_size = output_size
        self.n_users = n_users
        self.n_items = n_items
        self.train_client_embeddings = train_client_embeddings
        self.train_item_embeddings = train_item_embeddings

        total_nodes = n_users + n_items
        self.node_feats = nn.Embedding(total_nodes, embedding_dim)
        
        # Create views for client_feats and item_feats from node_feats
        # self.client_feats = nn.Embedding.from_pretrained(self.node_feats.weight[:n_users], freeze=False)
        # self.item_feats = nn.Embedding.from_pretrained(self.node_feats.weight[n_users:], freeze=False)


        self.gnn = self._init_gnn(gnn_name, in_feats=embedding_dim, h_feats=output_size, **gnn_kwargs_dict)
        self.link_predictor = self._init_link_predictor(link_predictor_name, output_size, link_predictor_add_sigmoid)

    def _init_gnn(self, gnn_name, in_feats, h_feats, **gnn_kwags):
        if gnn_name == 'GraphSAGE':
            return GraphSAGE(in_feats=in_feats, h_feats=h_feats, **gnn_kwags)
        if gnn_name == 'GAT':
            return GAT(in_feats=in_feats, h_feats=h_feats, **gnn_kwags)
        raise Exception(f'No such graph model {gnn_name}')
    
    def _init_link_predictor(self, link_predictor_name, output_size, link_predictor_add_sigmoid):
        if link_predictor_name == 'MLP':
            return MLPPredictor(output_size, link_predictor_add_sigmoid)
        raise Exception(f'No such link predictor {link_predictor_name}')

    def forward(self, subgraph):
        subgraph_node_embeddings = self.gnn(subgraph, self.node_feats[subgraph.ndata['_ID']])
        return subgraph_node_embeddings
    



class GnnModule(pl.LightningModule):
    def __init__(self, 
                 gnn_link_predictor: GnnLinkPredictor,
                 optimizer_partial,
                 lr_scheduler_partial,
                 neg_edge_sampler,
                 neg_items_per_pos: int=1,
                 lp_criterion_name: str='BCELoss',):
        super().__init__()
        self.gnn_link_predictor = gnn_link_predictor
        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial
        self.neg_edge_sampler = neg_edge_sampler

        # loss
        self.neg_items_per_pos = neg_items_per_pos
        self.lp_criterion = getattr(nn, lp_criterion_name)()


    def calc_loss(self, g, node_embeddings):
        pos_src, pos_dst = g.edges()
        pos_scores = self.gnn_link_predictor.link_predictor(pos_src, pos_dst, node_embeddings)
        neg_src, neg_dst = self.neg_edge_sampler.sample(
            self.neg_items_per_pos * len(g.number_of_edges()))
        neg_scores = self.gnn_link_predictor.link_predictor(neg_src, neg_dst, node_embeddings)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])])
        loss = self.lp_criterion(scores, labels)
        return loss


    def training_step(self, subgraph, _):
        subgraph_node_embeddings = self.gnn_link_predictor(subgraph)
        return self.calc_loss(subgraph, subgraph_node_embeddings)
    

    def validation_step(self, subgraph, _):
        subgraph_node_embeddings = self.gnn_link_predictor(subgraph)
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

