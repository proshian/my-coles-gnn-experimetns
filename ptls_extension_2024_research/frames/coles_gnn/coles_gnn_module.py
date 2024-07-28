import torch
import pytorch_lightning as pl
from ptls.frames.abs_module import ABSModule
from ptls.frames.coles.losses import ContrastiveLoss
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from ptls.nn.head import Head
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from ptls_extension_2024_research import TrxEncoder_WithCIEmbeddings
from ptls_extension_2024_research.nn.trx_encoder.client_item_encoder import StaticGNNTrainableClientItemEncoder
from ptls_extension_2024_research.frames.coles_client_id_aware.coles_module__trx_with_ci_embs import CoLESModule_CITrx
from ptls_extension_2024_research.frames.gnn.gnn_module import GnnModule
from ptls_extension_2024_research.graphs.utils import RandEdgeSamplerFull
from ptls_extension_2024_research.lightning_utlis import LogLstEl


def get_ci_embedder_from_seq_encoder(seq_encoder):
    trx_encoder = seq_encoder.trx_encoder
    assert isinstance(trx_encoder, TrxEncoder_WithCIEmbeddings), f"Unexpected trx_encoder type: {type(trx_encoder)}"
    gnns_ci_embedders = [embedder for embedder in trx_encoder.client_item_embeddings if isinstance(embedder, StaticGNNTrainableClientItemEncoder)]
    assert len(gnns_ci_embedders) == 1, f"Unexpected number of GNNClientItemEncoder instances: {len(gnns_ci_embedders)}"
    return gnns_ci_embedders[0]

class ColesGnnModule(pl.LightningModule):
    """
    Arguments:
    ----------
    neg_edge_sampler is an object of an edge sampler calss
    """
    def __init__(self,
                 seq_encoder: SeqEncoderContainer = None,
                 coles_head=None,
                 coles_loss=None,
                 coles_validation_metric=None,
                 neg_items_per_pos = 1,
                 lr_criterion_name = 'BCELoss',
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 neg_edge_sampler=None):
        super().__init__()

        gnn = self.get_gnn_from_seq_encoder(seq_encoder)
        self.coles_module = CoLESModule_CITrx(seq_encoder, coles_head, coles_loss, coles_validation_metric, 
                                           optimizer_partial=None, lr_scheduler_partial=None)
        self.gnn_module = GnnModule(gnn, optimizer_partial=None, lr_scheduler_partial=None, 
                                    neg_edge_sampler=neg_edge_sampler,
                                    neg_items_per_pos = neg_items_per_pos, 
                                    lr_criterion_name = lr_criterion_name)
        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial

    
    def get_ci_embedder_from_seq_encoder(self, seq_encoder):
        return get_ci_embedder_from_seq_encoder(seq_encoder)

    def get_gnn_from_seq_encoder(self, seq_encoder):
        return self.get_ci_embedder_from_seq_encoder(seq_encoder).gnn_link_predictor

    # def forward(self, x):
    #     pass

    def training_step(self, batch, _):
        subgraph = self.get_subgraph(batch)
        gnn_loss = self.gnn_module.training_step(subgraph, _)
        coles_loss = self.coles_module.training_step(batch, _)
        full_loss = gnn_loss + coles_loss
        return full_loss
        

    def validation_step(self, batch, _):
        self.coles_module.validation_step(batch, _)
        self.gnn_module.validation_step(batch, _) 

    def on_validation_epoch_end(self):
        self.coles_module.on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self.parameters())
        scheduler = self._lr_scheduler_partial(optimizer)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                'scheduler': scheduler,
                'monitor': self.metric_name,
            }
        return [optimizer], [scheduler]


    





class ColesGnnModuleFullGraph(pl.LightningModule):
    """
    A special case of ColesGnnModule where RandEdgeSamplerFull is used
    """
    def __init__(self,
                seq_encoder: SeqEncoderContainer = None,
                coles_head=None,
                coles_loss=None,
                coles_validation_metric=None,
                rand_edge_sampler_seed = None,
                neg_items_per_pos = 1,
                lp_criterion_name = 'BCELoss',
                optimizer_partial=None,
                lr_scheduler_partial=None) -> None:
        super().__init__()  # calling "GrandParent's" init
        self.coles_module = CoLESModule_CITrx(seq_encoder, coles_head, coles_loss, coles_validation_metric, 
                                           optimizer_partial=None, lr_scheduler_partial=None)

        ci_embedder = self.get_ci_embedder_from_seq_encoder(seq_encoder)
        gnn = ci_embedder.gnn_link_predictor
        self.client_item_g = ci_embedder.data_adapter.client_item_g
        rand_edge_sampler = RandEdgeSamplerFull(
            self.client_item_g.g, rand_edge_sampler_seed)
        
        self.gnn_module = GnnModule(gnn, optimizer_partial=None, lr_scheduler_partial=None, 
                                    neg_edge_sampler=rand_edge_sampler,
                                    neg_items_per_pos = neg_items_per_pos, 
                                    lp_criterion_name = lp_criterion_name)
        
        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial

    def get_ci_embedder_from_seq_encoder(self, seq_encoder):
        return get_ci_embedder_from_seq_encoder(seq_encoder)

    def get_gnn_from_seq_encoder(self, seq_encoder):
        return self.get_ci_embedder_from_seq_encoder(seq_encoder).gnn_link_predictor

    # def forward(self, x):
    #     pass

    def training_step(self, batch, _):
        # Достать из батча client_ids и item_ids
        # В данном частном случае можно передавать что угодно, но в общем случае нужно решить этот оврпос
        subgraph = self.client_item_g.create_subgraph(_, _)
        gnn_loss = self.gnn_module.training_step(subgraph, _)
        coles_loss, log_list = self.coles_module._training_step(batch, _)
        full_loss = gnn_loss + coles_loss

        for el in log_list:
            self.log(f"coles/{el.name}", el.value, *el.args, **el.kwargs)
        return full_loss
        

    def validation_step(self, batch, _):
        # Достать из батча client_ids и item_ids
        # В данном частном случае можно передавать что угодно, но в общем случае нужно решить этот оврпос
        subgraph = self.client_item_g.create_subgraph(_, _)
        self.coles_module.validation_step(batch, _)
        self.gnn_module.validation_step(subgraph, _) 

    def _on_validation_epoch_end__coles(self):
        self.log(f'coles/valid/{self.coles_module.metric_name}', self.coles_module._validation_metric.compute(), prog_bar=True)
        self.coles_module._validation_metric.reset()

    def on_validation_epoch_end(self):
        self._on_validation_epoch_end__coles()

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self.parameters())
        scheduler = self._lr_scheduler_partial(optimizer)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                'scheduler': scheduler,
                'monitor': self.metric_name,
            }
        return [optimizer], [scheduler]




# class ColesGnnModuleFullGraph(ColesGnnModule):
#     """
#     A special case of ColesGnnModule where RandEdgeSamplerFull is used
#     """
#     def __init__(self,
#                 seq_encoder: SeqEncoderContainer = None,
#                 coles_head=None,
#                 coles_loss=None,
#                 coles_validation_metric=None,
#                 rand_edge_sampler_seed = None,
#                 neg_items_per_pos = 1,
#                 lp_criterion_name = 'BCELoss',
#                 optimizer_partial=None,
#                 lr_scheduler_partial=None) -> None:
#         super(pl.LightningModule, self).__init__()  # calling "GrandParent's" init
#         self.coles_module = CoLESModule_CITrx(seq_encoder, coles_head, coles_loss, coles_validation_metric, 
#                                            optimizer_partial=None, lr_scheduler_partial=None)

#         ci_embedder = self.get_ci_embedder_from_seq_encoder(seq_encoder)
#         gnn = ci_embedder.gnn_link_predictor
        
#         graph = ci_embedder.data_adapter.client_item_g.g
#         rand_edge_sampler = RandEdgeSamplerFull(
#             graph, rand_edge_sampler_seed)
        
#         self.gnn_module = GnnModule(gnn, optimizer_partial=None, lr_scheduler_partial=None, 
#                                     neg_edge_sampler=rand_edge_sampler,
#                                     neg_items_per_pos = neg_items_per_pos, 
#                                     lp_criterion_name = lp_criterion_name)
        
#         self._optimizer_partial = optimizer_partial
#         self._lr_scheduler_partial = lr_scheduler_partial





"""
* shared_step - общий шаг для обучения и валидации: принимает фичи и user_ids, возвращает user_embeddings и user_ids
* forward - принимает фичи, возвращает user_embeddings
* training_step - шаг обучения: принимает батч и индекс батча, возвращает лосс
* validation_step - шаг валидации: принимает батч и индекс батча, вычисляет мтерики, ничего не возвращает
* on_validation_epoch_end - логирует метрики, вычисленные на валидации
* configure_optimizers - конфигурирует оптимизаторы и lr_scheduler'ы
"""






# class ABSModule(pl.LightningModule):
#     @property
#     def metric_name(self):
#         raise NotImplementedError()

#     @property
#     def is_requires_reduced_sequence(self):
#         raise NotImplementedError()

#     def shared_step(self, x, y):
#         """

#         Args:
#             x:
#             y:

#         Returns: y_h, y

#         """
#         raise NotImplementedError()

#     def __init__(self, validation_metric=None,
#                        seq_encoder=None,
#                        loss=None,
#                        optimizer_partial=None,
#                        lr_scheduler_partial=None):
#         """
#         Parameters
#         ----------
#         params : dict
#             params for creating an encoder
#         seq_encoder : torch.nn.Module
#             sequence encoder, if not provided, will be constructed from params
#         """
#         super().__init__()
#         # self.save_hyperparameters()

#         self._loss = loss
#         self._seq_encoder = seq_encoder
#         self._seq_encoder.is_reduce_sequence = self.is_requires_reduced_sequence
#         self._validation_metric = validation_metric

#         self._optimizer_partial = optimizer_partial
#         self._lr_scheduler_partial = lr_scheduler_partial

#     @property
#     def seq_encoder(self):
#         return self._seq_encoder

#     def forward(self, x):
#         return self._seq_encoder(x)

#     def training_step(self, batch, _):
#         y_h, y = self.shared_step(*batch)
#         loss = self._loss(y_h, y)
#         self.log('loss', loss)
#         if type(batch) is tuple:
#             x, y = batch
#             if isinstance(x, PaddedBatch):
#                 self.log('seq_len', x.seq_lens.float().mean(), prog_bar=True)
#         else:
#             # this code should not be reached
#             self.log('seq_len', -1, prog_bar=True)
#             raise AssertionError('batch is not a tuple')
#         return loss

#     def validation_step(self, batch, _):
#         y_h, y = self.shared_step(*batch)
#         self._validation_metric(y_h, y)

#     def on_validation_epoch_end(self):
#         self.log(f'valid/{self.metric_name}', self._validation_metric.compute(), prog_bar=True)
#         self._validation_metric.reset()

#     def configure_optimizers(self):
#         optimizer = self._optimizer_partial(self.parameters())
#         scheduler = self._lr_scheduler_partial(optimizer)
        
#         if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#             scheduler = {
#                 'scheduler': scheduler,
#                 'monitor': self.metric_name,
#             }
#         return [optimizer], [scheduler]
