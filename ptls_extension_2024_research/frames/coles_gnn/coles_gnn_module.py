import pytorch_lightning as pl
from ptls.frames.abs_module import ABSModule
from ptls.frames.coles.losses import ContrastiveLoss
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from ptls.nn.head import Head
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls_extension_2024_research import TrxEncoder_WithCIEmbeddings
from ptls_extension_2024_research.nn.trx_encoder.client_item_encoder import GNNClientItemEncoder


class ColesGnnModule(pl.LightningModule):
    """
    """
    def __init__(self,
                 seq_encoder: SeqEncoderContainer = None,
                 head=None,
                 coles_loss=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None):
        if head is None:
            head = Head(use_norm_encoder=True)

        if coles_loss is None:
            coles_loss = ContrastiveLoss(
                margin=0.5, sampling_strategy=HardNegativePairSelector(neg_count=5))

        if validation_metric is None:
            validation_metric = BatchRecallTopK(K=4, metric='cosine')

        super().__init__(validation_metric,
                         seq_encoder,
                         coles_loss,
                         optimizer_partial,
                         lr_scheduler_partial)

        self.gnn = self.get_gnn_from_seq_encoder(seq_encoder)
        self._head = head

    
    def get_gnn_from_seq_encoder(self, seq_encoder):
        trx_encoder = seq_encoder.seq_encoder.trx_encoder
        assert isinstance(trx_encoder, TrxEncoder_WithCIEmbeddings), f"Unexpected trx_encoder type: {type(trx_encoder)}"
        gnns_ci_embedders = [embedder for embedder in trx_encoder.client_item_embeddings if isinstance(embedder, GNNClientItemEncoder)]
        assert len(gnns_ci_embedders) == 1, f"Unexpected number of GNNClientItemEncoder: {len(gnns_ci_embedders)}"
        return gnns_ci_embedders[0].gnn

    






"""
* shared_step - общий шаг для обучения и валидации: принимает фичи и user_ids, возвращает user_embeddings и user_ids
* forward - принимает фичи, возвращает user_embeddings
* training_step - шаг обучения: принимает батч и индекс батча, возвращает лосс
* validation_step - шаг валидации: принимает батч и индекс батча, вычисляет мтерики, ничего не возвращает
* on_validation_epoch_end - логирует метрики, вычисленные на валидации
* configure_optimizers - конфигурирует оптимизаторы и lr_scheduler'ы
"""


class ABSModule(pl.LightningModule):
    @property
    def metric_name(self):
        raise NotImplementedError()

    @property
    def is_requires_reduced_sequence(self):
        raise NotImplementedError()

    def shared_step(self, x, y):
        """

        Args:
            x:
            y:

        Returns: y_h, y

        """
        raise NotImplementedError()

    def __init__(self, validation_metric=None,
                       seq_encoder=None,
                       loss=None,
                       optimizer_partial=None,
                       lr_scheduler_partial=None):
        """
        Parameters
        ----------
        params : dict
            params for creating an encoder
        seq_encoder : torch.nn.Module
            sequence encoder, if not provided, will be constructed from params
        """
        super().__init__()
        # self.save_hyperparameters()

        self._loss = loss
        self._seq_encoder = seq_encoder
        self._seq_encoder.is_reduce_sequence = self.is_requires_reduced_sequence
        self._validation_metric = validation_metric

        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial

    @property
    def seq_encoder(self):
        return self._seq_encoder

    def forward(self, x):
        return self._seq_encoder(x)

    def training_step(self, batch, _):
        y_h, y = self.shared_step(*batch)
        loss = self._loss(y_h, y)
        self.log('loss', loss)
        if type(batch) is tuple:
            x, y = batch
            if isinstance(x, PaddedBatch):
                self.log('seq_len', x.seq_lens.float().mean(), prog_bar=True)
        else:
            # this code should not be reached
            self.log('seq_len', -1, prog_bar=True)
            raise AssertionError('batch is not a tuple')
        return loss

    def validation_step(self, batch, _):
        y_h, y = self.shared_step(*batch)
        self._validation_metric(y_h, y)

    def on_validation_epoch_end(self):
        self.log(f'valid/{self.metric_name}', self._validation_metric.compute(), prog_bar=True)
        self._validation_metric.reset()

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self.parameters())
        scheduler = self._lr_scheduler_partial(optimizer)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                'scheduler': scheduler,
                'monitor': self.metric_name,
            }
        return [optimizer], [scheduler]
