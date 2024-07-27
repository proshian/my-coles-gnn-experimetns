import torch  # for typing
from ptls.data_load.padded_batch import PaddedBatch  # for typing
from ptls.frames.coles import CoLESModule


class CoLESModule_CITrx(CoLESModule):
    """
    Same as ptls.frames.coles.CoLESModule, except 
    TrxEncoder_WithClientItemEmbeddings is used as trx_encoder and thus 
    it takes a tuple:
    (`padded_batch_of_dict_with_seq_feats`, `client_ids`)
    instead of just `padded_batch_of_dict_with_seq_feats`
    """
    def shared_step(self, x: PaddedBatch, client_ids: torch.Tensor):
        y_h = self((x, client_ids))
        if self._head is not None:
            y_h = self._head(y_h)
        return y_h, client_ids

    # temp solution to avoid logging outside callbacks in coles_gnn_modules
    def training_step(self, batch, _):
        y_h, y = self.shared_step(*batch)
        loss = self._loss(y_h, y)
        # self.log('loss', loss)
        if type(batch) is tuple:
            x, y = batch
            # if isinstance(x, PaddedBatch):
            #     self.log('seq_len', x.seq_lens.float().mean(), prog_bar=True)
        else:
            # this code should not be reached
            # self.log('seq_len', -1, prog_bar=True)
            raise AssertionError('batch is not a tuple')
        return loss
