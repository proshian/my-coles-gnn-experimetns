# ptls extensions used in the 2024 research project

# Main components:
## TrxEncoder_WithCIEmbeddings
* location: `nn.trx_encoder.trx_encoder_with_client_item_embeddings.TrxEncoder_WithCIEmbeddings`

Extends `TrxEncoder` from ptls by adding client-item embeddings. Those are obtained from an instance of `BaseClientItemEncoder` by providing it with a client_ids tensor of shape `(batch_size,)` and a item_ids tensor of shape `(batch_size, seq_len)`. 

Another technical difference from `TrxEncoder` is that input is not just `PaddedBatch` of sequential transaction features but also a `torch.Tensor` of client_ids. 

In our research client-item embeddings are GNN embeddings of edges and items in a user-item graph.

## AvgPoolLinearSeqEncoder
* location: `nn.seq_encoder.containers.AvgPoolLinearSeqEncoder`

A seq_encoder that averages the embeddings of the sequence and passes it through a linear layer.

## CoLESDataset
* location: `frames.coles_client_id_aware.coles_dataset_real_client_ids

Same as `ColesDataset` from ptls, but 
* Collate_fn returns REAL client_ids instead of 
    just different integers for different clients retirieved via enumerate.
* An i-th dataset element contains not only n dicts representing 
    splits of sequential features, but also an id. This is required
    to get real ids in collate_fn. Even if we don't take client_ids 
    from the table but consider index in a dataset as an id, we can't get
    theese indexes in `collate_fn` without making them a part of a dataset element.

## CoLESModule_CITrx
* location: `frames.coles_client_id_aware.coles_module__trx_with_ci_embs.CoLESModule_CITrx`

Similar to as `CoLESModule` from ptls, except it's expected that our version of `ColesDataset` is used and trx_encoder is our `TrxEncoder_WithCIEmbeddings` instead of `TrxEncoder`. This means that real client_ids should be provided in the dataset and trx_encoder should be able to create client-item embeddings.
