import numpy as np
import torch
import dgl
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class GraphBuilder(ABC):
    @abstractmethod
    def _build_weighted_edge_df(self, df, src_col, dst_col):
        pass

    @abstractmethod
    def _build_simple_edge_df(self, df, src_col, dst_col):
        pass

    def build(self, df, client_col, item_col, use_weights):
        if use_weights:
            df, client_col, item_col, weight_col = self._build_weighted_edge_df(df, client_col, item_col)
        else:
            df, client_col, item_col = self._build_simple_edge_df(df, client_col, item_col)
            weight_col = None

        g, client_id2graph_id, item_id2graph_id, items_cnt = create_graph_from_df(df, client_col, item_col, weight_col)
        return g, client_id2graph_id, item_id2graph_id, items_cnt


def create_graph_from_df(df, client_col: str, item_col: str, weight_col: Optional[str] = None):
    # Create a dictionary to map node names to integers
    unique_nodes_client = np.sort(df[client_col].unique().astype(int))
    unique_nodes_item = np.sort(df[item_col].unique().astype(int))

    # create index mapping
    client_id2graph_id = torch.full(size = (unique_nodes_client.max()+1,), fill_value=-1, dtype=torch.long)
    client_id2graph_id[unique_nodes_client] = torch.arange(len(unique_nodes_client))

    # items always follow user index
    item_id2graph_id = torch.full(size = (unique_nodes_item.max() + 1,), fill_value=-1, dtype=torch.long)
    item_id2graph_id[unique_nodes_item] = torch.arange(len(unique_nodes_item)) + len(unique_nodes_client)

    print(item_id2graph_id)

    # Convert source and destination columns to integer indices
    src = client_id2graph_id[df[client_col].values]
    dst = item_id2graph_id[df[item_col].values]

    src_bi = torch.cat([src, dst])
    dst_bi = torch.cat([dst, src])

    # Create the graph
    g = dgl.graph((src_bi, dst_bi))

    # Add edge weights
    if weight_col is not None:
        weights = torch.tensor(df[weight_col].values, dtype=torch.float32)
        weights_bi = torch.cat([weights, weights])
        g.edata['weight'] = weights_bi

    return g, client_id2graph_id, item_id2graph_id, len(unique_nodes_item)