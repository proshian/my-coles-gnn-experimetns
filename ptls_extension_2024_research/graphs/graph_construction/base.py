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

    def build(self, df, src_col, dst_col, use_weights):
        if use_weights:
            df, src_col, dst_col, weight_col = self._build_weighted_edge_df(df, src_col, dst_col)
        else:
            df, src_col, dst_col = self._build_simple_edge_df(df, src_col, dst_col)
            weight_col = None

        g, node2index, index2node = create_graph_from_df(df, src_col, dst_col, weight_col)
        return g, node2index, index2node


def create_graph_from_df(df, src_col: str, dst_col: str, weight_col: Optional[str] = None):
    # Create a dictionary to map node names to integers
    unique_nodes_src = df[src_col].unique()
    unique_nodes_dst = df[dst_col].unique()
    unique_nodes = np.concatenate([unique_nodes_src, unique_nodes_dst])
    node_map = {node: i for i, node in enumerate(unique_nodes)}

    # Create reverse mapping (integer to node name)
    reverse_node_map = {i: node for node, i in node_map.items()}

    # Convert source and destination columns to integer indices
    src = torch.tensor([node_map[node] for node in df[src_col]])
    dst = torch.tensor([node_map[node] for node in df[dst_col]])

    src_bi = torch.cat([src, dst])
    dst_bi = torch.cat([dst, src])

    # Create the graph
    g = dgl.graph((src_bi, dst_bi))

    # Add edge weights
    weights = torch.tensor(df[weight_col].values, dtype=torch.float32)
    weights_bi = torch.cat([weights, weights])
    g.edata['weight'] = weights_bi

    # Add node names as a node feature
    #      g.ndata['name'] = torch.tensor([reverse_node_map[i] for i in range(g.num_nodes())])

    return g, node_map, reverse_node_map
