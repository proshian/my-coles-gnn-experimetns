import dgl
import numpy as np
import torch
from torch import nn


class RandEdgeSampler:
    def __init__(self, train_graph: dgl.DGLGraph, seed=None):
        self.seed = None
        src, dst = train_graph.edges()
        self.src_list = np.unique(src.numpy()).astype(int)
        self.dst_list = np.unique(dst.numpy()).astype(int)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


class MLPPredictor(nn.Module):
    def __init__(self, h_feats, add_sigmoid=True):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
        self.act = nn.ReLU()
        self.add_sigmoid = add_sigmoid

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(self.act(self.W1(h)))}
        # return {'score': self.sigmoid(self.W2(F.relu(self.W1(h)))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            if not self.add_sigmoid:
                return g.edata['score']
            return g.edata['score'].sigmoid()


def create_subgraph_with_all_neighbors(graph: dgl.DGLGraph, node_ids: torch.Tensor):
    # Find all neighbors by using the predecessors and successors
    in_neighbors = graph.in_edges(node_ids)[0].unique()
    out_neighbors = graph.out_edges(node_ids)[1].unique()

    # Combine the nodes of interest with their in-neighbors and out-neighbors
    all_nodes = torch.cat([node_ids, in_neighbors, out_neighbors]).unique()
    all_nodes, _ = torch.sort(all_nodes)


    # Induce a subgraph with all the relevant nodes
    subgraph = dgl.node_subgraph(graph, nodes=all_nodes)
    return subgraph
