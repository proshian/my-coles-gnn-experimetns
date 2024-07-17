from dataclasses import dataclass
import dgl
import torch


@dataclass
class ClientItemGraph:
    g: dgl.DGLGraph

    def create_subgraph(self, client_ids: torch.Tensor, item_ids: torch.Tensor):
        item_ids = torch.flatten(item_ids)

        # Subset of nodes
        nodes_of_interest = torch.cat([client_ids, item_ids])

        # Find all neighbors by using the predecessors and successors
        in_neighbors = self.g.in_edges(nodes_of_interest)[0].unique()
        out_neighbors = self.g.out_edges(nodes_of_interest)[1].unique()

        # Combine the nodes of interest with their in-neighbors and out-neighbors
        all_nodes = torch.cat([nodes_of_interest, in_neighbors, out_neighbors]).unique()

        # Induce a subgraph with all the relevant nodes
        subgraph = dgl.node_subgraph(g, all_nodes)

        return subgraph

    @classmethod
    def from_graph_file(cls, graph_file_path: str):
        g_list, _ = dgl.load_graphs(graph_file_path, [0])
        g = g_list[0]
        return cls(g)

