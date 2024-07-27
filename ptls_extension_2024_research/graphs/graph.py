from dataclasses import dataclass
import dgl
import torch


@dataclass
class ClientItemGraph:
    """
    A class to represent a graph with clients and items.
    Given a list of client_ids and item_ids, it creates 
    a subgraph with the clients, items, and their neighbors.
    """
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

        all_nodes = torch.sort(all_nodes)

        # Induce a subgraph with all the relevant nodes
        subgraph = dgl.node_subgraph(self.g, all_nodes)

        return subgraph.to(torch.device('cuda'))

    @classmethod
    def from_graph_file(cls, graph_file_path: str):
        g_list, _ = dgl.load_graphs(graph_file_path, [0])
        g = g_list[0].cuda()
        return cls(g)


@dataclass
class ClientItemGraphFull:
    """
    A special case of the ClientItemGraph where the subgraph 
    is the same as the original full graph.
    """
    g: dgl.DGLGraph
    device_name: str
    
    def __post_init__ (self):
        device = torch.device(self.device_name)
        self.g.ndata['_ID'] = torch.arange(0, self.g.number_of_nodes(), device=device)

    def create_subgraph(self, client_ids: torch.Tensor, item_ids: torch.Tensor):
        return self.g

    @classmethod
    def from_graph_file(cls, graph_file_path: str, device_name: str):
        g_list, _ = dgl.load_graphs(graph_file_path, [0])
        g = g_list[0].to(torch.device(device_name))
        return cls(g, device_name)
