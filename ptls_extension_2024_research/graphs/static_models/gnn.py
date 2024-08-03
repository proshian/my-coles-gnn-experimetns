from dgl.nn import SAGEConv, GATConv
import torch
import torch.nn as nn

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, use_edge_weights, num_layers):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, h_feats, 'mean'))
        self.use_edge_weights = use_edge_weights
        for _ in range(num_layers):
            self.layers.append(SAGEConv(h_feats, h_feats, 'mean'))

    def forward(self, g, in_feat, edge_weights):
        h = in_feat
        for layer in self.layers[:-1]:
            h = layer(g, h, edge_weight=edge_weights)
            h = torch.relu(h)
        h = self.layers[-1](g, h, edge_weights)
        return h


class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, use_edge_weights, num_heads, num_layers,
                 feat_drop=0.6, attn_drop=0.6):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        assert h_feats % num_heads == 0
        self.layer_hfeats = h_feats // num_heads
        self.use_edge_weights = use_edge_weights
        self.layers.append(GATConv(in_feats, self.layer_hfeats, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, activation=nn.ELU()))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(h_feats, self.layer_hfeats, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, activation=nn.ELU()))

    def forward(self, g, in_feat, edge_weights=None):
        h = in_feat
        for layer in self.layers[:-1]:
            h = layer(g, h, edge_weight=edge_weights)
            h = h.flatten(1)
        h = self.layers[-1](g, h, edge_weight=edge_weights)
        h = h.flatten(1)
        return h
        # return h.mean(1)
