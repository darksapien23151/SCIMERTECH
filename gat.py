import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import dgl

class GAT(nn.Module):
    def __init__(self,
                 node_input_dim=36,
                 edge_input_dim=6,
                 node_hidden_dim=36,
                 num_heads=3,
                 num_layers=3):
        super(GAT, self).__init__()
        self.num_layers = num_layers

        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)

        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(in_feats=node_hidden_dim,
                        out_feats=node_hidden_dim // num_heads,
                        num_heads=num_heads,
                        feat_drop=0.1,
                        attn_drop=0.1,
                        residual=True,
                        activation=F.elu)
            )

        self.output_layer = nn.Linear(node_hidden_dim, node_hidden_dim)

    def forward(self, g, node_features):
        h = self.lin0(node_features)

        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)

        h = self.output_layer(h)
        return h
