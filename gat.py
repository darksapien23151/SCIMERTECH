import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from rdkit import Chem
from sklearn.model_selection import train_test_split


class GATNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, out_dim, heads=1, concat=False)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x

# Initialize model
in_dim = 1
hidden_dim = 8
out_dim = len(spectral_targets[0])  
num_heads = 4

model = GATNet(in_dim, hidden_dim, out_dim, num_heads)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
