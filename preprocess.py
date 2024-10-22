import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from rdkit import Chem

# Load train, test, and val CSVs
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
val_df = pd.read_csv('val.csv')

def preprocess_data(df):
    graphs = []
    spectral_targets = []
    
    for idx, row in df.iterrows():
        smiles = row[0] 
        atomic_counts = eval(row[1]) 
        spectra_data = eval(row[2]) 
        
        graph, spectra = mol_to_graph(smiles, atomic_counts, spectra_data)
        graphs.append(graph)
        spectral_targets.append(spectra)
    
    return graphs, spectral_targets

train_graphs, train_targets = preprocess_data(train_df)
val_graphs, val_targets = preprocess_data(val_df)

train_loader = DataLoader(train_graphs, batch_size=2, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=2, shuffle=False)
