import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from rdkit import Chem

def mol_to_graph(smiles, atomic_counts, spectra_data):
    mol = Chem.MolFromSmiles(smiles)
    
    num_atoms = mol.GetNumAtoms()
    
    # Node features
    node_feats = torch.tensor(atomic_counts, dtype=torch.float).unsqueeze(-1) 
    
    # (bonds)
    edge_index = []
    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        edge_index.append([src, dst])
        edge_index.append([dst, src]) 
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  

    pyg_graph = Data(x=node_feats, edge_index=edge_index)

    return pyg_graph, torch.tensor(spectra_data, dtype=torch.float32)
