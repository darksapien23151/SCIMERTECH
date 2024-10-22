import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from rdkit import Chem
from sklearn.model_selection import train_test_split
def evaluate(model, loader, targets):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for i, batch in enumerate(loader):
            spectra = torch.stack([targets[j] for j in range(i * loader.batch_size, min((i + 1) * loader.batch_size, len(targets)))])
            output = model(batch)
            loss = criterion(output, spectra)
            val_loss += loss.item()
        print(f'Validation Loss: {val_loss / len(loader)}')

# Evaluate the model
evaluate(model, val_loader, val_targets)
