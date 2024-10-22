import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from rdkit import Chem
from sklearn.model_selection import train_test_split

def train(model, loader, targets, epochs=100):
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            
            spectra = torch.stack([targets[j] for j in range(i * loader.batch_size, min((i + 1) * loader.batch_size, len(targets)))])
            
            # Forward pass
            output = model(batch)
            loss = criterion(output, spectra) 
            loss.backward()  
            optimizer.step()  
            
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(loader)}')

# Train the model
train(model, train_loader, train_targets, epochs=100)
