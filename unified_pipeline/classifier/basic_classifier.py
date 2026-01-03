import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

# 1. Define the MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        logger.info("Initilizing Simple MLP Classifier")
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Helps prevent overfitting on specific neurons
            
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            # Output Layer (No Sigmoid here!)
            nn.Linear(hidden_dim // 2, 1) 
        )

    def forward(self, x):
        return self.network(x)
