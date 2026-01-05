import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

class BasicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, device='cpu'):
        super(BasicMLP, self).__init__()
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.to(self.device)

    def forward(self, x):
        return self.network(x)
    
    def train_classifer(self, train_loader, test_loader, epoch_checkpoints):
        """
        Args:
            train_loader: DataLoader for training
            test_loader: DataLoader for testing
            epoch_checkpoints: List of epochs at which to record accuracy (e.g. [5, 10])
        """
        results = {} 
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        max_epoch = max(epoch_checkpoints)
        
        logger.info(f"Starting training for {max_epoch} epochs on {self.device}...")
        
        self.train()
        for epoch in range(1, max_epoch + 1):
            
            # Training Loop
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = self(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            # Evaluation Loop
            if epoch in epoch_checkpoints:
                self.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        logits = self(batch_X)
                        predicted_probs = torch.sigmoid(logits)
                        
                        predictions = (predicted_probs > 0.5).float()
                        correct += (predictions == batch_y).sum().item()
                        total += batch_y.size(0)
                
                acc = correct / total if total > 0 else 0
                results[epoch] = acc
                logger.info(f"Epoch {epoch}: Accuracy = {acc:.4f}")
                
                self.train() # Switch back to train mode
                
        return results