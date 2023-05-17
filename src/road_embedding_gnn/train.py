import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from sklearn.metrics import f1_score


class TrivialClassifier(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(TrivialClassifier, self).__init__()
        self.fc1 = nn.Linear(in_feats, h_feats)
        self.act1 = nn.ReLU()
        self.fc3 = nn.Linear(h_feats, num_classes)
        self.act3 = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.act1(self.fc1(x))
        out = self.act3(self.fc3(out))
        return out
    
class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphConvolutionalNetwork, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


# Define the training loop
def train_model(model, X, y, epochs, batch_size) -> nn.Module:
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    num_samples = X.size(0)
    for epoch in range(epochs):
        # Set the model to training mode
        epoch_loss = 0.0

        # Zero the gradients
        optimizer.zero_grad()
        
        # Randomly shuffle the data
        indices = torch.randperm(num_samples)
        X = X[indices]
        y = y[indices]
        
        for i in range(0, num_samples, batch_size):
            # Extract the current batch
            inputs = X[i:i+batch_size]
            labels = y[i:i+batch_size]

            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
            # Accumulate the loss for the epoch
            epoch_loss += loss.item()
        
        model.eval()  # Set the model to evaluation mode
        f1 = f1_score(y, torch.argmax(model(X), dim=1), average='micro')

        # Print the average loss for the epoch
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / (num_samples / batch_size)}, F1: {f1:.3f}")
    return model
