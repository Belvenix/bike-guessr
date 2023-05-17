import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO)

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
def train_trivial_model(model, features, y, epochs, batch_size) -> nn.Module:
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    num_samples = features.size(0)
    for epoch in range(epochs):
        # Set the model to training mode
        epoch_loss = 0.0

        # Zero the gradients
        optimizer.zero_grad()
        
        # Randomly shuffle the data
        indices = torch.randperm(num_samples)
        features = features[indices]
        y = y[indices]
        
        for i in range(0, num_samples, batch_size):
            # Extract the current batch
            inputs = features[i:i+batch_size]
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
        # f1 = f1_score(labels, torch.argmax(model(features)), average='micro')

        # Print the average loss for the epoch
        #logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / (num_samples / batch_size)}, F1: {f1:.3f}")
    return model

def train_gnn_model(model, g, epochs) -> nn.Module:
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Define features and y
    features = g.ndata['feat']
    labels = g.ndata['label']
    num_samples = features.size(0)
    for epoch in range(epochs):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()  # Set the model to evaluation mode
        f1 = f1_score(labels, pred, average='micro')

        # Print the average loss for the epoch
        #logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item() / (num_samples)}, F1: {f1:.3f}")
    return model
