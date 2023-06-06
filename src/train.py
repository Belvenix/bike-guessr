import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TENSORBOARD_LOG_DIR
from dgl.nn import GraphConv
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(TENSORBOARD_LOG_DIR)

class TrivialClassifier(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(TrivialClassifier, self).__init__()
        self.fc1 = nn.Linear(in_feats, h_feats)
        self.act1 = nn.ReLU()
        self.fc3 = nn.Linear(h_feats, num_classes)

    def forward(self, x):
        out = self.act1(self.fc1(x))
        out = self.fc3(out)
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


def train_trivial_model(model, g, epochs, model_name) -> nn.Module:
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Define features and y
    inputs = g.ndata['feat']
    labels = g.ndata['label']
    num_samples = inputs.size(0)

    # Randomly shuffle the data
    indices = torch.randperm(num_samples)
    inputs = inputs[indices]
    labels = labels[indices]

    # Train the model
    for epoch in range(epochs):
        model.train()  # Set the model to training mode

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = criterion(outputs, labels)
        loss_item = loss.item()

        # Backward
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()

        # Compute prediction
        pred = outputs.argmax(1)
        f1 = f1_score(labels, pred, average='micro')

        # Print the average loss for the epoch
        logging.debug(f"Epoch {epoch+1}/{epochs}, Loss: {loss_item}, F1: {f1:.3f}")
        writer.add_scalar(f'{model_name}/Loss/train', loss_item, epoch)
    writer.flush()
    return model

def train_gnn_model(model, g, epochs, model_name) -> nn.Module:
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Define features and y
    features = g.ndata['feat']
    labels = g.ndata['label']
    num_samples = features.size(0)

    # Train the model
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        
        # Zero the gradients
        optimizer.zero_grad()

        # Forward
        logits = model(g, features)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = criterion(logits, labels)
        loss_item = loss.item()

        # Backward
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        
        # Compute prediction
        pred = logits.argmax(1)
        f1 = f1_score(labels, pred, average='micro')

        # Print the average loss for the epoch
        logging.debug(f"Epoch {epoch+1}/{epochs}, Loss: {loss_item}, F1: {f1:.3f}")
        writer.add_scalar(f'{model_name}/Loss/train', loss_item, epoch)
    writer.flush()
    return model
