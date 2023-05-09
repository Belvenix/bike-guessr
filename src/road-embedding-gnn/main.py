
import torch
from .utils import build_model, load_best_configs, build_args
from sklearn.metrics import f1_score
import networkx as nx
import dgl


with open('./gae_gat_128_32_2_0.001_01_28_14_05_21.bin', 'rb') as f:
    args = build_args()
    args = load_best_configs(args, "configs.yml")
    encoder = build_model(args, 'gae')
    encoder.load_state_dict(torch.load(f))


wroclaw_graph = nx.readwrite.graphml.read_graphml('./Wroclaw.xml')

print(len(wroclaw_graph.nodes))

wroclaw_transformed = dgl.load_graphs('./wrocek.bin')[0][0]

print(wroclaw_transformed)

g, X, y = wroclaw_transformed, wroclaw_transformed.ndata['feat'], wroclaw_transformed.ndata['label']
X = encoder.encode(g, X)

print(X)

print(X.shape)

X,y = X.detach(), y.detach()

import torch.nn as nn
import torch.nn.functional as F


# Define the classifier model
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_layer_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layer_dim)
        self.act1 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_layer_dim, output_dim)
        self.act3 = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.act1(self.fc1(x))
        out = self.act3(self.fc3(out))
        return out

# Define the dimensions
input_dim = 32
output_dim = 2
hidden_dim = 128

# Create an instance of the classifier model
model = Classifier(input_dim, hidden_dim, output_dim)



# Define the training loop
def train_model(model, X, y, epochs, batch_size):
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / (num_samples / batch_size)}, F1: {f1:.3f}")

# Set hyperparameters
epochs = 10
batch_size = 64

# Train the model
train_model(model, X, y, epochs, batch_size)