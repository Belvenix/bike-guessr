
import torch
from utils import build_model, load_best_configs, build_args
from train import train_model, Classifier
import networkx as nx
import dgl


with open('src\\road-embedding-gnn\\encoder-weights.bin', 'rb') as f:
    args = build_args()
    args = load_best_configs(args, "src\\road-embedding-gnn\\configs.yml")
    encoder = build_model(args)
    encoder.load_state_dict(torch.load(f))


wroclaw_graph = nx.readwrite.graphml.read_graphml('src\\road-embedding-gnn\\Wroclaw.xml')

print(len(wroclaw_graph.nodes))

wroclaw_transformed = dgl.load_graphs('src\\road-embedding-gnn\\wrocek.bin')[0][0]

print(wroclaw_transformed)

g, X, y = wroclaw_transformed, wroclaw_transformed.ndata['feat'], wroclaw_transformed.ndata['label']
X = encoder.encode(g, X)

print(X)

print(X.shape)

X,y = X.detach(), y.detach()


# Define the dimensions
input_dim = 32
output_dim = 2
hidden_dim = 128

# Create an instance of the classifier model
model = Classifier(input_dim, hidden_dim, output_dim)

# Set hyperparameters
epochs = 10
batch_size = 64

# Train the model
train_model(model, X, y, epochs, batch_size)