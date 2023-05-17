
import torch
from utils import build_model, load_best_configs, build_args
from train import train_model, Classifier
import networkx as nx
import dgl
import random
from tqdm import tqdm
from torch import nn
from sklearn.metrics import f1_score

with open('src\\road-embedding-gnn\\encoder-weights.bin', 'rb') as f:
    args = build_args()
    args = load_best_configs(args, "src\\road-embedding-gnn\\configs.yml")
    encoder = build_model(args)
    encoder.load_state_dict(torch.load(f))

# Define the dimensions
input_dim = 32
output_dim = 2
hidden_dim = 128

# Set hyperparameters
epochs = 10
batch_size = 64

# Create an instance of the classifier model
trivial_model = Classifier(input_dim, hidden_dim, output_dim)

def train_loop(model: nn.Module) -> nn.Module:
    train_transformed = dgl.load_graphs('data\\data_transformed\\train.bin')[0]
    random.shuffle(train_transformed)
    for train_graph in tqdm(train_transformed):
        g, X, y = train_graph, train_graph.ndata['feat'], train_graph.ndata['label']
        X = encoder.encode(g, X).detach()
        
        # Train the model
        model = train_model(model, X, y, epochs, batch_size)
    return model

trained_model = train_loop(trivial_model)

def test_model(model: nn.Module) -> None:
    test_transformed = dgl.load_graphs('data\\data_transformed\\validation.bin')[0]
    for test_graph in tqdm(test_transformed):
        g, X, y = test_graph, test_graph.ndata['feat'], test_graph.ndata['label']
        X = encoder.encode(g, X).detach()
        
        # Test the model
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        print(f'F1 score for test set: {f1_score(y, predicted, average="micro"):.3f}')
    print('Finished Training')
    torch.save(model.state_dict(), 'src\\road-embedding-gnn\\classifier-weights.bin')

test_model(trained_model)