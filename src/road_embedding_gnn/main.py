import logging
import random

import dgl
import torch
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm
from train import GraphConvolutionalNetwork, TrivialClassifier, train_gnn_model, train_trivial_model
from utils import build_args, build_model, load_best_configs

with open('src\\road_embedding_gnn\\encoder-weights.bin', 'rb') as f:
    args = build_args()
    args = load_best_configs(args, "src\\road_embedding_gnn\\configs.yml")
    encoder = build_model(args)
    encoder.load_state_dict(torch.load(f))

# Define the dimensions
input_dim = 32
output_dim = 2
hidden_dim = 128

# Set hyperparameters
epochs = 10
batch_size = 64

def train_loop(model: nn.Module) -> nn.Module:
    train_transformed = dgl.load_graphs('data\\data_transformed\\train.bin')[0]
    random.shuffle(train_transformed)
    for train_graph in tqdm(train_transformed):
        g, X, y = train_graph, train_graph.ndata['feat'], train_graph.ndata['label']
        X = encoder.encode(g, X).detach()
        
        # Train the model
        model = train_trivial_model(model, X, y, epochs, batch_size)
    return model


def test_model(model: nn.Module) -> None:
    test_transformed = dgl.load_graphs('data\\data_transformed\\validation.bin')[0]
    for test_graph in tqdm(test_transformed):
        g, X, y = test_graph, test_graph.ndata['feat'], test_graph.ndata['label']
        X = encoder.encode(g, X).detach()
        
        # Test the model
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        logging.info(f'F1 score for test set: {f1_score(y, predicted, average="micro"):.3f}')
    logging.info('Finished Training')
    torch.save(model.state_dict(), 'src\\road_embedding_gnn\\classifier-weights.bin')


def train_loop_gnn(model: nn.Module) -> nn.Module:
    train_transformed = dgl.load_graphs('data\\data_transformed\\train.bin')[0]
    random.shuffle(train_transformed)
    for train_graph in tqdm(train_transformed):
        g, X = train_graph, train_graph.ndata['feat']
        X = encoder.encode(g, X).detach()
        train_graph.ndata['feat'] = X
        
        # Train the model
        model = train_gnn_model(model, g, epochs)
    return model


def test_model_gnn(model: nn.Module) -> None:
    test_transformed = dgl.load_graphs('data\\data_transformed\\validation.bin')[0]
    for test_graph in tqdm(test_transformed):
        g, X, y = test_graph, test_graph.ndata['feat'], test_graph.ndata['label']
        X = encoder.encode(g, X).detach()
        test_graph.ndata['feat'] = X
        
        # Test the model
        outputs = model(g, X)
        pred = outputs.argmax(1)
        logging.info(f'F1 score for test set: {f1_score(y, pred, average="micro"):.3f}')
    logging.info('Finished Training')
    torch.save(model.state_dict(), 'src\\road_embedding_gnn\\gnn-classifier-weights.bin')



if __name__ == '__main__':
    # Trivial model
    trivial_model = TrivialClassifier(input_dim, hidden_dim, output_dim)
    trained_model = train_loop(trivial_model)
    test_model(trained_model)

    # GNN model
    gnn_model = GraphConvolutionalNetwork(input_dim, hidden_dim, output_dim)
    trained_gnn = train_loop_gnn(gnn_model)
    test_model_gnn(trained_gnn)