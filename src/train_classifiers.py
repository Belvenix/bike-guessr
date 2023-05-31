import logging
import pickle
import random
import typing as tp
from pathlib import Path

import dgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from config import (
    CLASSIFIER_OUTPUTS_SAVE_DIR,
    CLASSIFIER_TRAIN_DATA_PATH,
    CLASSIFIER_VALIDATION_DATA_PATH,
    CLASSIFIER_WEIGHTS_SAVE_DIR,
    PLOT_SAVE_DIR,
)
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm
from train import GraphConvolutionalNetwork, TrivialClassifier, train_gnn_model, train_trivial_model
from utils import build_args, build_model, load_best_configs

logging.basicConfig(level=logging.INFO)

logging.info("Loading encoder...")
with open('./encoder-weights.bin', 'rb') as f:
    args = build_args()
    args = load_best_configs(args, "./configs.yml")
    encoder = build_model(args)
    encoder.load_state_dict(torch.load(f))

# Define the dimensions
input_dim = 32
output_dim = 2
hidden_dim = 128

# Set hyperparameters
epochs = 10
batch_size = 256

# Comparison parameters
repeats = 50


def train_loop(model: nn.Module) -> nn.Module:
    train_transformed = dgl.load_graphs(CLASSIFIER_TRAIN_DATA_PATH)[0]
    random.shuffle(train_transformed)
    for train_graph in train_transformed:
        g, X, y = train_graph, train_graph.ndata['feat'], train_graph.ndata['label']
        X = encoder.encode(g, X).detach()
        
        # Train the model
        model = train_trivial_model(model, X, y, epochs, batch_size)
    return model


def test_model(model: nn.Module) -> tp.List[float]:
    test_transformed = dgl.load_graphs(CLASSIFIER_VALIDATION_DATA_PATH)[0]
    f1_scores, outputs = [], []
    for test_graph in test_transformed:
        g, X, y = test_graph, test_graph.ndata['feat'], test_graph.ndata['label']
        X = encoder.encode(g, X).detach()
        
        # Test the model
        output = model(X).detach()
        _, pred = torch.max(output.data, 1)
        f1_scores.append(round(f1_score(y, pred, average="micro"), 5))
        outputs.append(output)
    torch.save(model.state_dict(), CLASSIFIER_WEIGHTS_SAVE_DIR / '/classifier-weights.bin')
    with open(CLASSIFIER_OUTPUTS_SAVE_DIR / 'classifier-outputs.pkl', 'wb') as f:
        pickle.dump(outputs, f)
    return f1_scores


def train_loop_gnn(model: nn.Module) -> nn.Module:
    train_transformed = dgl.load_graphs(CLASSIFIER_TRAIN_DATA_PATH)[0]
    random.shuffle(train_transformed)
    for train_graph in train_transformed:
        g, X = train_graph, train_graph.ndata['feat']
        X = encoder.encode(g, X).detach()
        train_graph.ndata['feat'] = X
        
        # Train the model
        model = train_gnn_model(model, g, epochs, 'GNN')
    return model


def test_model_gnn(model: nn.Module) -> tp.List[float]:
    test_transformed = dgl.load_graphs(CLASSIFIER_VALIDATION_DATA_PATH)[0]
    f1_scores, outputs = [], []
    for test_graph in test_transformed:
        g, X, y = test_graph, test_graph.ndata['feat'], test_graph.ndata['label']
        X = encoder.encode(g, X).detach()
        test_graph.ndata['feat'] = X
        
        # Test the model
        output = model(g, X).detach()
        pred = output.argmax(1)
        f1_scores.append(round(f1_score(y, pred, average="micro"), 5))
        outputs.append(output)
    torch.save(model.state_dict(), CLASSIFIER_WEIGHTS_SAVE_DIR / 'gnn-classifier-weights.bin')
    with open(CLASSIFIER_OUTPUTS_SAVE_DIR / 'gnn-classifier-outputs.pkl', 'wb') as f:
        pickle.dump(outputs, f)
    return f1_scores


def train_loop_gnn_without_encoding(model: nn.Module) -> nn.Module:
    train_transformed = dgl.load_graphs(CLASSIFIER_TRAIN_DATA_PATH)[0]
    random.shuffle(train_transformed)
    for train_graph in train_transformed:
  
        # Train the model
        model = train_gnn_model(model, train_graph, epochs, 'GNN-without-encoding')
    return model


def test_model_gnn_without_encoding(model: nn.Module) -> tp.List[float]:
    test_transformed = dgl.load_graphs(CLASSIFIER_VALIDATION_DATA_PATH)[0]
    f1_scores, outputs = [], []
    for test_graph in test_transformed:
        g, X, y = test_graph, test_graph.ndata['feat'], test_graph.ndata['label']

        # Test the model
        output = model(g, X).detach()
        pred = output.argmax(1)
        f1_scores.append(round(f1_score(y, pred, average="micro"), 5))
        outputs.append(output)
    torch.save(model.state_dict(), CLASSIFIER_WEIGHTS_SAVE_DIR / 'gnn-classifier-without-encoding-weights.bin')
    with open(CLASSIFIER_OUTPUTS_SAVE_DIR / 'gnn-classifier-without-encoding-outputs.pkl', 'wb') as f:
        pickle.dump(outputs, f)
    return f1_scores


def plot_f1_scores(
        trivial_f1_means: tp.List[float], 
        gnn_f1_means: tp.List[float], 
        gnn_we_f1_means: tp.List[float]
    ) -> None:
    data = pd.DataFrame({'Method': ['Trivial'] * len(trivial_f1_means) +
                                ['GNN'] * len(gnn_f1_means) +
                                ['GNN_WE'] * len(gnn_we_f1_means),
                        'F1 Score': trivial_f1_means + gnn_f1_means + gnn_we_f1_means})
    sns.set(style='whitegrid')  # Optional: Set a white grid background
    plt.figure(figsize=(8, 6))  # Optional: Set the figure size

    # Create the boxplot
    sns.boxplot(x='Method', y='F1 Score', data=data)
    plt.title('F1 Scores of Trivial and GNN Models with and without encoding')
    plt.savefig(PLOT_SAVE_DIR / 'f1-scores.png')


if __name__ == '__main__':
    trivial_f1_means = []
    gnn_f1_means = []
    gnn_we_f1_means = []

    logging.info("Begin training...")
    # Trivial model
    for _ in tqdm(range(repeats), desc="Training trivial model..."):
        trivial_model = TrivialClassifier(input_dim, hidden_dim, output_dim)
        trained_model = train_loop(trivial_model)
        trivial_f1 = np.mean(test_model(trained_model))
        trivial_f1_means.append(trivial_f1)

    # GNN model
    for _ in tqdm(range(repeats), desc="Training GNN model..."):
        gnn_model = GraphConvolutionalNetwork(input_dim, hidden_dim, output_dim)
        trained_gnn = train_loop_gnn(gnn_model)
        gnn_f1 = np.mean(test_model_gnn(trained_gnn))
        gnn_f1_means.append(gnn_f1)

    # GNN model without encoding
    for _ in tqdm(range(repeats), desc="Training GNN model without encoding..."):
        gnn_model_without_encoding = GraphConvolutionalNetwork(95, hidden_dim, output_dim)
        trained_gnn_without_encoding = train_loop_gnn_without_encoding(gnn_model_without_encoding)
        gnn_we_f1 = np.mean(test_model_gnn_without_encoding(trained_gnn_without_encoding))
        gnn_we_f1_means.append(gnn_we_f1)

    # Print results
    logging.info("Results:")
    logging.info(f"Trivial model F1 scores: {trivial_f1_means}")
    logging.info(f"GNN model F1 scores: {gnn_f1_means}")
    logging.info(f"GNN model without encoding F1 scores: {gnn_we_f1_means}")
    plot_f1_scores(trivial_f1_means, gnn_f1_means, gnn_we_f1_means)
    logging.info("Mean results:")
    logging.info(f"Trivial model F1 score: {np.mean(trivial_f1_means)}")
    logging.info(f"GNN model F1 score: {np.mean(gnn_f1_means)}")
    logging.info(f"GNN model without encoding F1 score: {np.mean(gnn_we_f1_means)}")
