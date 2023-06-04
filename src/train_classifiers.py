import logging
import pickle
import random
import typing as tp

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
    ENCODER_CONFIG_PATH,
    ENCODER_WEIGHTS_PATH,
    PLOT_SAVE_DIR,
)
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from tqdm import tqdm
from train import GraphConvolutionalNetwork, TrivialClassifier, train_gnn_model, train_trivial_model
from utils import build_args, build_model, load_best_configs
from wild_debugging import exception_exit_handler

logging.basicConfig(level=logging.INFO)

logging.info("Loading encoder...")
with open(ENCODER_WEIGHTS_PATH, 'rb') as f:
    args = build_args()
    args = load_best_configs(args, ENCODER_CONFIG_PATH.absolute())
    encoder = build_model(args)
    encoder.load_state_dict(torch.load(f))

# Define the dimensions
input_dim = 32
output_dim = 2
hidden_dim = 128

# Set hyperparameters
epochs = 10
batch_size = 512

# Comparison parameters
repeats = 50

# File parameters
trivial_outputs = CLASSIFIER_OUTPUTS_SAVE_DIR / 'classifier-outputs.pkl'
gnn_outputs = CLASSIFIER_OUTPUTS_SAVE_DIR / 'gnn-classifier-outputs.pkl'
gnn_without_encoding_outputs = CLASSIFIER_OUTPUTS_SAVE_DIR / 'gnn-classifier-without-encoding-outputs.pkl'


def train_loop(model: nn.Module) -> nn.Module:
    train_transformed = dgl.load_graphs(str(CLASSIFIER_TRAIN_DATA_PATH))[0]
    random.shuffle(train_transformed)
    for train_graph in train_transformed:
        g, X, y = train_graph, train_graph.ndata['feat'], train_graph.ndata['label']
        X = encoder.encode(g, X).detach()
        
        # Train the model
        model = train_trivial_model(model, X, y, epochs, batch_size)
    return model


def test_model(model: nn.Module) -> tp.List[float]:
    test_transformed = dgl.load_graphs(str(CLASSIFIER_VALIDATION_DATA_PATH))[0]
    f1_scores, confusion_matrices, outputs = [], [], []
    for test_graph in test_transformed:
        g, X, y = test_graph, test_graph.ndata['feat'], test_graph.ndata['label']
        X = encoder.encode(g, X).detach()
        
        # Test the model
        output = model(X).detach()
        _, pred = torch.max(output.data, 1)
        f1_scores.append(round(f1_score(y, pred, average="binary"), 5))
        confusion_matrices.append(confusion_matrix(y, pred))
        outputs.append(output)
    torch.save(model.state_dict(), CLASSIFIER_WEIGHTS_SAVE_DIR / 'classifier-weights.bin')
    with open(trivial_outputs, 'wb') as f:
        pickle.dump(outputs, f)
    return f1_scores, confusion_matrices


def train_loop_gnn(model: nn.Module) -> nn.Module:
    train_transformed = dgl.load_graphs(str(CLASSIFIER_TRAIN_DATA_PATH))[0]
    random.shuffle(train_transformed)
    for train_graph in train_transformed:
        g, X = train_graph, train_graph.ndata['feat']
        X = encoder.encode(g, X).detach()
        train_graph.ndata['feat'] = X
        
        # Train the model
        model = train_gnn_model(model, g, epochs, 'GNN')
    return model


def test_model_gnn(model: nn.Module) -> tp.List[float]:
    test_transformed = dgl.load_graphs(str(CLASSIFIER_VALIDATION_DATA_PATH))[0]
    f1_scores, confusion_matrices, outputs = [], [], []
    for test_graph in test_transformed:
        g, X, y = test_graph, test_graph.ndata['feat'], test_graph.ndata['label']
        X = encoder.encode(g, X).detach()
        test_graph.ndata['feat'] = X
        
        # Test the model
        output = model(g, X).detach()
        pred = output.argmax(1)
        f1_scores.append(round(f1_score(y, pred, average="binary"), 5))
        confusion_matrices.append(confusion_matrix(y, pred))
        outputs.append(output)
    torch.save(model.state_dict(), CLASSIFIER_WEIGHTS_SAVE_DIR / 'gnn-classifier-weights.bin')
    with open(gnn_outputs, 'wb') as f:
        pickle.dump(outputs, f)
    return f1_scores, confusion_matrices


def train_loop_gnn_without_encoding(model: nn.Module) -> nn.Module:
    train_transformed = dgl.load_graphs(str(CLASSIFIER_TRAIN_DATA_PATH))[0]
    random.shuffle(train_transformed)
    for train_graph in train_transformed:
  
        # Train the model
        model = train_gnn_model(model, train_graph, epochs, 'GNN-without-encoding')
    return model


def test_model_gnn_without_encoding(model: nn.Module) -> tp.List[float]:
    test_transformed = dgl.load_graphs(str(CLASSIFIER_VALIDATION_DATA_PATH))[0]
    f1_scores, confusion_matrices, outputs = [], [], []
    for test_graph in test_transformed:
        g, X, y = test_graph, test_graph.ndata['feat'], test_graph.ndata['label']

        # Test the model
        output = model(g, X).detach()
        pred = output.argmax(1)
        f1_scores.append(round(f1_score(y, pred, average="binary"), 5))
        confusion_matrices.append(confusion_matrix(y, pred))
        outputs.append(output)
    torch.save(model.state_dict(), CLASSIFIER_WEIGHTS_SAVE_DIR / 'gnn-classifier-without-encoding-weights.bin')
    with open(gnn_without_encoding_outputs, 'wb') as f:
        pickle.dump(outputs, f)
    return f1_scores, confusion_matrices


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


def plot_confusion_matrices(
        trivial_confusion_matrices: np.ndarray,
        gnn_confusion_matrices: np.ndarray,
        gnn_we_confusion_matrices: np.ndarray
    ) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    sns.color_palette("pastel")
    sns.heatmap(trivial_confusion_matrices[0], annot=True, ax=axs[0], fmt = '.5f')
    axs[0].set_title('Trivial Confusion Matrix')
    sns.heatmap(gnn_confusion_matrices[0], annot=True, ax=axs[1], fmt = '.5f')
    axs[1].set_title('GNN Confusion Matrix')
    sns.heatmap(gnn_we_confusion_matrices[0], annot=True, ax=axs[2], fmt = '.5f')
    axs[2].set_title('GNN_WE Confusion Matrix')
    plt.savefig(PLOT_SAVE_DIR / 'confusion-matrices.png')


def plot_data_balance():
    val_graphs = dgl.load_graphs(str(CLASSIFIER_VALIDATION_DATA_PATH))[0]

    class_balances = {0: 0, 1: 0}  # Initialize class balances for binary labels
    for graph in val_graphs:
        # Access the 'label' ndata attribute of the current graph
        labels = graph.ndata['label']
        
        # Compute the class balance
        class_counts = {0: (labels == 0).sum(), 1: (labels == 1).sum()}
        
        # Update the class balances dictionary
        for cls, count in class_counts.items():
            class_balances[cls] += count
    all_nodes = sum(class_balances.values()).item()
    logging.info(f"Class balance. Bike: {class_balances[1].item() / all_nodes:.4f}, No Bike: {class_balances[0] / all_nodes:.4f}")


@exception_exit_handler
def main():
    plot_data_balance()
    trivial_f1_means, gnn_f1_means, gnn_we_f1_means = [], [], []
    trivial_f1_file, gnn_f1_file, gnn_we_f1_file = CLASSIFIER_OUTPUTS_SAVE_DIR / 'trivial-f1-scores.pkl', CLASSIFIER_OUTPUTS_SAVE_DIR / 'gnn-f1-scores.pkl', CLASSIFIER_OUTPUTS_SAVE_DIR /  'gnn-we-f1-scores.pkl'
    trivial_confusion_matrices, gnn_confusion_matrices, gnn_we_confusion_matrices = [], [], []

    logging.info("Begin training...")
    # Trivial model
    logging.info("Trivial model training...")
    if not trivial_outputs.exists():
        for _ in tqdm(range(repeats), desc="Training trivial model..."):
            trivial_model = TrivialClassifier(input_dim, hidden_dim, output_dim)
            trained_model = train_loop(trivial_model)
            f1_scores, confusion_matrices = test_model(trained_model)
            trivial_f1, mean_confusion_matrix = np.mean(f1_scores), np.mean(confusion_matrices, axis=0) 
            trivial_f1_means.append(trivial_f1)
            trivial_confusion_matrices.append(mean_confusion_matrix)
        with open(trivial_f1_file, 'wb') as f:
            pickle.dump((trivial_f1_means, trivial_confusion_matrices), f)
    else:
        logging.warning("Trivial model already trained. Skipping training...")
        with open(trivial_f1_file, 'rb') as f:
            trivial_f1_means, trivial_confusion_matrices = pickle.load(f)

    # GNN model
    logging.info("GNN model training...")
    if not gnn_outputs.exists():
        for _ in tqdm(range(repeats), desc="Training GNN model..."):
            gnn_model = GraphConvolutionalNetwork(input_dim, hidden_dim, output_dim)
            trained_gnn = train_loop_gnn(gnn_model)
            gnn_f1, confusion_matrices = test_model_gnn(trained_gnn)
            gnn_f1, mean_confusion_matrix = np.mean(gnn_f1), np.mean(confusion_matrices, axis=0) 
            gnn_f1_means.append(gnn_f1)
            gnn_confusion_matrices.append(mean_confusion_matrix)
        with open(gnn_f1_file, 'wb') as f:
            pickle.dump((gnn_f1_means, gnn_confusion_matrices), f)
    else:
        logging.warning("GNN model already trained. Skipping training...")
        with open(gnn_f1_file, 'rb') as f:
            gnn_f1_means, gnn_confusion_matrices = pickle.load(f)

    # GNN model without encoding
    logging.info("GNN model without encoding training...")
    if not gnn_without_encoding_outputs.exists():
        for _ in tqdm(range(repeats), desc="Training GNN model without encoding..."):
            gnn_model_without_encoding = GraphConvolutionalNetwork(95, hidden_dim, output_dim)
            trained_gnn_without_encoding = train_loop_gnn_without_encoding(gnn_model_without_encoding)
            gnn_we_f1, confusion_matrices = test_model_gnn_without_encoding(trained_gnn_without_encoding)
            gnn_we_f1, mean_confusion_matrix = np.mean(gnn_we_f1), np.mean(confusion_matrices, axis=0)
            gnn_we_f1_means.append(gnn_we_f1)
            gnn_we_confusion_matrices.append(mean_confusion_matrix)
        with open(gnn_we_f1_file, 'wb') as f:
            pickle.dump((gnn_we_f1_means, gnn_we_confusion_matrices), f)
    else:
        logging.warning("GNN model without encoding already trained. Skipping training...")
        with open(gnn_we_f1_file, 'rb') as f:
            gnn_we_f1_means, gnn_we_confusion_matrices = pickle.load(f)

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
    plot_confusion_matrices(trivial_confusion_matrices, gnn_confusion_matrices, gnn_we_confusion_matrices)


if __name__ == '__main__':
    main()