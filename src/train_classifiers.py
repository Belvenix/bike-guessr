import logging
import pickle
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
    CLASSIFIER_VALIDATION_DATA_PATH,
    ENCODER_CONFIG_PATH,
    ENCODER_WEIGHTS_PATH,
    PLOT_SAVE_DIR,
)
from torch import nn
from train import GraphConvolutionalNetwork, TrivialClassifier, full_train
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
epochs = 250
batch_size = 512

# File parameters
trivial_outputs = CLASSIFIER_OUTPUTS_SAVE_DIR / 'classifier-outputs.pkl'
gnn_outputs = CLASSIFIER_OUTPUTS_SAVE_DIR / 'gnn-classifier-outputs.pkl'
gnn_without_encoding_outputs = CLASSIFIER_OUTPUTS_SAVE_DIR / 'gnn-classifier-without-encoding-outputs.pkl'


def train_test_skip_existing(
        model_name: str,
        model_outputs_file: Path,
        model: nn.Module,
        use_encoding: bool = True,
    ) -> tp.Tuple[tp.List[float], tp.List[float]]:
    """Trains and tests a model, then saves the test outputs to a file.
    
    The function first checks if the model has already been trained and tested. If so, it loads the
    test outputs from the file. Otherwise, it trains and tests the model, then saves the test
    outputs to the file.

    Args:
        model_name: The name of the model.
        model_outputs_file: The path to the file where the test outputs are saved.
        model: The model to train and test.
        use_encoding: Whether to use the encoder.

    Returns:
        The mean F1 score and the mean confusion matrix.
    """
    f1_means, confusion_matrices = [], []
    encoder_ = encoder if use_encoding else None
    logging.info(f"Training {model_name}...")
    if not model_outputs_file.exists():
        trained_model, (f1_scores, confusion_matrices) = full_train(model, epochs, encoder_)
        f1_means.append(np.mean(f1_scores))
        confusion_matrices.append(confusion_matrices)
        with open(model_outputs_file, 'wb') as f:
            pickle.dump((f1_means, confusion_matrices), f)
    else:
        logging.warning(f"{model_name} already trained. Skipping training...")
        with open(model_outputs_file, 'rb') as f:
            f1_means, confusion_matrices = pickle.load(f)
    return f1_means, confusion_matrices


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
    logging.info(f"Class balance. \
                 Bike: {class_balances[1].item() / all_nodes:.4f}, \
                 No Bike: {class_balances[0] / all_nodes:.4f}")


#@exception_exit_handler
def main():
    
    # Define 
    trivial_metrics_file = CLASSIFIER_OUTPUTS_SAVE_DIR / 'trivial-metrics.pkl'
    gnn_metrics_file = CLASSIFIER_OUTPUTS_SAVE_DIR / 'gnn-metrics.pkl'
    gnn_we_metrics_file = CLASSIFIER_OUTPUTS_SAVE_DIR / 'gnn-we-metrics.pkl'
    
    logging.debug("Begin training...")
    
    # Trivial model
    trivial_f1_means, trivial_confusion_matrices = train_test_skip_existing(
        'Trivial model', trivial_metrics_file, TrivialClassifier(input_dim, hidden_dim, output_dim)
    )

    # GNN model
    gnn_f1_means, gnn_confusion_matrices = train_test_skip_existing(
        'GNN model', gnn_metrics_file, GraphConvolutionalNetwork(input_dim, hidden_dim, output_dim)
    )

    # GNN model without encoding
    gnn_we_f1_means, gnn_we_confusion_matrices = train_test_skip_existing(
        model_name='GNN model without encoding', 
        model_outputs_file=gnn_we_metrics_file, 
        model=GraphConvolutionalNetwork(95, hidden_dim, output_dim), 
        use_encoding=False
    )

    logging.debug("Training complete.")
    
    # Print results
    logging.info("Mean results:")
    logging.info(f"Trivial model F1 score: {np.mean(trivial_f1_means)}")
    logging.info(f"GNN model F1 score: {np.mean(gnn_f1_means)}")
    logging.info(f"GNN model without encoding F1 score: {np.mean(gnn_we_f1_means)}")

    # Plot results
    plot_data_balance()
    plot_f1_scores(trivial_f1_means, gnn_f1_means, gnn_we_f1_means)
    plot_confusion_matrices(trivial_confusion_matrices, gnn_confusion_matrices, gnn_we_confusion_matrices)


if __name__ == '__main__':
    main() 