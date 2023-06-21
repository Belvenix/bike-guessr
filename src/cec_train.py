import logging
import random
import typing as tp

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from config import (
    CLASSIFIER_TEST_DATA_PATH,
    CLASSIFIER_TRAIN_DATA_PATH,
    CLASSIFIER_VALIDATION_DATA_PATH,
    CLASSIFIER_WEIGHTS_SAVE_DIR,
)
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from train import GraphConvolutionalNetwork, MaskedGraphConvolutionalNetwork, TrivialClassifier
from utils import fast_retrieve_nx_prediction_graph, load_graphs


def calculate_connectedness(g: nx.MultiDiGraph, outputs: torch.Tensor) -> float:
    preds = torch.argmax(outputs, dim=1)
    bike_graph = fast_retrieve_nx_prediction_graph(g, preds)
    largest_component = nx.MultiDiGraph(bike_graph.subgraph(max(nx.weakly_connected_components(bike_graph), key=len)))
    return largest_component.number_of_nodes() / g.number_of_nodes()


class ConnectednessLoss(nn.Module):
    def __init__(self):
        super(ConnectednessLoss, self).__init__()

    def forward(self, g: nx.MultiDiGraph, outputs: torch.Tensor) -> torch.Tensor:
        connectedness = calculate_connectedness(g, outputs)
        return -torch.log(torch.Tensor([connectedness]))


class CrossEntropyConnectLoss(nn.Module):
    def __init__(self, weight_ce, weight_connect):
        super(CrossEntropyConnectLoss, self).__init__()
        self._weight_ce = weight_ce
        self._weight_connect = weight_connect
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.connect_loss = ConnectednessLoss()

    def forward(self, g: nx.Graph, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.cross_entropy_loss(outputs, targets)
        loss_connect = self.connect_loss(g, outputs)

        return self._weight_ce * loss_ce + self._weight_connect * loss_connect


def recalculate_pareto_frontier(data):
    pareto_frontier = [data[0]]  # Initialize the Pareto frontier with the first data point

    for point in data[1:]:
        # Check if the current point dominates any point on the Pareto frontier
        is_dominated = False
        frontier_copy = pareto_frontier.copy()  # Create a copy to avoid modifying the frontier during iteration

        for frontier_point in frontier_copy:
            if point['f1'] <= frontier_point['f1'] and point['connectedness'] <= frontier_point['connectedness']:
                # If the current point is worse or equal in both metrics, it is dominated
                is_dominated = True
                break

            if point['f1'] >= frontier_point['f1'] and point['connectedness'] >= frontier_point['connectedness']:
                # If the current point is better or equal in both metrics, it dominates the frontier point
                pareto_frontier.remove(frontier_point)

        if not is_dominated:
            pareto_frontier.append(point)  # Add the current point to the Pareto frontier

    return pareto_frontier


def cec_validate_model_combined(
        model: nn.Module,
        dgl_graphs: tp.List[dgl.DGLGraph],
        nx_graphs: tp.List[nx.MultiDiGraph],
) -> tp.Tuple[tp.List[float], tp.List[float], tp.List[float]]:
    """Tests a model on the validation set.

    Note:
        The model is tested on data from CLASSIFIER_VALIDATION_DATA_PATH.

    Args:
        model (nn.Module): The model to test.
        dgl_graphs (tp.List[dgl.DGLGraph]): The DGL graphs to test on.
        nx_graphs (tp.List[nx.MultiDiGraph]): The NetworkX graphs to test on.

    Returns:
        The F1 scores, confusion matrices and connectedness for each graph in the validation set.
    """
    f1_scores, confusion_matrices, outputs, connectedness = [], [], [], []
    for (dgl_g, nx_g) in zip(dgl_graphs, nx_graphs):
        g, X, y = dgl_g, dgl_g.ndata['feat'], dgl_g.ndata['label']

        # Test the model
        if isinstance(model, TrivialClassifier):
            output = model(X).detach()
        elif isinstance(model, GraphConvolutionalNetwork):
            output = model(g, X).detach()
        elif isinstance(model, MaskedGraphConvolutionalNetwork):
            model.set_mask(False)
            output, _ = model(g, X)
            output = output.detach()
        else:
            raise ValueError("Unsupported model type.")

        pred = output.argmax(1)
        f1_scores.append(f1_score(y, pred, average="macro"))
        confusion_matrices.append(confusion_matrix(y, pred))
        outputs.append(output)
        connectedness.append(calculate_connectedness(nx_g, output))

    return f1_scores, confusion_matrices, connectedness


def cec_full_train(
        writer: SummaryWriter,
        model: nn.Module,
        epochs: int,
        encoder: nn.Module,
        early_stopping_patience: int = 5
) -> tp.Tuple[nn.Module, tp.Tuple[tp.List[float], tp.List[float]]]:
    """Trains a model on the training set and tests it on the validation set.

    The training and testing is done in batches of graph size. The model is tested after each batch on the
    validation set. The model with each better F1 score is saved. The model with the best F1 is returned.

    Note:
        The model is trained on data from CLASSIFIER_TRAIN_DATA_PATH.

    Args:
        writer (SummaryWriter): The TensorBoard writer to use for logging.
        model (nn.Module): The model to train.
        epochs (int): The number of epochs to train for.
        encoder (nn.Module): If encoder is not None, the model is trained on the encoded data.
        early_stopping_patience (int): The number of epochs to wait before stopping training if the F1 score on
            the validation set does not improve. Defaults to 5.
        criterion (nn.Module): The loss function to use. Defaults to nn.CrossEntropyLoss().

    Returns:
        The F1 scores, connectedness and confusion matrices for each graph in the validation set.
    """
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the loss function
    criterion = CrossEntropyConnectLoss(1, 1)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Load the data and shuffle it
    dgl_train_graphs, _ = dgl.load_graphs(str(CLASSIFIER_TRAIN_DATA_PATH))
    dgl_test_graphs, _ = dgl.load_graphs(str(CLASSIFIER_TEST_DATA_PATH))
    dgl_val_graphs, _ = dgl.load_graphs(str(CLASSIFIER_VALIDATION_DATA_PATH))
    nx_train_graphs, nx_test_graphs, nx_val_graphs = load_graphs()
    train: tp.Tuple[dgl.DGLGraph, nx.MultiDiGraph] = [(t, g) for t, g in zip(dgl_train_graphs, nx_train_graphs)]
    random.shuffle(train)

    # Check if the model uses encoding
    use_encoding = bool(encoder)
    if use_encoding:
        # Encode the data
        logging.info("Encoding data...")
        for (dgl_train_graph, _) in train:
            dgl_train_graph.ndata['feat'] = encoder.encode(dgl_train_graph, dgl_train_graph.ndata['feat']).detach()
        for dgl_test_graph in dgl_test_graphs:
            dgl_test_graph.ndata['feat'] = encoder.encode(dgl_test_graph, dgl_test_graph.ndata['feat']).detach()
        for dgl_val_graph in dgl_val_graphs:
            dgl_val_graph.ndata['feat'] = encoder.encode(dgl_val_graph, dgl_val_graph.ndata['feat']).detach()
        logging.info("Finished encoding data.")

    # Initialize early stopping variables
    best_epoch, pareto_front = 0, []
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Initialize epoch variables
        epoch_f1_train, epoch_connectedness_train = [], []
        loss_item = 0
        for (dgl_graph, nx_graph) in tqdm(train, desc="Batches"):
            # Set the model to training mode
            model.train()

            # Zero the gradients
            optimizer.zero_grad()

            # Extract and transform features and labels
            model_name = model.__class__.__name__ + ('' if use_encoding else '-without-encoding')
            is_gnn = isinstance(model, GraphConvolutionalNetwork)
            dgl_graph = dgl_graph.to(device)
            features = dgl_graph.ndata['feat'].to(device)
            labels = dgl_graph.ndata['label'].to(device)

            # Forward pass
            if isinstance(model, MaskedGraphConvolutionalNetwork):
                model.set_mask(True)
                outputs, masked_node_indices = model(dgl_graph, features)
                try:
                    loss = criterion(nx_graph, outputs[masked_node_indices], labels[masked_node_indices])
                except IndexError as e:
                    logging.error(f'IndexError: {e}')
                    logging.error(f'masked_node_indices: {masked_node_indices}')
                    logging.error(f'masked_node_indices shape: {masked_node_indices.shape}')
                    loss = criterion(nx_graph, outputs, labels)
            else:
                outputs = model(dgl_graph, features) if is_gnn else model(features)
                loss = criterion(nx_graph, outputs, labels)

            # Extract loss
            loss_item += loss.item()

            # Backward
            loss.backward()
            optimizer.step()

            # Evaluate
            model.eval()

            # Compute metrics on the training set
            pred = outputs.argmax(1)
            epoch_f1_train.append(f1_score(labels, pred, average='micro'))
            epoch_connectedness_train.append(calculate_connectedness(nx_graph, outputs))

        # Compute metrics on the validation set
        model.eval()
        epoch_f1_val, _, epoch_connectedness_val = \
            cec_validate_model_combined(model, dgl_graphs=dgl_val_graphs, nx_graphs=nx_val_graphs)
        metric = {'f1': np.mean(epoch_f1_val), 'connectedness': np.mean(epoch_connectedness_val), 'epoch': epoch+1}
        pareto_front = [metric, *pareto_front]

        # Check pareto optimality early stopping condition
        pareto_front = recalculate_pareto_frontier(pareto_front)
        if metric in pareto_front:
            logging.debug(f'New pareto optimal model found: {metric}')
            best_epoch = epoch
            torch.save(model.state_dict(), CLASSIFIER_WEIGHTS_SAVE_DIR / f'best-cec-{model_name}.bin')

        # Print the average loss for the epoch
        logging.debug(f"Epoch {epoch+1}/{epochs}, \
                      Loss: {loss_item}, \
                      F1: {np.mean(epoch_f1_train):.3f}, \
                      Connectedness: {np.mean(epoch_connectedness_train):.3f}")
        writer.add_scalar('Loss/train', loss_item, epoch)
        writer.add_scalar('F1/train', np.mean(epoch_f1_train), epoch)
        writer.add_scalar('F1/val', np.mean(epoch_f1_val), epoch)
        writer.add_scalar('Connectedness/train', np.mean(epoch_connectedness_train), epoch)
        writer.add_scalar('Connectedness/val', np.mean(epoch_connectedness_val), epoch)
        writer.flush()

        # Early stopping
        if epoch - best_epoch > early_stopping_patience:
            logging.debug(f'Early stopping at epoch {epoch}. Last best epoch: {best_epoch}.')
            break

    # Load the best model
    model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS_SAVE_DIR / f'best-cec-{model_name}.bin'))
    model.eval()

    # Post training validation
    test_f1, test_conf_matrices, test_connectedness = \
        cec_validate_model_combined(model, dgl_graphs=dgl_test_graphs, nx_graphs=nx_test_graphs)
    return (test_f1, test_conf_matrices, test_connectedness)
