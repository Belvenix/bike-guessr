import logging
import pickle
import random
import typing as tp

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    CLASSIFIER_OUTPUTS_SAVE_DIR,
    CLASSIFIER_TRAIN_DATA_PATH,
    CLASSIFIER_VALIDATION_DATA_PATH,
    CLASSIFIER_WEIGHTS_SAVE_DIR,
)
from dgl.nn.pytorch.conv.graphconv import GraphConv
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import fast_retrieve_nx_prediction_graph


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


class MaskedGraphConvolutionalNetwork(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, mask_rate=0.5, replace_rate=0.05):
        def py_clip(x, minval, maxval):
            return minval if x < minval else maxval if x > maxval else x
        super(MaskedGraphConvolutionalNetwork, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.act1 = nn.PReLU()
        self.conv2 = GraphConv(h_feats, h_feats)
        self.act2 = nn.PReLU()
        self.conv3 = GraphConv(h_feats, num_classes)
        self._mask_rate = py_clip(mask_rate, 1e-5, 1)
        self._replace_rate = py_clip(replace_rate, 1e-5, 1)
        self._mask = True

    def forward(self, g: dgl.DGLGraph, in_feat: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        masked_feat, masked_node_indices = self.masking_function(g, in_feat, self._mask_rate)
        h = self.act1(self.conv1(g, masked_feat))
        h = self.act2(self.conv2(g, h))
        h = self.conv3(g, h)
        return h, masked_node_indices

    def masking_function(
            self,
            g: dgl.DGLGraph,
            in_feat: torch.Tensor,
            mask_rate: float
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        if not self._mask:
            return in_feat, torch.Tensor()
        # Apply masking logic here
        num_nodes = g.num_nodes()
        num_mask_nodes = int(mask_rate * num_nodes)

        masked_node_indices = torch.randperm(num_nodes, device=in_feat.device)[:num_mask_nodes]

        num_noise_nodes = int(self._replace_rate * num_mask_nodes)
        noisy_node_indices = torch.randperm(len(masked_node_indices), device=in_feat.device)[:num_noise_nodes]
        noise_replacement_node_indices = torch.randperm(num_nodes, device=in_feat.device)[:num_noise_nodes]

        out_feat = in_feat.clone()
        out_feat[masked_node_indices] = 0.0
        out_feat[noisy_node_indices] = in_feat[noise_replacement_node_indices]

        # Return the masked input features
        return out_feat, masked_node_indices

    def set_mask(self, mask: bool):
        self._mask = mask


def validate_model_combined(
        model: nn.Module,
        save: bool = False,
        encoder: nn.Module = None
) -> tp.Tuple[tp.List[float], tp.List[float]]:
    """Tests a model on the validation set.

    Note:
        The model is tested on data from CLASSIFIER_VALIDATION_DATA_PATH.

    Args:
        model (nn.Module): The model to test.
        save (bool): If True, the model weights and predictions are saved to CLASSIFIER_WEIGHTS_SAVE_DIR.
        encoder (nn.Module): If encoder is not None, the model is tested on the encoded data.

    Returns:
        The F1 scores and confusion matrices for each graph in the validation set.
    """
    use_encoding = bool(encoder)
    test_transformed, _ = dgl.load_graphs(str(CLASSIFIER_VALIDATION_DATA_PATH))
    f1_scores, confusion_matrices, outputs = [], [], []
    model_name = model.__class__.__name__ + ('' if use_encoding else '-without-encoding')
    for test_graph in test_transformed:
        g, X, y = test_graph, test_graph.ndata['feat'], test_graph.ndata['label']

        if use_encoding:
            X = encoder.encode(g, X).detach()
            g.ndata['feat'] = X

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
    if save:
        weights_save_dir = CLASSIFIER_WEIGHTS_SAVE_DIR / f'{model_name}.bin'
        output_file = CLASSIFIER_OUTPUTS_SAVE_DIR / f'{model_name}-outputs.pkl'
        torch.save(model.state_dict(), weights_save_dir)
        with open(output_file, 'wb') as f:
            pickle.dump(outputs, f)

    return f1_scores, confusion_matrices


def full_train(
        writer: SummaryWriter,
        model: nn.Module,
        epochs: int,
        encoder: nn.Module,
        early_stopping_patience: int = 25,
        criterion: nn.Module = None
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
            the validation set does not improve. Defaults to 25.
        criterion (nn.Module): The loss function to use. Defaults to nn.CrossEntropyLoss().

    Returns:
        The trained model and the F1 scores and confusion matrices for each graph in the validation set.
    """
    # Define the loss function
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Load the data and shuffle it
    train_transformed, _ = dgl.load_graphs(str(CLASSIFIER_TRAIN_DATA_PATH))
    random.shuffle(train_transformed)

    # Check if the model uses encoding
    use_encoding = bool(encoder)

    # Train the model
    f1_val_best, best_epoch = 0, 0
    for epoch in tqdm(range(epochs)):

        # Set the model to training mode
        model.train()
        f1_train = []
        loss_item = 0
        for train_graph in train_transformed:

            # Zero the gradients
            optimizer.zero_grad()

            # Extract and transform features and labels
            g, X = train_graph.clone(), train_graph.clone().ndata['feat']
            model_name = model.__class__.__name__ + ('' if use_encoding else '-without-encoding')
            is_gnn = isinstance(model, GraphConvolutionalNetwork)
            if use_encoding:
                g.ndata['feat'] = encoder.encode(g, X).detach()
            features = g.ndata['feat']
            labels = g.ndata['label']

            # Forward pass
            if isinstance(model, MaskedGraphConvolutionalNetwork):
                model.set_mask(True)
                outputs, masked_node_indices = model(g, features)
                try:
                    loss = criterion(outputs[masked_node_indices], labels[masked_node_indices])
                except IndexError as e:
                    logging.error(f'IndexError: {e}')
                    logging.error(f'masked_node_indices: {masked_node_indices}')
                    logging.error(f'masked_node_indices shape: {masked_node_indices.shape}')
                    loss = criterion(outputs, labels)
            else:
                outputs = model(g, features) if is_gnn else model(features)
                loss = criterion(outputs, labels)

            # Extract loss
            loss_item += loss.item()

            # Backward
            loss.backward()
            optimizer.step()

            # Evaluate
            model.eval()

            # Compute metrics on the training set
            pred = outputs.argmax(1)
            f1_train.append(f1_score(labels, pred, average='micro'))

        # Compute metrics on the validation set
        f1_val, _ = validate_model_combined(model, encoder=encoder)
        if np.mean(f1_val) > f1_val_best:
            logging.debug(f'New best F1: {np.mean(f1_val):.3f}, last best F1: {f1_val_best:.3f}')
            best_epoch = epoch
            f1_val_best = np.mean(f1_val)
            torch.save(model.state_dict(), CLASSIFIER_WEIGHTS_SAVE_DIR / f'best-{model_name}.bin')

        # Print the average loss for the epoch
        logging.debug(f"Epoch {epoch+1}/{epochs}, Loss: {loss_item}, F1: {np.mean(f1_train):.3f}")
        writer.add_scalar('Loss/train', loss_item, epoch)
        writer.add_scalar('F1/train', np.mean(f1_train), epoch)
        writer.add_scalar('F1/val', np.mean(f1_val), epoch)
        writer.flush()

        # Early stopping
        if epoch - best_epoch > early_stopping_patience:
            logging.debug(f'Early stopping at epoch {epoch}. Last best epoch: {best_epoch}.')
            break

    # Load the best model
    model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS_SAVE_DIR / f'best-{model_name}.bin'))

    # Post training validation
    f1_val, confusion_matrices = validate_model_combined(model, encoder=encoder, save=True)
    return model, (f1_val, confusion_matrices)
