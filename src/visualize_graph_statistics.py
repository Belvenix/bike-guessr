import typing as tp

import networkx as nx
import osmnx as ox
import pandas as pd
import powerlaw
import torch
from config import (
    CLASSIFIER_WEIGHTS_SAVE_DIR,
    GRAPHML_TEST_DATA_DIR,
    GRAPHML_TRAIN_DATA_DIR,
    GRAPHML_VALIDATION_DATA_DIR,
    VISUALIZATION_OUTPUT_DIR,
)
from networkx.classes.multidigraph import MultiDiGraph
from torch import Tensor
from tqdm import tqdm
from train import MaskedGraphConvolutionalNetwork
from transform_graphs import _load_transform_linegraph

# Define the dimensions
input_dim = 95
output_dim = 4
hidden_dim = 128


def retrieve_cycle_indices(preds: Tensor) -> tp.List[int]:
    """Retrieve indices of cycle predictions.

    Args:
        preds (Tensor): Predictions tensor.
    """
    return (preds > 0).nonzero().squeeze().tolist()


def retrieve_nx_prediction_graphs(graph_networkx: MultiDiGraph, preds: Tensor):
    cycle: nx.Graph = nx.function.create_empty_copy(graph_networkx)
    non_cycle: nx.Graph = nx.function.create_empty_copy(graph_networkx)
    cycle_indices = retrieve_cycle_indices(preds)
    for x in tqdm(set(graph_networkx.edges()), total=len(set(graph_networkx.edges()))):
        edge = graph_networkx[x[0]][x[1]][0]
        edge_attributes = edge.copy()
        is_pred = int(edge['idx']) in cycle_indices

        if is_pred:
            cycle.add_edges_from([(x[0], x[1], edge_attributes)])
        else:
            non_cycle.add_edges_from([(x[0], x[1], edge_attributes)])


def load_graphs():
    train, test, validation = GRAPHML_TEST_DATA_DIR, GRAPHML_TRAIN_DATA_DIR, GRAPHML_VALIDATION_DATA_DIR,
    train_graph_files, test_graph_files, validation_graph_files = \
        list(train.glob('*.xml')), list(test.glob('*.xml')), list(validation.glob('*.xml'))
    train_graphs = [ox.load_graphml(p) for p in tqdm(train_graph_files)]
    test_graphs = [ox.load_graphml(p) for p in tqdm(test_graph_files)]
    validation_graphs = [ox.load_graphml(p) for p in tqdm(validation_graph_files)]
    return train_graphs, test_graphs, validation_graphs


def load_model() -> MaskedGraphConvolutionalNetwork:
    best_model_weights = CLASSIFIER_WEIGHTS_SAVE_DIR / 'best-MaskedGraphConvolutionalNetwork-without-encoding.bin.bin'
    model = MaskedGraphConvolutionalNetwork(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(best_model_weights))
    return model


def get_predictions(model: MaskedGraphConvolutionalNetwork, graphs: tp.List[nx.MultiDiGraph]) -> tp.List[Tensor]:
    predictions = []
    model.set_mask(False)
    for graph in tqdm(graphs):
        line_graph = _load_transform_linegraph(graph)
        g, features = line_graph.clone(), line_graph.clone().ndata['feat']
        preds, _ = model(g, features)
        preds: Tensor = preds.cpu().detach()
        predictions.append(preds)
    return predictions


def calculate_power_law_exponent(graph: nx.MultiDiGraph) -> float:
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    fit = powerlaw.Fit(degree_sequence)
    return fit.power_law.alpha


def calculate_edge_distribution_entropy(graph: nx.MultiDiGraph) -> float:
    return 0.0


def calculate_graph_statistics(graphs: tp.List[nx.MultiDiGraph]) -> pd.DataFrame:
    statistics = {
        'num_nodes': [],
        'num_edges': [],
        'num_components': [],
        'average_degree': [],
        'clustering_coefficient': [],
        'assortativity_coefficient': [],
        'power_law_exponent': [],
        'edge_distribution_entropy': [],
        'algebraic_connectivity': []
    }
    for graph in tqdm(graphs):
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        num_components = nx.algorithms.number_connected_components(graph)
        largest_component = max(nx.connected_components(graph), key=len)
        average_degree = sum([graph.degree(node) for node in largest_component]) / len(largest_component)
        clustering_coefficient = nx.average_clustering(graph)
        assortativity_coefficient = nx.degree_assortativity_coefficient(graph)
        power_law_exponent = calculate_power_law_exponent()
        edge_distribution_entropy = calculate_edge_distribution_entropy()
        algebraic_connectivity = nx.algebraic_connectivity(graph)
        statistics['num_nodes'].append(num_nodes)
        statistics['num_edges'].append(num_edges)
        statistics['num_components'].append(num_components)
        statistics['average_degree'].append(average_degree)
        statistics['clustering_coefficient'].append(clustering_coefficient)
        statistics['assortativity_coefficient'].append(assortativity_coefficient)
        statistics['power_law_exponent'].append(power_law_exponent)
        statistics['edge_distribution_entropy'].append(edge_distribution_entropy)
        statistics['algebraic_connectivity'].append(algebraic_connectivity)
    statistics_df = pd.DataFrame(statistics)
    statistics_df.to_csv(VISUALIZATION_OUTPUT_DIR / 'graph-statistics.csv')
    return statistics_df


def main():
    train, test, validation = GRAPHML_TEST_DATA_DIR, GRAPHML_TRAIN_DATA_DIR, GRAPHML_VALIDATION_DATA_DIR,
    train_graph_files, test_graph_files, validation_graph_files = \
        list(train.glob('*.xml')), list(test.glob('*.xml')), list(validation.glob('*.xml'))
    train_graphs, test_graphs, validation_graphs = [ox.load_graphml(p) for p in train_graph_files], \
        [ox.load_graphml(p) for p in test_graph_files], [ox.load_graphml(p) for p in validation_graph_files]
    model = load_model()
    train_tensor_predictions = get_predictions(model, train_graphs)
    test_tensor_predictions = get_predictions(model, test_graphs)
    validation_tensor_predictions = get_predictions(model, validation_graphs)
    train_graph_predictions = [retrieve_nx_prediction_graphs(graph, preds) for graph, preds in
                               zip(train_graphs, train_tensor_predictions)]
    test_graph_predictions = [retrieve_nx_prediction_graphs(graph, preds) for graph, preds in
                                zip(test_graphs, test_tensor_predictions)]
    validation_graph_predictions = [retrieve_nx_prediction_graphs(graph, preds) for graph, preds in
                                    zip(validation_graphs, validation_tensor_predictions)]
    train_graph_statistics = calculate_graph_statistics(train_graphs)
    test_graph_statistics = calculate_graph_statistics(test_graphs)
    validation_graph_statistics = calculate_graph_statistics(validation_graphs)


if __name__ == '__main__':
    main()
