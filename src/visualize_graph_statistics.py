import logging
import math
import typing as tp

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import powerlaw
import seaborn as sns
import torch
from config import (
    CLASSIFIER_WEIGHTS_SAVE_DIR,
    PLOT_SAVE_DIR,
    VISUALIZATION_OUTPUT_DIR,
)
from torch import Tensor
from train import MaskedGraphConvolutionalNetwork
from transform_graphs import transform_grahpml
from utils import fast_retrieve_nx_prediction_graph, load_graphs

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# Define the dimensions
input_dim = 95
output_dim = 4
hidden_dim = 128


def load_model() -> MaskedGraphConvolutionalNetwork:
    best_model_weights = CLASSIFIER_WEIGHTS_SAVE_DIR / 'best-MaskedGraphConvolutionalNetwork-without-encoding.bin'
    model = MaskedGraphConvolutionalNetwork(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(best_model_weights))
    return model


def get_predictions(model: MaskedGraphConvolutionalNetwork, graphs: tp.List[nx.MultiDiGraph]) -> tp.List[Tensor]:
    predictions = []
    model.set_mask(False)
    for graph in graphs:
        dgl_linegraph = transform_grahpml(graph)
        output, _ = model(dgl_linegraph, dgl_linegraph.ndata['feat'])
        preds: Tensor = output.argmax(1).cpu().detach()
        predictions.append(preds)
    return predictions


def calculate_power_law_exponent(graph: nx.MultiDiGraph) -> float:
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    fit = powerlaw.Fit(degree_sequence)
    return fit.power_law.alpha


def calculate_edge_distribution_entropy(graph: nx.MultiDiGraph, largest_component: nx.MultiDiGraph) -> float:
    largest_component_degrees = [graph.degree(node) for node in largest_component]
    num_nodes = largest_component.number_of_nodes()
    num_edges = largest_component.number_of_edges()

    ede_sum = 0.0
    for degree in largest_component_degrees:
        if degree == 0:
            continue
        ede_sum += (degree / (2 * num_edges)) * math.log(degree / (2 * num_edges))
    ede = (1 / math.log(num_nodes)) * ede_sum
    return ede


def calculate_graph_statistics(graphs: tp.List[nx.MultiDiGraph], name: str) -> pd.DataFrame:
    statistics = {
        'num_nodes': [],
        'num_edges': [],
        'num_components': [],
        'average_degree': [],
        'clustering_coefficient': [],
        'assortativity_coefficient': [],
        'power_law_exponent': [],
        'edge_distribution_entropy': [],
        'algebraic_connectivity': [],
        'connectedness': [],
    }
    for graph in graphs:
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        assert num_edges > 0, 'The graph must have at least one edge'
        num_components = len(list(nx.algorithms.weakly_connected_components(graph)))
        largest_component = nx.MultiDiGraph(graph.subgraph(max(nx.weakly_connected_components(graph), key=len)))
        average_degree = sum([graph.degree(node) for node in largest_component]) / len(largest_component)
        clustering_coefficient = nx.average_clustering(nx.Graph(graph))
        assortativity_coefficient = nx.degree_assortativity_coefficient(graph)
        power_law_exponent = calculate_power_law_exponent(graph)
        edge_distribution_entropy = calculate_edge_distribution_entropy(graph, largest_component)
        algebraic_connectivity = nx.algebraic_connectivity(largest_component.to_undirected())
        connectedness = largest_component.number_of_nodes() / graph.number_of_nodes()
        statistics['num_nodes'].append(num_nodes)
        statistics['num_edges'].append(num_edges)
        statistics['num_components'].append(num_components)
        statistics['average_degree'].append(average_degree)
        statistics['clustering_coefficient'].append(clustering_coefficient)
        statistics['assortativity_coefficient'].append(assortativity_coefficient)
        statistics['power_law_exponent'].append(power_law_exponent)
        statistics['edge_distribution_entropy'].append(edge_distribution_entropy)
        statistics['algebraic_connectivity'].append(algebraic_connectivity)
        statistics['connectedness'].append(connectedness)
    statistics_df = pd.DataFrame(statistics)
    statistics_df.to_csv(VISUALIZATION_OUTPUT_DIR / f'{name}-graph-statistics.csv')
    return statistics_df


def retrieve_cycle_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    cycle_subgraph = nx.subgraph_view(graph, filter_edge=lambda u, v, e: (graph[u][v][0].get('label') != '0'))
    return nx.MultiDiGraph(cycle_subgraph)


def data_processing(
        graphs: tp.List[nx.MultiDiGraph],
        model: MaskedGraphConvolutionalNetwork,
        name: str
):
    # Retrieve cycle graphs since they are merged into one graph
    logging.info(f'Retrieving cycle {name} graphs')
    cycle_graphs = [retrieve_cycle_graph(graph) for graph in graphs]

    # Retrieve predicted graphs
    logging.info(f'Retrieving predicted {name} graphs')
    tensor_predictions = get_predictions(model, graphs)
    predicted_graphs = [fast_retrieve_nx_prediction_graph(graph, preds) for graph, preds in
                        zip(graphs, tensor_predictions)]

    # Calculate both cycle and predicted graph statistics
    logging.info(f'Calculating {name} graph statistics')
    cycle_stats = calculate_graph_statistics(cycle_graphs, name)
    predicted_stats = calculate_graph_statistics(predicted_graphs, f'pred-{name}')

    return cycle_stats, predicted_stats


def visualize_graph_statistics(
        cycle_stats: pd.DataFrame,
        predicted_stats: pd.DataFrame,
        name: str
):
    sns.set_palette(sns.color_palette('pastel'))

    # Plotting
    logging.info(f'Visualizing {name} graph statistics')
    cols = list(cycle_stats.columns)
    for col in cols:
        data = {"Method": [], col: []}
        for metric, name in zip([list(cycle_stats[col]), list(predicted_stats[col])], ['Real data', 'Predicted data']):
            data[col] += metric
            data["Method"] += [name] * len(metric)
        df = pd.DataFrame(data)
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Method', y=col, data=df)
        plt.title(f'Metric ({col}) on real and predicted data')
        plt.show()
        plt.savefig(PLOT_SAVE_DIR / f'data-metric-{col}.png')


def main():
    # Check if the stats already exists if yes then load it
    try:
        logging.info('Trying to load stats...')
        train_cycle_stats = pd.read_csv(VISUALIZATION_OUTPUT_DIR / 'train-graph-statistics.csv')
        train_predicted_stats = pd.read_csv(VISUALIZATION_OUTPUT_DIR / 'pred-train-graph-statistics.csv')
        validation_cycle_stats = pd.read_csv(VISUALIZATION_OUTPUT_DIR / 'validation-graph-statistics.csv')
        validation_predicted_stats = pd.read_csv(VISUALIZATION_OUTPUT_DIR / 'pred-validation-graph-statistics.csv')
        test_cycle_stats = pd.read_csv(VISUALIZATION_OUTPUT_DIR / 'test-graph-statistics.csv')
        test_predicted_stats = pd.read_csv(VISUALIZATION_OUTPUT_DIR / 'pred-test-graph-statistics.csv')
    except FileNotFoundError:
        logging.info('Stats not found, calculating them')

        # Load graphs - train, test, validation
        logging.info('Loading graphs')
        train_graphs, test_graphs, validation_graphs = load_graphs()

        # Load model
        logging.info('Loading model')
        model = load_model()

        # Data processing
        logging.info('Train data processing')
        train_cycle_stats, train_predicted_stats = data_processing(train_graphs, model, 'train')

        logging.info('Test data processing')
        test_cycle_stats, test_predicted_stats = data_processing(test_graphs, model, 'test')

        logging.info('Validation data processing')
        validation_cycle_stats, validation_predicted_stats = data_processing(validation_graphs, model, 'validation')

    # Visualization
    logging.info('Visualizing train data')
    visualize_graph_statistics(train_cycle_stats, train_predicted_stats, 'train')

    logging.info('Visualizing test data')
    visualize_graph_statistics(test_cycle_stats, test_predicted_stats, 'test')

    logging.info('Visualizing validation data')
    visualize_graph_statistics(validation_cycle_stats, validation_predicted_stats, 'validation')


if __name__ == '__main__':
    main()
