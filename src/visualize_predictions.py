import logging
import pickle
import typing as tp
from pathlib import Path

import folium
import networkx as nx
import osmnx as ox
import torch
from config import (
    CLASSIFIER_OUTPUTS_SAVE_DIR,
    GRAPHML_VALIDATION_DATA_DIR,
    VISUALIZATION_LEGEND_PATH,
    VISUALIZATION_OUTPUT_DIR,
)
from folium.plugins import FloatImage
from networkx.classes.multidigraph import MultiDiGraph
from torch import Tensor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def add_edge_to_graph(
        graph: nx.Graph, 
        edge: tp.Tuple[int, int], 
        edge_attributes: tp.Dict[str, tp.Any], 
        popup: bool
    ) -> None:
    """Add edge to graph with popup if needed.

    Args:
        graph (nx.Graph): Graph to add edge to.
        edge (tp.Tuple[int, int]): Edge to add.
        edge_attributes (tp.Dict[str, tp.Any]): Edge attributes.
        popup (bool): Whether to add popup to edge.
    """
    edge_attributes['vis_data'] = {
                'href': f"https://www.openstreetmap.org/way/{edge_attributes['osmid']}"
    }
    if not popup:
        edge_attributes = None
    graph.add_edges_from([(edge[0], edge[1], edge_attributes)])


def retrieve_cycle_indices(preds: Tensor) -> tp.List[int]:
    """Retrieve indices of cycle predictions.

    Args:
        preds (Tensor): Predictions tensor.
    """
    return (preds == 1).nonzero().squeeze().tolist()


def divide_graphs(
        grapf_networkx: MultiDiGraph, 
        preds: Tensor,
        popup: bool
    ) -> tp.Tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph]:
    """Divide graph into true positive, true negative, false positive and false negative.

    Args:
        grapf_networkx (MultiDiGraph): Graph to divide.
        preds (Tensor): Predictions tensor.
        popup (bool): Whether to add popup to edge.

    Raises:
        ValueError: If graph labels are not binary.

    Returns:
        tp.Tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph]: 
            True positive, true negative, false positive and false negative graphs.
    """
    pred_ids = retrieve_cycle_indices(preds)

    cycle_true_positive: nx.Graph = nx.function.create_empty_copy(grapf_networkx)
    cycle_true_negative: nx.Graph = nx.function.create_empty_copy(grapf_networkx)
    cycle_false_positive: nx.Graph = nx.function.create_empty_copy(grapf_networkx)
    cycle_false_negative: nx.Graph = nx.function.create_empty_copy(grapf_networkx)

    for x in tqdm(set(grapf_networkx.edges()), total=len(set(grapf_networkx.edges()))):
        edge = grapf_networkx[x[0]][x[1]][0]
        edge_attributes = edge.copy()
        label = int(edge_attributes['label'])
        is_pred = int(edge['idx']) in pred_ids

        # True positive
        if label == 1 and is_pred:
            add_edge_to_graph(cycle_true_positive, x, edge_attributes, popup)

        # True negative
        elif label == 0 and not is_pred:
            add_edge_to_graph(cycle_true_negative, x, edge_attributes, popup)

        # False positive
        elif label == 0 and is_pred:
            add_edge_to_graph(cycle_false_positive, x, edge_attributes, popup)

        # False negative
        elif label == 1 and not is_pred:  
            add_edge_to_graph(cycle_false_negative, x, edge_attributes, popup)

        else:
            raise ValueError(f"Edge label should be 0 or 1 and is {int(edge_attributes['label'])}")
    return cycle_true_positive, cycle_true_negative, cycle_false_positive, cycle_false_negative


def add_graph_to_map(
        graph: nx.Graph,
        feature_group_name: str,
        folium_map: folium.Map,
        color: str,
        edge_width: int,
        popup: bool
    ) -> None:
    """Add graph to map.

    This method utilizes unknown feature of osmnx library, which is not documented. The graph_map parameter
    can be either folium.Map or folium.FeatureGroup. If it is folium.FeatureGroup, the graph will be added
    to this feature group. This is useful for adding multiple graphs to one map.

    Args:
        graph (nx.Graph): Graph to add.
        feature_group_name (str): Name of the feature group.
        folium_map (folium.Map): Map to add graph to.
        color (str): Color of the graph.
        edge_width (int): Width of the edges.
        popup (bool): Whether to add popup to edge.
    """
    try:
        feature_group = folium.FeatureGroup(name=feature_group_name)
        logging.info(f'{feature_group_name} edges: {graph.number_of_edges()}')
        if not popup:
            m = ox.plot_graph_folium(
                graph, graph_map=feature_group, color=color, edge_width=edge_width)
        else:
            m = ox.plot_graph_folium(
                graph, graph_map=feature_group, popup_attribute='vis_data', color=color, edge_width=edge_width)
        m.add_to(folium_map)
    except Exception as e:
        logging.error(f"Error in {feature_group_name}: " + str(e))


def add_legend_to_map(folium_map: folium.Map) -> None:
    """Add legend to map.

    Args:
        folium_map (folium.Map): Map to add legend to.
    """
    with open(VISUALIZATION_LEGEND_PATH, 'rb') as f:
        import base64
        encoded = base64.b64encode(f.read()).decode('utf-8')
    FloatImage(f"data:image/png;base64,{encoded}", bottom=5, left=86).add_to(folium_map)


def show_preds(grapf_networkx: MultiDiGraph, preds: Tensor, name: str, popup: bool):
    cycle_graphs = divide_graphs(grapf_networkx, preds, popup)

    logging.debug("Plotting")
    prediction_map = folium.Map(tiles='cartodbpositron', location=[44.4949, 11.3426], zoom_start=12)
    for graph, graph_name in zip(cycle_graphs, ['True positive', 'True negative', 'False positive', 'False negative']):
        logging.debug(f"Adding {graph_name}")
        add_graph_to_map(graph, graph_name, prediction_map, 'red', 2, popup)

    logging.debug("Adding legend")
    add_legend_to_map(prediction_map)
    logging.debug("Saving")
    visualization_path = str((VISUALIZATION_OUTPUT_DIR / f"{name}.html").absolute())
    prediction_map.save(visualization_path)
    logging.info(f"Saved {visualization_path}")


def main(model_prediction):
    # TODO: remove this when we have a better way to handle multiple predictions ie saving the name of the graphml
    first_graph_name = "Bolonia__Wlochy_recent.xml"
    graphml_path = str((GRAPHML_VALIDATION_DATA_DIR / first_graph_name))
    graph_ox: nx.MultiDiGraph = ox.load_graphml(graphml_path)

    with open(Path(model_prediction), 'rb') as handle:
        multiple_prediction_logits = pickle.load(handle)
        # TODO: remove this when we have a better way to handle multiple predictions ie saving the name of the graphml
        prediction_logits: torch.Tensor = multiple_prediction_logits[0]
        predictions = prediction_logits.argmax(dim=1)

    show_preds(graph_ox, predictions, Path(model_prediction).name, True)


if __name__ == "__main__":
    for model_prediction in CLASSIFIER_OUTPUTS_SAVE_DIR.glob("*outputs.pkl"):
        main(model_prediction)
