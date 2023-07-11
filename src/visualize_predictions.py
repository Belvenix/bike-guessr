import logging
import typing as tp
from pathlib import Path

import dgl
import folium
import networkx as nx
import osmnx as ox
import torch
from config import (
    CLASSIFIER_TEST_DATA_PATH,
    CLASSIFIER_WEIGHTS_SAVE_DIR,
    GRAPHML_TEST_DATA_DIR,
    VISUALIZATION_LEGEND_PATH,
    VISUALIZATION_OUTPUT_DIR,
)
from folium.plugins import FloatImage
from networkx.classes.multidigraph import MultiDiGraph
from torch import Tensor
from tqdm import tqdm
from train import GraphConvolutionalNetwork, MaskedGraphConvolutionalNetwork, TrivialClassifier
from train_classifiers import load_encoder

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


# Define the dimensions
input_dim = 32
output_dim = 4
hidden_dim = 128


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
    return (preds > 0).nonzero().squeeze().tolist()


def divide_graphs(
        graph_networkx: MultiDiGraph, 
        preds: Tensor,
        popup: bool
) -> tp.Tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph]:
    """Divide graph into true positive, true negative, false positive and false negative.

    Args:
        graph_networkx (MultiDiGraph): Graph to divide.
        preds (Tensor): Predictions tensor.
        popup (bool): Whether to add popup to edge.

    Raises:
        ValueError: If graph labels are not binary.

    Returns:
        tp.Tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph]: 
            True positive, true negative, false positive and false negative graphs.
    """
    pred_ids = retrieve_cycle_indices(preds)

    cycle_true_positive: nx.Graph = nx.function.create_empty_copy(graph_networkx)
    cycle_true_negative: nx.Graph = nx.function.create_empty_copy(graph_networkx)
    cycle_false_positive: nx.Graph = nx.function.create_empty_copy(graph_networkx)
    cycle_false_negative: nx.Graph = nx.function.create_empty_copy(graph_networkx)

    for x in tqdm(set(graph_networkx.edges()), total=len(set(graph_networkx.edges()))):
        edge = graph_networkx[x[0]][x[1]][0]
        edge_attributes = edge.copy()
        label = int(edge_attributes['label'])
        is_pred = int(edge['idx']) in pred_ids

        # True positive
        if label > 0 and is_pred:
            add_edge_to_graph(cycle_true_positive, x, edge_attributes, popup)

        # True negative
        elif label == 0 and not is_pred:
            add_edge_to_graph(cycle_true_negative, x, edge_attributes, popup)

        # False positive
        elif label == 0 and is_pred:
            add_edge_to_graph(cycle_false_positive, x, edge_attributes, popup)

        # False negative
        elif label > 0 and not is_pred:  
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


def add_title_to_map(folium_map: folium.Map, title: str) -> None:
    title_html = f"""
        <h3 align="center" style="font-size:20px"><b>{title}</b></h3>
    """
    folium_map.get_root().html.add_child(folium.Element(title_html))


def show_preds(grapf_networkx: MultiDiGraph, preds: Tensor, name: str, popup: bool):
    assert grapf_networkx.number_of_edges() == len(preds)
    cycle_graphs = divide_graphs(grapf_networkx, preds, popup)
    graph_names = ['True positive', 'True negative', 'False positive', 'False negative']
    colors = ['green', 'blue', 'red', 'orange']

    logging.info("Plotting")
    prediction_map = folium.Map(tiles='cartodbpositron', location=[51.1079, 17.0385], zoom_start=12)

    for graph, graph_name, c in zip(cycle_graphs, graph_names, colors):
        logging.info(f"Adding {graph_name}")
        add_graph_to_map(graph, graph_name, prediction_map, c, 2, popup)
    folium.LayerControl().add_to(prediction_map)
    logging.info("Adding legend")
    add_legend_to_map(prediction_map)
    logging.info("Adding model name")
    add_title_to_map(prediction_map, name)
    logging.info("Saving")
    visualization_path = str((VISUALIZATION_OUTPUT_DIR / f"{name}.html").absolute())
    prediction_map.save(visualization_path)
    logging.info(f"Saved {visualization_path}")


def get_outputs(dgl_graph: dgl.DGLGraph) -> tp.List[Tensor]:
    logging.info("Loading encoder...")
    encoder = load_encoder()

    # MGCN with encoder
    mgcn = MaskedGraphConvolutionalNetwork(input_dim, hidden_dim, output_dim)
    mgcn_path = str((CLASSIFIER_WEIGHTS_SAVE_DIR / "best-cec-MaskedGraphConvolutionalNetwork.bin"))
    mgcn.load_state_dict(torch.load(mgcn_path))

    # MGCN without encoder
    mgcn_we = MaskedGraphConvolutionalNetwork(95, hidden_dim, output_dim)
    mgcn_we_path = str((CLASSIFIER_WEIGHTS_SAVE_DIR / "best-cec-MaskedGraphConvolutionalNetwork-without-encoding.bin"))
    mgcn_we.load_state_dict(torch.load(mgcn_we_path))

    # Trivial classifier
    trivial = TrivialClassifier(input_dim, hidden_dim, output_dim)
    trivial_path = str((CLASSIFIER_WEIGHTS_SAVE_DIR / "best-cec-TrivialClassifier.bin"))
    trivial.load_state_dict(torch.load(trivial_path))

    # GCN with encoder
    gcn = GraphConvolutionalNetwork(input_dim, hidden_dim, output_dim)
    gcn_path = str((CLASSIFIER_WEIGHTS_SAVE_DIR / "best-cec-GraphConvolutionalNetwork.bin"))
    gcn.load_state_dict(torch.load(gcn_path))

    # GCN without encoder
    gcn_we = GraphConvolutionalNetwork(95, hidden_dim, output_dim)
    gcn_we_path = str((CLASSIFIER_WEIGHTS_SAVE_DIR / "best-cec-GraphConvolutionalNetwork-without-encoding.bin"))
    gcn_we.load_state_dict(torch.load(gcn_we_path))

    for model in [mgcn, mgcn_we, trivial, gcn, gcn_we]:
        model.eval()

    with torch.no_grad():
        dgl_copy, dgl_encoded = dgl_graph.clone(), dgl_graph.clone()
        dgl_encoded.ndata['feat'] = encoder.encode(dgl_encoded, dgl_encoded.ndata['feat']).detach()
        mgcn_out, _ = mgcn(dgl_encoded, dgl_encoded.ndata['feat'])
        mgcn_we_out, _ = mgcn_we(dgl_copy, dgl_copy.ndata['feat'])
        trivial_out = trivial(dgl_encoded.ndata['feat'])
        gcn_out = gcn(dgl_encoded, dgl_encoded.ndata['feat'])
        gcn_we_out = gcn_we(dgl_copy, dgl_copy.ndata['feat'])
    return [mgcn_out, mgcn_we_out, trivial_out, gcn_out, gcn_we_out]


def get_noncec_outputs(dgl_graph: dgl.DGLGraph) -> tp.List[Tensor]:
    # MGCN without encoder
    mgcn_we = MaskedGraphConvolutionalNetwork(95, hidden_dim, output_dim)
    mgcn_we_path = str((CLASSIFIER_WEIGHTS_SAVE_DIR / "best-MaskedGraphConvolutionalNetwork-without-encoding.bin"))
    mgcn_we.load_state_dict(torch.load(mgcn_we_path))
    mgcn_we.eval()
    with torch.no_grad():
        dgl_copy = dgl_graph.clone()
        mgcn_we_out, _ = mgcn_we(dgl_copy, dgl_copy.ndata['feat'])
    return [mgcn_we_out]


def main():
    # TODO: remove this when we have a better way to handle multiple predictions ie saving the name of the graphml
    first_graph_name = "Wroclaw__Polska_recent.xml"
    graphml_path = str((GRAPHML_TEST_DATA_DIR / first_graph_name))
    graph_ox: nx.MultiDiGraph = ox.load_graphml(graphml_path)
    graph_dgl: dgl.DGLGraph = dgl.load_graphs(str(CLASSIFIER_TEST_DATA_PATH))[0][0]
    outputs = get_outputs(graph_dgl)
    predictions = torch.stack([torch.argmax(output, dim=1) for output in outputs], dim=0)
    for pred, name in zip(predictions, ['CEC MGCN', 'CEC MGCN without encoder', 'CEC Trivial', 'CEC GCN', 'CEC GCN without encoder']):
        show_preds(graph_ox, pred, name, True)


def main2():
    first_graph_name = "Wroclaw__Polska_recent.xml"
    graphml_path = str((GRAPHML_TEST_DATA_DIR / first_graph_name))
    graph_ox: nx.MultiDiGraph = ox.load_graphml(graphml_path)
    graph_dgl: dgl.DGLGraph = dgl.load_graphs(str(CLASSIFIER_TEST_DATA_PATH))[0][0]
    outputs = get_noncec_outputs(graph_dgl)
    predictions = torch.stack([torch.argmax(output, dim=1) for output in outputs], dim=0)
    show_preds(graph_ox, predictions[0], 'MGCN_without_encoder_wroclaw', True)


if __name__ == "__main__":
    main2()
