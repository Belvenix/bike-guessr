import argparse
import logging
import pickle
import typing as tp
from pathlib import Path

import folium
import networkx as nx
import osmnx as ox
import torch
from dgl.data.utils import load_graphs
from folium.plugins import FloatImage
from networkx.classes.multidigraph import MultiDiGraph
from torch import Tensor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def _divide_graphs(
        grapf_networkx: MultiDiGraph, 
        preds: Tensor,
        popup: bool
    ) -> tp.Tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph]:
    pred_ids = ((preds == True).nonzero(as_tuple=True)[0]).tolist()

    cycle_true_positive: nx.Graph = nx.function.create_empty_copy(grapf_networkx)
    cycle_true_negative: nx.Graph = nx.function.create_empty_copy(grapf_networkx)
    cycle_false_positive: nx.Graph = nx.function.create_empty_copy(grapf_networkx)
    cycle_false_negative: nx.Graph = nx.function.create_empty_copy(grapf_networkx)

    for x in tqdm(set(grapf_networkx.edges()), total=len(set(grapf_networkx.edges()))):
        edge = grapf_networkx[x[0]][x[1]][0]
        dif_attributes = edge.copy()
        label = int(dif_attributes['label'])
        is_pred = int(edge['idx']) in pred_ids

        # True positive
        if label == 1 and is_pred:
            dif_attributes['vis_data'] = {
                'href': f"https://www.openstreetmap.org/way/{edge['osmid']}"
            }
            if not popup:
                dif_attributes = None
            cycle_true_positive.add_edges_from([(x[0], x[1], dif_attributes)])

        # True negative
        elif label == 0 and not is_pred:
            dif_attributes['vis_data'] = {
                'href': f"https://www.openstreetmap.org/way/{edge['osmid']}"
            }
            if not popup:
                dif_attributes = None
            cycle_true_negative.add_edges_from([(x[0], x[1], dif_attributes)])

        # False positive
        elif label == 0 and is_pred:
            dif_attributes['vis_data'] = {
                'href': f"https://www.openstreetmap.org/way/{edge['osmid']}"
            }
            if not popup:
                dif_attributes = None
            cycle_false_positive.add_edges_from(
                [(x[0], x[1], dif_attributes)])

        # False negative
        elif label == 1 and not is_pred:  
            dif_attributes['vis_data'] = {
                'href': f"https://www.openstreetmap.org/way/{edge['osmid']}"
            }
            if not popup:
                dif_attributes = None
            cycle_false_negative.add_edges_from([(x[0], x[1], dif_attributes)])

        else:
            raise ValueError(f"Edge label should be 0 or 1 and is {int(dif_attributes['label'])}")
    return cycle_true_positive, cycle_true_negative, cycle_false_positive, cycle_false_negative


def _add_graph_to_map(
        graph: nx.Graph,
        feature_group_name: str,
        folium_map: folium.Map,
        color: str,
        edge_width: int,
        popup: bool
    ) -> None:
    try:
        feature_group = folium.FeatureGroup(name=feature_group_name)
        logging.info(f'{feature_group_name} edges: {graph.number_of_edges()}')
        if not popup:
            m = ox.plot_graph_folium(graph, graph_map=feature_group, color=color, edge_width=edge_width)
        else:
            m = ox.plot_graph_folium(graph, graph_map=feature_group, popup_attribute='vis_data', color=color, edge_width=edge_width)
        m.add_to(folium_map)
    except Exception as e:
        logging.error(f"Error in {feature_group_name}: " + str(e))


def _add_legend_to_map(folium_map: folium.Map) -> None:
    with open("./src/visualization/imgs/bicycle_prediction_legend.png", 'rb') as f:
        import base64
        encoded = base64.b64encode(f.read()).decode('utf-8')
    FloatImage(f"data:image/png;base64,{encoded}", bottom=5, left=86).add_to(folium_map)


def show_preds(grapf_networkx: MultiDiGraph, mask: Tensor, preds: Tensor, name: str, popup: bool, save_folder: str):
    assert grapf_networkx.number_of_edges() == mask.shape[0]

    cycle_graphs = _divide_graphs(grapf_networkx, preds, popup)

    logging.debug("Plotting")
    prediction_map = folium.Map(tiles='cartodbpositron', location=[44.4949, 11.3426], zoom_start=12)
    for graph, graph_name in zip(cycle_graphs, ['True positive', 'True negative', 'False positive', 'False negative']):
        logging.debug(f"Adding {graph_name}")
        _add_graph_to_map(graph, graph_name, prediction_map, 'red', 2, popup)

    logging.debug("Adding legend")
    _add_legend_to_map(prediction_map)
    logging.debug("Saving")
    prediction_map.save(f"{save_folder}{name}_{popup!s}_full.html")
    logging.info(f"Saved {save_folder}{name}_{popup!s}_full.html")


def main(args):
    mask_choices = {
        "train": "train_mask",
        "dev": "dev_mask",
        "test": "test_mask"
    }
    mask_to_visualize = mask_choices[args.mask]
    graph_ox = ox.io.load_graphml(args.raw + args.ox)
    # list of graphs
    dgl_graphs = load_graphs(args.transformed + args.dgl)[0][0]

    with open(Path(args.pred), 'rb') as handle:
        predictions = pickle.load(handle)[0]
        predictions = predictions.max(1)[0].type_as(dgl_graphs.ndata[mask_to_visualize])

    all_elem = torch.ones(predictions.shape[0])
    show_preds(graph_ox, all_elem, predictions, Path(args.pred).name, True, args.visualise)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BikeGuessr Visualize Predictions Script")
    parser.add_argument("--raw", type=str, default="../docker_data/data/data_val/", help="Path to the raw data folder")
    parser.add_argument("--transformed", type=str, default="../docker_data/data/data_transformed/", help="Path to the transformed data folder")
    parser.add_argument("--ox", type=str, default="Bolonia__Wlochy_recent.xml", help="Name of the OSMnx graph file")
    parser.add_argument("--dgl", type=str, default="validation.bin", help="Name of the DGL graph file")
    parser.add_argument("--pred", type=str, default="../docker_data/data/outputs/gnn-classifier-outputs.pkl", help="Name of the prediction file")
    parser.add_argument("--visualise", type=str, default="../docker_data/data/visualizations/", help="Path to the visualizations folder")
    args = parser.parse_args()
    for model_prediction in Path("./src/docker_data/data/outputs/").glob("*outputs.pkl"):
        args.pred = model_prediction
        main(args)
