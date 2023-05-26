import argparse
import pickle

import networkx as nx
import osmnx as ox
import torch
from dgl.data.utils import load_graphs
from folium.plugins import FloatImage
from networkx.classes.multidigraph import MultiDiGraph
from torch import Tensor
from tqdm import tqdm


def _show_preds(grapf_networkx: MultiDiGraph, mask: Tensor, preds: Tensor, name: str, popup: bool):
    assert grapf_networkx.number_of_edges() == mask.shape[0]

    mask_ids = ((mask == True).nonzero(as_tuple=True)[0]).tolist()
    pred_ids = ((preds == True).nonzero(as_tuple=True)[0]).tolist()

    year = str(2022)
    dif_masked_cycle = nx.create_empty_copy(grapf_networkx)
    dif_masked_road = dif_masked_cycle.copy()
    dif_masked_different = dif_masked_cycle.copy()

    diff_unmasked = dif_masked_cycle.copy()

    for x in tqdm(set(grapf_networkx.edges()), total=len(set(grapf_networkx.edges()))):
        edge = grapf_networkx[x[0]][x[1]][0]
        # print(edge)
        if int(edge['idx']) in mask_ids:
            dif_attributes = edge.copy()
            if int(dif_attributes['label']) == 1 and int(edge['idx']) in pred_ids:  # if cycle
                vis_data = {
                    'href': f"https://www.openstreetmap.org/way/{edge['osmid']}",
                    'years': ['cycle', 'masked'],
                    'data': {}
                }
                vis_data['data'] = {year: [dif_attributes['label'], True]}
                dif_attributes['vis_data'] = vis_data
                if not popup:
                    dif_attributes = None
                dif_masked_cycle.add_edges_from([(x[0], x[1], dif_attributes)])
            elif int(edge['idx']) in pred_ids:
                vis_data = {
                    'href': f"https://www.openstreetmap.org/way/{edge['osmid']}",
                    'years': ['cycle', 'masked'],
                    'data': {}
                }
                vis_data['data'] = {year: [dif_attributes['label'], True]}
                dif_attributes['vis_data'] = vis_data
                if not popup:
                    dif_attributes = None
                dif_masked_different.add_edges_from(
                    [(x[0], x[1], dif_attributes)])
            else:  # if not cycle
                vis_data = {
                    'href': f"https://www.openstreetmap.org/way/{edge['osmid']}",
                    'years': ['cycle', 'masked'],
                    'data': {}
                }
                vis_data['data'] = {year: [dif_attributes['label'], True]}
                dif_attributes['vis_data'] = vis_data
                if not popup:
                    dif_attributes = None
                dif_masked_road.add_edges_from([(x[0], x[1], dif_attributes)])
        else:
            vis_data = {
                'href': f"https://www.openstreetmap.org/way/{edge['osmid']}",
                'years': ['cycle', 'masked'],
                'data': {}
            }
            dif_attributes = edge.copy()
            vis_data['data'] = {year: [dif_attributes['label'], False]}

            dif_attributes['vis_data'] = vis_data
            if not popup:
                dif_attributes = None
            diff_unmasked.add_edges_from([(x[0], x[1], dif_attributes)])
    try:
        if not popup:
            m = ox.plot_graph_folium(
                dif_masked_different, color="orange", edge_width=1)
        else:
            m = ox.plot_graph_folium(
                dif_masked_different, popup_attribute='vis_data', color="orange", edge_width=2)
    except Exception as e:
        print("Error in pred as noncycleway" + str(e))
    try:
        if not popup:
            m = ox.plot_graph_folium(diff_unmasked, graph_map=m, color="blue", edge_width=1)
        else:
            m = ox.plot_graph_folium(
                diff_unmasked, popup_attribute='vis_data', graph_map=m, color="blue", edge_width=1)
    except Exception as e:
        print("Error in pred as unmasked" + str(e))
    try:
        if not popup:
            m = ox.plot_graph_folium(
                dif_masked_cycle, graph_map=m, color="green", edge_width=1)
        else:
            m = ox.plot_graph_folium(
                dif_masked_cycle, popup_attribute='vis_data', graph_map=m, color="green", edge_width=2)
    except Exception as e:
        print("Error in pred as true cycle" + str(e))
    try:
        if not popup:
            m = ox.plot_graph_folium(
                dif_masked_road, graph_map=m, color="red", edge_width=1)
        else:
            m = ox.plot_graph_folium(
                dif_masked_road, popup_attribute='vis_data', graph_map=m, color="red", edge_width=2)
    except Exception as e:
        print("Error in pred as new cycle" + str(e))

    FloatImage("./imgs/legend_full.jpg", bottom=5, left=86).add_to(m)
    m.save(f"../docker_data/data/visualizations/{name}_{str(popup)}_full.html")



def main(raw_data_directory, transformed_data_directory, ox_graph_name, dgl_graph_name, prediction_file, mask_to_visualize):
    graph_ox = ox.io.load_graphml(raw_data_directory + ox_graph_name)
    # list of graphs
    dgl_graphs = load_graphs(transformed_data_directory + dgl_graph_name)[0][0]

    with open(prediction_file, 'rb') as handle:
        predictions = pickle.load(handle)[0]
        print(predictions)
        predictions = predictions.max(1)[1].type_as(dgl_graphs.ndata[mask_to_visualize])

    all_elem = torch.ones(predictions.shape[0])
    _show_preds(graph_ox, all_elem, predictions, prediction_file.split('.')[0], True)

if __name__ == "__main__":
    mask_choices = {
        "train": "train_mask",
        "dev": "dev_mask",
        "test": "test_mask"
    }
    parser = argparse.ArgumentParser(description="BikeGuessr Visualize Predictions Script")
    parser.add_argument("--raw", type=str, default="../docker_data/data/data_val/", help="Path to the raw data folder")
    parser.add_argument("--transformed", type=str, default="../docker_data/data/data_transformed/", help="Path to the transformed data folder")
    parser.add_argument("--ox", type=str, default="Bolonia_Włochy_recent.xml", help="Name of the OSMnx graph file")
    parser.add_argument("--dgl", type=str, default="validation.bin", help="Name of the DGL graph file")
    parser.add_argument("--pred", type=str, default="../docker_data/data/outputs/classifier-outputs.pkl", help="Name of the prediction file")
    parser.add_argument("--mask", type=str, default="test", choices=mask_choices.keys(), help="Mask to visualize (train, dev, or test)")
    args = parser.parse_args()

    main(args.raw, args.transformed, args.ox, args.dgl, args.pred, mask_choices[args.mask])
