import argparse
import logging
import constants

import networkx as nx
import osmnx as ox
from tqdm import tqdm

from params import TRAINING_SET, VALIDATION_SET


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='bikeguessr_download_cycleway')
    data_to_download = parser.add_mutually_exclusive_group(required=False)
    data_to_download.add_argument('-a', '--all', action='store_true')
    data_to_download.add_argument('-w', '--wroclaw', action='store_true')
    data_to_download.add_argument('-g', '--gdansk', action='store_true')
    data_to_download.add_argument('-wa', '--walbrzych', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-v', '--validation', action='store_true')

    return parser.parse_args()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    

def download_graph(place: str, tqdm_iterator: tqdm, target_dir: str):
    place_iter.set_description(f"# {place}")
    place_parts = place.split(',')
    assert len(place_parts) >= 1
    output = place_parts[0] + "_" + place_parts[-1]+"_recent"
    #output = place_parts[0] + "_" + place_parts[-1]+"_recent"
    output = output.replace(' ', "")
    gdf = ox.geocoder.geocode_to_gdf(place)#, by_osmid=True)
    polygon = gdf['geometry'][0]
    filters = ['["highway"~"cycleway"]', '["bicycle"~"designated"]', '["bicycle"~"permissive"]',
         '["bicycle"~"yes"]','["cycleway"~"lane"]','["cycleway"~"track"]']

    tqdm_iterator.set_description(f"# {place} downloading cycle")
    graphs_with_cycle = []
    for cf in filters:
        try:
            tqdm_iterator.set_description(f"# {place} downloading cycle with filter {cf}")
            graphs_with_cycle.append(
                                    ox.graph.graph_from_polygon(
                                            polygon,
                                            network_type='bike',
                                            custom_filter=cf, 
                                            retain_all=True)
                                    )
        except:
            logging.warning(f"Downloading cycle for {place} and filter {cf} was corrupted. Skipping")
    tqdm_iterator.set_description(f"# {place} downloading non cycle")
    graph_without_cycle = ox.graph.graph_from_polygon(
        polygon, network_type='drive', retain_all=True)

    tqdm_iterator.set_description(f"# {place} merging")
    previous = graph_without_cycle
    nx.set_edge_attributes(previous, 0, "label")
    for bike_graph in graphs_with_cycle:
        nx.set_edge_attributes(bike_graph, 1, "label")
        merged_graph = nx.compose(previous, bike_graph)
        previous = merged_graph

    merged_graph_copy = merged_graph.copy()

    edge_id = 0
    for edge in merged_graph.edges():
        for connection in merged_graph[edge[0]][edge[1]].keys():
            for key, val in merged_graph[edge[0]][edge[1]][connection].items():
                graph_edge = merged_graph_copy[edge[0]][edge[1]][connection]
                graph_edge['idx'] = edge_id
        edge_id += 1

    tqdm_iterator.set_description(f"# {place} saving")
    merged_graph = ox.utils_graph.remove_isolated_nodes(merged_graph_copy)
    merged_graph.name = output
    ox.save_graphml(merged_graph, filepath="./data/{}/{}.xml".format(target_dir, output))


if __name__ == "__main__":
    args = build_args()
    places_to_download = []
    if args.all:
        places_to_download = constants.ALL
    elif args.wroclaw:
        places_to_download = constants.WROCLAW
    elif args.gdansk:
        places_to_download = constants.GDANSK
    elif args.walbrzych:
        places_to_download = constants.WALBRZYCH
    elif args.netherlands:
        places_to_download = constants.NETHERLANDS
    else:
        places_to_download = constants.DENMARK
    
    target_dir = "data_train"
    if args.train:
        target_dir = "data_train"
        places_to_download = constants.TRAINING_SET
    if args.validation:
        target_dir = "data_val"
        places_to_download = constants.VALIDATION_SET


    place_iter = tqdm(places_to_download, total=len(places_to_download))
    for place in place_iter:
        place_iter.set_description(
            f"# {place}")
        try:
            download_graph(place, target_dir)
        except:
            logging.warning(f'{place} was corrupted. Skipping...')
