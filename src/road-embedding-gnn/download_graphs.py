import argparse
import logging

import networkx as nx
import osmnx as ox
from tqdm import tqdm
import os
from params import TRAINING_SET, VALIDATION_SET


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='bikeguessr_download_cycleway')
    data_to_download = parser.add_mutually_exclusive_group(required=False)
    data_to_download.add_argument('-a', '--all', action='store_true')
    data_to_download.add_argument('-w', '--wroclaw', action='store_true')
    data_to_download.add_argument('-g', '--gdansk', action='store_true')
    data_to_download.add_argument('-wa', '--walbrzych', action='store_true')
    data_to_download.add_argument('-t', '--train', action='store_true')
    data_to_download.add_argument('-v', '--validation', action='store_true')

    return parser.parse_args()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    

def download_graph(place: str, target_dir: str, place_iter: tqdm):
    place_parts = place.split(',')
    assert len(place_parts) >= 1
    output = place_parts[0] + "_" + place_parts[-1]+"_recent"
    output = output.replace(' ', "")

    useful_tags_path = ['bridge', 'tunnel', 'oneway', 'lanes', 'ref', 'name',
                        'highway', 'maxspeed', 'service', 'access', 'area',
                        'landuse', 'width', 'est_width', 'junction', 'surface',
                        'bicycle', 'cycleway', 'busway', 'sidewalk', 'psv']
    ox.utils.config(useful_tags_way=useful_tags_path)



    gdf = ox.geocoder.geocode_to_gdf(place)#, by_osmid=True)
    polygon = gdf['geometry'][0]

    new_filters = [
        '["highway"~"cycleway"]', 
        '["bicycle"~"(designated|yes|use_sidepath)"]',
        '["cycleway"~"(lane|track)"]',
        '["cycleway:left"~"(lane|track)"]',
        '["cycleway:right"~"(lane|track)"]',
        '["cycleway:both"~"(lane|track)"]'
    ]

    place_iter.set_description(f"# {place.split(',')[0]} Downloading graphs")
    graphs_with_cycle = []
    for cf in new_filters:
        try:
            graphs_with_cycle.append(
                                    ox.graph.graph_from_polygon(
                                            polygon,
                                            network_type='bike',
                                            custom_filter=cf, 
                                            retain_all=True)
                                    )
        except:
            pass
    graph_without_cycle = ox.graph.graph_from_polygon(
        polygon, network_type='drive', retain_all=True)

    place_iter.set_description(f"# {place.split(',')[0]} Merging")
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
            for _, _ in merged_graph[edge[0]][edge[1]][connection].items():
                graph_edge = merged_graph_copy[edge[0]][edge[1]][connection]
                graph_edge['idx'] = edge_id
        edge_id += 1

    place_iter.set_description(f"# {place.split(',')[0]} Saving")
    merged_graph = ox.utils_graph.remove_isolated_nodes(merged_graph_copy)
    merged_graph.name = output
    ox.save_graphml(merged_graph, filepath="./data/{}/{}.xml".format(target_dir, output))


if __name__ == "__main__":
    args = build_args()
    places_to_download = []
    if args.all:
        places_to_download = ["Copenhagen Municipality, Region Stołeczny, Dania",
                            "Gmina Aarhus, Jutlandia Środkowa, Dania",
                            "Odense Kommune, Dania Południowa, Dania",
                            "Gmina Aalborg, Jutlandia Północna, Dania",
                            "Frederiksberg Municipality, Region Stołeczny, Dania",
                            "Gimina Gentofte, Region Stołeczny, Dania",
                            "Gmina Esbjerg, Dania Południowa, Dania",
                            "Gmina Gladsaxe, Region Stołeczny, Dania",
                            "Gmina Holstebro, Holstebro Municipality, Dania",
                            "Gmina Randers, Jutlandia Środkowa, Dania",
                            "Gmina Kolding, Dania Południowa, Dania",
                                    ]
            
    if args.wroclaw:
        places_to_download = ["Wrocław, województwo dolnośląskie, Polska"]
    if args.gdansk:
        places_to_download = ["Gdańsk, województwo pomorskie, Polska"]
    if args.walbrzych:
        places_to_download = ["Wałbrzych, województwo dolnośląskie, Polska"]
    
    target_dir = "data_train"
    if args.train:
        target_dir= "data_train"
        places_to_download = TRAINING_SET
        if not os.path.exists("./data/data_train"):
            os.makedirs("./data/data_train")
    if args.validation:
        target_dir = "data_val"
        places_to_download = VALIDATION_SET
        if not os.path.exists("./data/data_val"):
            os.makedirs("./data/data_val")


    place_iter = tqdm(places_to_download, total=len(places_to_download))
    for place in place_iter:
        place_iter.set_description(
            f"# {place.split(',')[0]}")
        try:
            download_graph(place, target_dir, place_iter)
        except:
            logging.warning(f'{place} was corrupted. Skipping...')
