import argparse
import contextlib
import logging
import re
import typing as tp
from pathlib import Path

import networkx as nx
import osmnx as ox
from config import (
    B_FILTER,
    GRAPHML_TEST_DATA_DIR,
    GRAPHML_TRAIN_DATA_DIR,
    GRAPHML_VALIDATION_DATA_DIR,
    L_FILTER,
    M_FILTER,
    R_FILTER,
    S_FILTER,
    T_FILTER,
)
from params import TEST_SET, TRAINING_SET, VALIDATION_SET
from tqdm import tqdm


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='bikeguessr_download_cycleway')
    data_to_download = parser.add_mutually_exclusive_group(required=False)
    data_to_download.add_argument('-t', '--train', action='store_true')
    data_to_download.add_argument('-v', '--validation', action='store_true')
    data_to_download.add_argument('-s', '--test', action='store_true')

    return parser.parse_args()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def replace_non_standard_letters(text):
    replacements = {
        "ć": "c",
        "ł": "l",
        "é": "l",
        "ś": "s",
        "ń": "n",
        "Ś": "S",
        "Ł": "L",
        "ó": "o",
        "ą": "a",
        "ź": "z",
        "ż": "z",
        "ę": "e",
        "ě": "e",
        "ô": "o",
        "ö": "o",
        "è": "e",
    }

    pattern = re.compile("|".join(re.escape(key) for key in replacements))
    replaced_text = pattern.sub(lambda match: replacements[match.group(0)], text)
    
    return replaced_text


def download_cycle_class(polygon, filters: tp.List[str], label: int):
    graph_cyclelanes = []
    for cf in filters:
        with contextlib.suppress(Exception):
            filtered_graph = ox.graph.graph_from_polygon(
                                            polygon,
                                            network_type='bike',
                                            custom_filter=cf, 
                                            retain_all=True)
            nx.set_edge_attributes(filtered_graph, label, "label")
            graph_cyclelanes.append(filtered_graph)
    return graph_cyclelanes

def download_graph(place: str, target_dir: Path, place_iter: tqdm):
    place_parts = place.split(',')
    if not len(place_parts) >= 1:
        raise ValueError("Place should consist of at least two parts: city and country")
    output = place_parts[0] + "_" + place_parts[-1]+"_recent"
    output = output.replace(' ', "_")
    output = replace_non_standard_letters(output)

    target_filepath = target_dir / f'{output}.xml'

    if target_filepath.exists():
        place_iter.set_description(f"# {output} Skipping...")
        return

    useful_tags_path = ['bridge', 'tunnel', 'oneway', 'lanes', 'ref', 'name',
                        'highway', 'maxspeed', 'service', 'access', 'area',
                        'landuse', 'width', 'est_width', 'junction', 'surface',
                        'bicycle', 'cycleway', 'busway', 'sidewalk', 'psv']
    try:
        ox.settings.useful_tags_path.extend(useful_tags_path)
    except AttributeError:
        ox.utils.config(useful_tags_way=useful_tags_path)


    gdf = ox.geocoder.geocode_to_gdf(place)
    polygon = gdf['geometry'][0]

    # TODO: Histogram do tego oraz podział na kategorie
    # Problem jest taki, że żaden z tych tagów nie gwarantuje poprawnego podziału i jest redundantny
    # Kwestia jest taka, że te tagi mogą w bardzo prosty sposób nachodzić na siebie

    place_iter.set_description(f"# {output} Downloading graphs")

    # Download base graph
    graph_without_cycle = ox.graph.graph_from_polygon(
        polygon, network_type='drive', retain_all=True)
    nx.set_edge_attributes(graph_without_cycle, 0, "label")
    
    # Define each label filter
    cyclelane_filters = L_FILTER + M_FILTER + B_FILTER
    cycletrack_filters = T_FILTER + S_FILTER
    cycleroad_filters = R_FILTER
    
    # Retrieve graphs for each label
    graph_cyclelanes = download_cycle_class(polygon, cyclelane_filters, 1)
    graph_cycletracks = download_cycle_class(polygon, cycletrack_filters, 2)
    graph_cycleroads = download_cycle_class(polygon, cycleroad_filters, 3)
    graphs_with_cycle = graph_cyclelanes + graph_cycletracks + graph_cycleroads

    # Merge graphs
    place_iter.set_description(f"# {output} Merging")
    previous = graph_without_cycle
    for bike_graph in graphs_with_cycle:
        merged_graph = nx.compose(previous, bike_graph)
        previous = merged_graph

    merged_graph_copy = merged_graph.copy()

    edge_id = 0
    for edge in merged_graph.edges():
        for connection in merged_graph[edge[0]][edge[1]]:
            for _, _ in merged_graph[edge[0]][edge[1]][connection].items():
                graph_edge = merged_graph_copy[edge[0]][edge[1]][connection]
                graph_edge['idx'] = edge_id
        edge_id += 1

    place_iter.set_description(f"# {output} Saving")
    merged_graph = ox.utils_graph.remove_isolated_nodes(merged_graph_copy)
    merged_graph.name = output
    ox.save_graphml(merged_graph, filepath=target_filepath)


if __name__ == "__main__":
    args = build_args()
    places_to_download = []
    
    target_dir = Path(".")
    if args.train:
        target_dir = GRAPHML_TRAIN_DATA_DIR
        places_to_download = TRAINING_SET
    if args.validation:
        target_dir = GRAPHML_VALIDATION_DATA_DIR
        places_to_download = VALIDATION_SET
    if args.test:
        target_dir = GRAPHML_TEST_DATA_DIR
        places_to_download = TEST_SET

    place_iter = tqdm(places_to_download, total=len(places_to_download))
    for place in place_iter:
        place_iter.set_description(
            f"# {place.split(',')[0]}")
        try:
            download_graph(place, target_dir, place_iter)
        except Exception as e:
            logging.warning(f'{place} was corrupted. Cause: {e} Skipping...')
