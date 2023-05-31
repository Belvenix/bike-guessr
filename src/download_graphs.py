import argparse
import contextlib
import logging
import os
import re
from pathlib import Path

import networkx as nx
import osmnx as ox
from config import GRAPHML_TRAIN_DATA_DIR, GRAPHML_VALIDATION_DATA_DIR
from params import TRAINING_SET, VALIDATION_SET
from tqdm import tqdm


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='bikeguessr_download_cycleway')
    data_to_download = parser.add_mutually_exclusive_group(required=False)
    data_to_download.add_argument('-t', '--train', action='store_true')
    data_to_download.add_argument('-v', '--validation', action='store_true')

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

    place_iter.set_description(f"# {output} Downloading graphs")
    graphs_with_cycle = []
    for cf in new_filters:
        with contextlib.suppress(Exception):
            graphs_with_cycle.append(
                                    ox.graph.graph_from_polygon(
                                            polygon,
                                            network_type='bike',
                                            custom_filter=cf, 
                                            retain_all=True)
                                    )

    graph_without_cycle = ox.graph.graph_from_polygon(
        polygon, network_type='drive', retain_all=True)

    place_iter.set_description(f"# {output} Merging")
    previous = graph_without_cycle
    nx.set_edge_attributes(previous, 0, "label")
    for bike_graph in graphs_with_cycle:
        nx.set_edge_attributes(bike_graph, 1, "label")
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

    place_iter = tqdm(places_to_download, total=len(places_to_download))
    for place in place_iter:
        place_iter.set_description(
            f"# {place.split(',')[0]}")
        try:
            download_graph(place, target_dir, place_iter)
        except Exception as e:
            logging.warning(f'{place} was corrupted. Cause: {e} Skipping...')
