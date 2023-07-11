import argparse
import logging
import re
import time
import traceback
import typing as tp
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
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
    VISUALIZATION_OUTPUT_DIR,
)
from params import TEST_SET, TRAINING_SET, VALIDATION_SET
from tqdm import tqdm

# Define each label filter
CYCLELANE_FILTERS = L_FILTER + M_FILTER + B_FILTER
CYCLETRACK_FILTERS = T_FILTER + S_FILTER
CYCLEROAD_FILTERS = R_FILTER


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


def download_cycle_class(
        polygon, 
        filters: tp.List[str], 
        label: int
) -> tp.Tuple[tp.List[nx.Graph], tp.List[tp.Tuple[int,str]]]:
    from requests.exceptions import ConnectionError
    graph_cyclelanes, filter_edge_counts = [], []
    for cf in filters:
        try:           
            filtered_graph = ox.graph_from_polygon(
                                            polygon,
                                            custom_filter=cf, 
                                            retain_all=True,
                                            simplify=False)
            nx.set_edge_attributes(filtered_graph, label, "label")
            filter_edge_counts.append((len(filtered_graph.edges), cf))
            graph_cyclelanes.append(filtered_graph)
            # Max 1 request per second https://operations.osmfoundation.org/policies/nominatim/
            time.sleep(1)
        # That's for the case when there is no data in the graph with given filter
        except ValueError as e:
            logging.debug(f"Empty response for filter {cf}: {e}")
            filter_edge_counts.append((0, cf))
        except ConnectionError as e:
            logging.debug(f"Connection error for filter {cf}: {e}")
            time.sleep(10)
        except Exception as e:
            logging.error(f"Error while downloading graph with filter {cf}: {e}")
    return graph_cyclelanes, filter_edge_counts


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
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='osmnx')
        ox.utils.config(useful_tags_way=useful_tags_path)


    gdf = ox.geocoder.geocode_to_gdf(place)
    polygon = gdf['geometry'][0]

    # TODO: Histogram do tego oraz podział na kategorie
    # Problem jest taki, że żaden z tych tagów nie gwarantuje poprawnego podziału i jest redundantny
    # Kwestia jest taka, że te tagi mogą w bardzo prosty sposób nachodzić na siebie

    place_iter.set_description(f"# {output} Downloading graphs")

    # Download base graph
    cycleway_options = '(track|lane|shared_lane|share_busway|separate)'
    drive_graph_filters = (
        f'["highway"]["area"!~"yes"]["access"!~"private"]'
        f'["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|'
        f"escalator|footway|path|pedestrian|planned|platform|proposed|raceway|service|"
        f'steps|track"]'
        f'["motor_vehicle"!~"no"]["motorcar"!~"no"]'
        f'["service"!~"alley|driveway|emergency_access|parking|parking_aisle|private"]'
        f'["cycleway:left"!~"{cycleway_options}"]["cycleway:right"!~"{cycleway_options}"]'
    )
    graph_without_cycle = ox.graph_from_polygon(
        polygon, custom_filter=drive_graph_filters, retain_all=True, simplify=False)
    nx.set_edge_attributes(graph_without_cycle, 0, "label")
    nocycle_edge_count = len(graph_without_cycle.edges)

    # Retrieve graphs for each label
    # Since they can fail, we need to keep track of which filters failed
    graph_cyclelanes, cyclelanes_edge_counts = download_cycle_class(polygon, CYCLELANE_FILTERS, 1)
    graph_cycletracks, cycletracks_edge_counts = download_cycle_class(polygon, CYCLETRACK_FILTERS, 2)
    graph_cycleroads, cycleroads_edge_counts = download_cycle_class(polygon, CYCLEROAD_FILTERS, 3)
    graphs_with_cycle = graph_cyclelanes + graph_cycletracks + graph_cycleroads

    # Save edge count statistics
    all_edge_counts_and_names: tp.List[int] = [(nocycle_edge_count, "nocycle"), *cyclelanes_edge_counts, *cycletracks_edge_counts, *cycleroads_edge_counts]
    edge_counts = [x[0] for x in all_edge_counts_and_names]
    edge_count_names = [x[1] for x in all_edge_counts_and_names]
    edge_dict = {"name": edge_count_names, "count": edge_counts}

    edge_count_df = pd.DataFrame(edge_dict)
    edge_count_df.to_csv(VISUALIZATION_OUTPUT_DIR / f"{output}_edge_counts.csv", index=False)

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
    return edge_count_df


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

    sum_counts = np.zeros(38)
    place_iter = tqdm(places_to_download, total=len(places_to_download))
    for place in place_iter:
        place_iter.set_description(
            f"# {place.split(',')[0]}")
        try:
            counter = download_graph(place, target_dir, place_iter)
            sum_counts += counter['count'].to_numpy()
        except Exception as e:
            logging.warning(f'{place} was corrupted. Cause: {e} Skipping...')
            traceback.print_exc()

    if counter:
        pd.DataFrame({'all_counts': sum_counts, 'filters': counter['name']}) \
            .to_csv(VISUALIZATION_OUTPUT_DIR / "all_edge_counts.csv", index=False)
