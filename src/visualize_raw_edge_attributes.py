import logging
import pickle
import typing as tp
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from config import (
    CLASSIFIER_OUTPUTS_SAVE_DIR,
    GRAPHML_TEST_DATA_DIR,
    GRAPHML_TRAIN_DATA_DIR,
    GRAPHML_VALIDATION_DATA_DIR,
)
from dgl.heterograph import DGLGraph
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG)


def extract_edge_attributes(
        merged_graph: nx.Graph,
) -> tp.Dict[str, tp.List[tp.Any]]:
    """Extracts all edge attributes from a merged graph.

    Args:
        merged_graph: The merged graph.

    Returns:
        A dictionary of all edge attributes.
    """
    all_attrs = {}
    merge_graph_edges = list(merged_graph.edges(data=True))
    for u, v, attrs in tqdm(merge_graph_edges, desc='Extracting edges to dictionary', total=len(merge_graph_edges)):
        # Exclude the desired attributes
        attrs = {k: v for k, v in attrs.items() if k not in ['label', 'name', 'idx', 'geometry']}
        for k, v in attrs.items():
            if k not in all_attrs:
                all_attrs[k] = []
            if isinstance(v, list):
                all_attrs[k].append(str(v))
            else:
                all_attrs[k].append(v)
    return all_attrs

def extract_edge_attributes_df(
        merged_graph: nx.Graph,
        all_attrs: tp.Dict[str, tp.List[tp.Any]]
) -> pd.DataFrame:
    all_attrs_keys = list(all_attrs.keys())
    main_attr_dict = {k: [] for k in all_attrs_keys}
    merge_graph_edges = list(merged_graph.edges(data=True))
    for u, v, attrs in tqdm(merge_graph_edges, desc='Extracting edges to df', total=len(merge_graph_edges)):
        # Exclude the desired attributes
        attrs = {k: v for k, v in attrs.items() if k not in ['label', 'name', 'idx', 'geometry']}
        for k in all_attrs_keys:
            if k in attrs:
                main_attr_dict[k].append(attrs[k])
            else:
                main_attr_dict[k].append(np.nan)
    main_dataframe = pd.DataFrame(main_attr_dict)
    return main_dataframe

def load_bikeguessr_graphs(
        directory: Path = None,
        output: Path = None
) -> tp.List[DGLGraph]:
    if output.exists():
        logging.info('loading bikeguessr graphs from pickle')
        with open(output, 'rb') as f:
            all_attrs, all_attrs_df = pickle.load(f)
        return all_attrs_df
    logging.info('load bikeguessr graphs')
    if directory is None:
        directory = Path(GRAPHML_TRAIN_DATA_DIR)
    
    found_files = list(directory.glob('*.xml'))
    logging.info(found_files)
    
    graphs = []
    for path in tqdm(found_files, desc='Loading graphml files'):
        graph: nx.Graph = ox.load_graphml(path)
        graphs.append(graph)

    logging.info('merging bikeguessr graphs')
    merged_graph = nx.compose_all(graphs)
    
    logging.info('extracting attributes from bikeguessr graphs')
    all_attrs = extract_edge_attributes(merged_graph)

    logging.info('extracting bikeguessr attributes to pandas DataFrame')
    all_attrs_df = extract_edge_attributes_df(merged_graph, all_attrs)

    with open(output, 'wb') as f:
        pickle.dump((all_attrs, all_attrs_df), f)
        
    
    logging.info('end load bikeguessr graphs')
    return all_attrs_df


if __name__ == "__main__":
    attr_dfs = []
    data_directories = [GRAPHML_TRAIN_DATA_DIR, GRAPHML_VALIDATION_DATA_DIR, GRAPHML_TEST_DATA_DIR]
    filenames = ['train-edges-df.pkl', 'validation-edges-df.pkl', 'test-edges-df.pkl']
    for directory, out_filename in zip(data_directories, filenames):
        output = CLASSIFIER_OUTPUTS_SAVE_DIR / out_filename
        attr_df = load_bikeguessr_graphs(directory=directory, output=output)
        attr_dfs.append(attr_df)
    logging.info('Describing dataframes')
    describe_filenames = ['train-edges-df-describe.csv', 'validation-edges-df-describe.csv', 'test-edges-df-describe.csv']
    for a, n in zip(attr_dfs, describe_filenames):
        logging.info(f'Describing {n}')
        a.describe().to_csv(CLASSIFIER_OUTPUTS_SAVE_DIR / n)
        a.describe(include='object').to_csv(CLASSIFIER_OUTPUTS_SAVE_DIR / n.replace('.csv', '-object.csv'))