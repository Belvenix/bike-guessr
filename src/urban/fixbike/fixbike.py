from .data_import import read_csv, create_nx_objects, transform_nx_and_save, import_check
from .identify_prioritize import read_transformed_data, identify_gaps, remove_extra_gaps, prioritize, save_results, check_ip
from .decluster_classify import import_identify_prioritize_data, decluster, plot_folium_declustered_gaps, manual_classify

from tqdm import tqdm
from datetime import datetime
from typing import Tuple

import os
import warnings
warnings.filterwarnings('ignore')

def import_city_data(city_name: str, tqdm_iterator: tqdm):
    #if import_check(city_name):
    #    return
    tqdm_iterator.set_description(f"# {datetime.now().isoformat()} Importing - Reading {city_name} csv")
    all_edges, all_nodes = read_csv(city_name)
    tqdm_iterator.set_description(f"# {datetime.now().isoformat()} Importing - Creating {city_name} nx")
    H = create_nx_objects(city_name, all_edges, all_nodes)
    tqdm_iterator.set_description(f"# {datetime.now().isoformat()} Importing - Transforming and saving {city_name} nx")
    transform_nx_and_save(city_name, H)


def identify_and_prioritize(city_name: str, tqdm_iterator: tqdm) -> None:
    if check_ip(city_name):
        return
    tqdm_iterator.set_description(f"# {datetime.now().isoformat()} I&P - Reading {city_name} loaded data")
    B, h, eids_conv, ebc, ced = read_transformed_data(city_name)
    tqdm_iterator.set_description(f"# {datetime.now().isoformat()} I&P - Identifing {city_name} gaps")
    mygaps = identify_gaps(city_name, h, eids_conv, ced)
    tqdm_iterator.set_description(f"# {datetime.now().isoformat()} I&P - Removing {city_name} extra gaps")
    mygaps = remove_extra_gaps(mygaps, B)
    tqdm_iterator.set_description(f"# {datetime.now().isoformat()} I&P - Prioritizing {city_name} gaps")
    mygaps = prioritize(mygaps, ebc, h)
    save_results(city_name, mygaps)


def decluster_and_classify(city_name: str, tqdm_iterator: tqdm, city_coords: Tuple[float, float]):
    tqdm_iterator.set_description(f"# {datetime.now().isoformat()} D&C - Reading {city_name} I&P")
    mygaps, H, h, ebc, ced, ted = import_identify_prioritize_data(city_name)
    tqdm_iterator.set_description(f"# {datetime.now().isoformat()} D&C - Declustering {city_name} I&P")
    gap_dec = decluster(city_name, mygaps, h, ebc, ced)
    try:
        plot_folium_declustered_gaps(city_name, H, ced, ted, gap_dec, city_coords)
    except Exception as e:
        print(f"Some error occured during plotting... Skipping...\n{e}")


def ipdc(city_name: str, tqdm_iterator: tqdm, city_coords: Tuple[float, float]):
    tqdm_iterator.set_description(f"# {datetime.now().isoformat()} Importing {city_name} data")
    import_city_data(city_name, tqdm_iterator)
    tqdm_iterator.set_description(f"# {datetime.now().isoformat()} Identifing and prioritizing {city_name} gaps")
    identify_and_prioritize(city_name, tqdm_iterator)
    tqdm_iterator.set_description(f"# {datetime.now().isoformat()} Declustering {city_name} gaps")
    decluster_and_classify(city_name, tqdm_iterator, city_coords)