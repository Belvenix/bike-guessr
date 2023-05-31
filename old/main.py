from linkprediction.local_random_walk import LRW
from linkprediction.resource_allocation import RA
from linkprediction.stochastic_block_model import SBM
from linkprediction.embedding_seal import runSEAL
from datamanipulation.divide_net import divide_net

from urban import ipdc
import traceback
import threading

import csv
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

CITY_TIMEOUT = 2 * 3600

if __name__ == "__main__":

    cities = {}

    with open('./src/cities_w_cords.csv') as f:
        csvreader = csv.DictReader(f, delimiter=';')
        for row in csvreader:
            cities[row['placeid']] = {}
            for field in csvreader.fieldnames[1:]:
                cities[row['placeid']][field] = row[field]   

    city_iter = tqdm(cities, total=len(cities))
    for city in city_iter:
        try:
            city_coords = cities[city]['lat'], cities[city]['lng']
            ipdc(city, city_iter, city_coords)
            # t = threading.Thread(target=ipdc, args=[city, city_iter, city_coords])
            # t.start()
            # t.join(CITY_TIMEOUT)
                
            #ipdc(city, city_iter, city_cords)
        except Exception as e:
            print(f"There was a problem with {city}. Skipping...")
            traceback.print_exc()