import osmnx as ox
import networkx as nx
from datamanipulation.divide_net import divide_net
from linkprediction.resource_allocation import RA
from linkprediction.local_random_walk import LRW
from linkprediction.stochastic_block_model import SBM
import numpy as np
from tqdm import tqdm_notebook

wroclaw = ox.io.load_graphml('./src/data/data_train/Wroclaw_Polska_recent.xml')
nx.readwrite.write_edgelist(wroclaw, './src/data/data_train/Wroclaw_Polska_recent.csv')
sparse_matrix = nx.convert_matrix.to_scipy_sparse_array(wroclaw)

train, test = divide_net(sparse_matrix, .9)

#nodedegree = np.sum(train, axis=1)
#tempauc = RA(train, test, nodedegree)
#print(f"RA auc is equal to: {tempauc}")
#tempauc = LRW(train, test, 3, 0.85)
#print(f"LRW with 3 steps auc is equal to: {tempauc}")

print(f"Running SBM")
tempauc = SBM(np.matrix(train).astype(bool), np.matrix(test).astype(bool), 12.0)
print(f"SBM with 12 groups auc is equal to: {tempauc}")