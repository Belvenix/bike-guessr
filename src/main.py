from linkprediction.local_random_walk import LRW
from linkprediction.resource_allocation import RA
from linkprediction.stochastic_block_model import SBM
from linkprediction.embedding_seal import runSEAL
from datamanipulation.divide_net import divide_net

import scipy.io

import numpy as np

filepath = 'src/data/jazz.mat'
mat = scipy.io.loadmat(filepath)
net = mat['net']

ratioTrain = 0.9

train, test = divide_net(net, ratioTrain)

nodedegree = np.sum(train, axis=1)
# tempauc = RA(train, test, nodedegree)
# print(f"RA auc is equal to: {tempauc}")
# tempauc = LRW(train, test, 3, 0.85)
# print(f"LRW with 3 steps auc is equal to: {tempauc}")
# tempauc = SBM(np.matrix(train), np.matrix(test), 12.0)
# print(f"SBM with 12 groups auc is equal to: {tempauc}")
# print("Running SEAL")
# runSEAL()
# print("Finished SEAL")