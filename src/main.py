from linkprediction.local_random_walk import LRW
from linkprediction.resource_allocation import RA
from linkprediction.stochastic_block_model import SBM

import scipy.io

import numpy as np

def DivideNet(net, ratioTrain):
    net = net.todense()
    # convert to upper triangular matrix
    net = np.triu(net) - np.diag(np.diag(net))
    num_testlinks = int(np.ceil((1-ratioTrain) * np.count_nonzero(net)))
    # find all edges in the network
    linklist = np.transpose(np.nonzero(net))
    links = np.zeros((num_testlinks, 2))
    i = 0
    while i < num_testlinks:
        if linklist.shape[0] <= 2:
            break
        index_link = int(np.random.randint(linklist.shape[0], size=1))
        uid1, uid2 = linklist[index_link]
        net[uid1, uid2] = 0
        links[i] = [uid1, uid2]
        i += 1
    links = links.astype(int)
    test = np.zeros(net.shape)
    test[links[:, 0], links[:, 1]] = 1
    train = net + np.transpose(net)
    test = test + np.transpose(test)
    return train, test


filepath = 'src/data/jazz.mat'
mat = scipy.io.loadmat(filepath)
net = mat['net']

ratioTrain = 0.9

train, test = DivideNet(net, ratioTrain)

nodedegree = np.sum(train, axis=1)
#tempauc = RA(train, test, nodedegree)
#print(f"RA auc is equal to: {tempauc}")
#tempauc = LRW(train, test, 3, 0.85)
#print(f"LRW with 3 steps auc is equal to: {tempauc}")
tempauc = SBM(np.matrix(train), np.matrix(test), 12.0)
print(f"SBM with 12 groups auc is equal to: {tempauc}")