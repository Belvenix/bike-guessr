# system
import os
import pickle

# data
import math
import numpy as np
import pandas as pd
from ast import literal_eval # to convert str into lists (imported osmid data)

# network
import networkx as nx
import igraph as ig

def check_ip(city_name) -> bool:
    return os.path.exists(f"./src/data/{city_name}/pickle/mygaps.pickle")

# computes pathlength by nx - handling error message if nodes are not connected/not part of the network
def pathlength_if_connected(my_nw, my_o, my_d):
    try:
        return(nx.dijkstra_path_length(my_nw, my_o, my_d, weight = "length"))
    except:
        return(math.inf)
    
# get list of edge coordinates for plotting from list of nx edge ids:
def get_path_coords(my_path, my_coorddict):
    pathcoords = []
    for edge_id in my_path:
        edge_coords = [(c[1], c[0]) for c in my_coorddict[tuple(sorted(edge_id))].coords]
        pathcoords.append(edge_coords)
    return(pathcoords) 


def read_transformed_data(city_name):
    # IMPORT OBJECTS FROM PREVIOUS STEPS

    H = nx.read_gpickle(f"./src/data/{city_name}/pickle/H.gpickle")
    B = nx.read_gpickle(f"./src/data/{city_name}/pickle/B.gpickle")
    h = ig.Graph.Read_Pickle(f"./src/data/{city_name}/pickle/h.pickle")

    nids_conv = pd.read_pickle(f"./src/data/{city_name}/pickle/nids_conv.pickle")
    eids_conv = pd.read_pickle(f"./src/data/{city_name}/pickle/eids_conv.pickle")

    ebc = pd.read_pickle(f"./src/data/{city_name}/pickle/ebc.pickle")

    # GENERATE NX ATTRIBUTE DICTIONARIES
    ced = nx.get_edge_attributes(H, "coord") # coordinates of edges dictionary ced
    ted = nx.get_edge_attributes(H, "category_edge") # type of edges dictionary ted
    tnd = nx.get_node_attributes(H, "category_node") # type of nodes dictionary tnd
    cnd = nx.get_node_attributes(H, "coord") # coordinates of nodes dictionary cnd
    return B, h, eids_conv, ebc, ced


def identify_gaps(city_name, h, eids_conv, ced, D_min = 1.5) -> pd.DataFrame:
    ### GET plist = LIST OF SHORTEST PATHS FOR ALL POSSIBLE CONTACT-TO-CONTACT NODE COMBINATIONS

    plist = []

    if not os.path.exists(f"./src/data/{city_name}/chunks"):
        os.mkdir(f"./src/data/{city_name}/chunks")

    # ALL CONTACT NODES FROM THE NETWORK
    nodestack = [node.index for node in h.vs() if h.vs[node.index]["category_node"]=="multi"]

    count = 0

    while nodestack:
        
        node = nodestack.pop()
        
        # ADDING SHORTEST PATHS FROM CURRENT NODE TO ALL OTHER NODES REMAINING IN THE STACK 
        plist = plist + h.get_shortest_paths(node, to=nodestack, weights="length", mode="out", output = "epath")
        
        # CHUNKWISE SAVING OF RESULTS (TO BE READ IN LATER)
        if len(plist) >= 2*10**5:
            with open(f"./src/data/{city_name}/chunks/c" + str(count) + ".pickle", 'wb') as handle:
                pickle.dump(plist, handle, protocol=pickle.HIGHEST_PROTOCOL)
            del(plist)
            count += 1
            plist = []
            
            
    # SAVING LAST CHUNK (WITH LEN < 2*10**5)
    with open(f"./src/data/{city_name}/chunks/c" + str(count) + ".pickle", 'wb') as handle:
        pickle.dump(plist, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del(plist)

    ### LOOP THROUGH ALL SHORTEST PATHS; KEEP ONLY THE PATHS THAT CONSIST ONLY OF CAR LINKS

    # cs: set of car edges
    cs = set()
    for edge in eids_conv["ig"]:
        if h.es[edge]["category_edge"] == "car":
            cs.add(edge)

    mygaps = []

    # CHUNKWISE:

    mychunks = [f"./src/data/{city_name}/chunks/" + filename for filename in os.listdir(f"./src/data/{city_name}/chunks/")]

    for chunk in mychunks:
        
        with open(chunk, 'rb') as f:
            pathlist = pickle.load(f)

        # adding the item to the gaplist only if it consists of only-car-edges
        gaplist = [item for item in pathlist if set(item).issubset(cs)]

        mygaps = mygaps + gaplist
        
        del(gaplist, pathlist)
        
    print(len(mygaps), " gaps found")

    # remove chunks (not needed anymore)
    for chunk in mychunks:
        os.remove(chunk)
    os.rmdir(f"./src/data/{city_name}/chunks")

    # CONVERT GAPS LIST TO DF AND ADD LENGTH, ORIGIN, DESTINATION

    # to df
    mygaps = pd.DataFrame({"path": mygaps})

    # add length
    mygaps["length"] = mygaps.apply(lambda x: np.sum([h.es[e]["length"] for e in x.path]), axis = 1)

    # add path in nx edge id
    mygaps["path_nx"] = mygaps.apply(lambda x: 
                                    [tuple(sorted(literal_eval(h.es[edge]["edge_id"]))) for edge in x.path], 
                                    axis = 1)

    # add origin and destination nodes
    # (separate procedure for gaps with edgenumber (enr) == 1 vs. gaps with enr > 1)
    mygaps["enr"] = mygaps.apply(lambda x: len(x.path), axis = 1)
    mygaps["o_nx"] = None
    mygaps["d_nx"] = None
    mygaps.loc[mygaps["enr"]==1, "o_nx"] = mygaps[mygaps["enr"] == 1].apply(lambda x: x.path_nx[0][0], axis = 1)
    mygaps.loc[mygaps["enr"]==1, "d_nx"] = mygaps[mygaps["enr"] == 1].apply(lambda x: x.path_nx[0][1], axis = 1)
    mygaps.loc[mygaps["enr"]!=1, "o_nx"] = mygaps[mygaps["enr"]!=1].apply(lambda x: set(x.path_nx[0]).difference(x.path_nx[1]).pop(), axis = 1)
    mygaps.loc[mygaps["enr"]!=1, "d_nx"] = mygaps[mygaps["enr"]!=1].apply(lambda x: set(x.path_nx[-1]).difference(x.path_nx[-2]).pop(), axis = 1)
    mygaps.drop(columns = "enr", inplace = True)

    # add coordinates for  plotting
    mygaps["gapcoord"] = mygaps.apply(lambda x: get_path_coords(x.path_nx, ced), axis = 1)
    return mygaps


def remove_extra_gaps(mygaps, B, D_min = 1.5) -> pd.DataFrame:
    # compute detour factor on bike network
    mygaps["length_b"] = mygaps.apply(lambda x: pathlength_if_connected(B, x.o_nx, x.d_nx), axis = 1)
    mygaps["detour"] = mygaps["length_b"]/mygaps["length"]
    mygaps = mygaps[mygaps["detour"]>=D_min].reset_index(drop = True)
    return mygaps


def prioritize(mygaps, ebc, h) -> pd.DataFrame:
    # compute benefit metric B_star(g)
    mygaps["B_star"] = mygaps.apply(lambda x: 
                                            np.sum([ebc.loc[ebc["edge_ig"]==i, "ebc_lambda"] * \
                                                    h.es[i]["length"] \
                                                    for i in x.path]), 
                                            axis = 1)
    mygaps["B"] = mygaps["B_star"] / mygaps["length"] # B(g) normed to length

    # sort gaps by descending benefit metric
    mygaps = mygaps.sort_values(by = "B", ascending = False).reset_index(drop = True)
    return mygaps


def save_results(city_name, mygaps) -> None:
    mygaps.to_pickle(f"./src/data/{city_name}/pickle/mygaps.pickle") 