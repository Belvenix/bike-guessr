# system
import os
import time
from typing import Tuple

# data
import numpy as np
import pandas as pd
import itertools
from ast import literal_eval # to convert str into lists (imported osmid data)

# network
import networkx as nx
import igraph as ig

# geo
import geopandas as gpd
import shapely

import logging

LOG = logging.getLogger(__name__)


def import_check(city_name) -> bool:
    filenames = ['mnw.gpickle', 'mnwl.gpickle', 'B.gpickle', 'h.pickle', 'b.pickle', 'eids_conv.pickle', 'nids_conv.pickle', 'ebc.pickle']
    for filename in filenames:
        if os.path.exists(f"./src/data/{city_name}/pickle/{filename}"):
            continue
        else:
            return False
    return True


def make_attr_dict(*args, **kwargs): 
    
    argCount = len(kwargs)
    
    if argCount > 0:
        attributes = {}
        for kwarg in kwargs:
            attributes[kwarg] = kwargs.get(kwarg, None)
        return attributes
    else:
        return None # (if no attributes are given)


def read_csv(city_name) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

    # bike edges
    be = pd.read_csv(f'./src/data/{city_name}/{city_name}_biketrack_edges.csv').drop(columns = ["key", "lanes", "name", "highway", "maxspeed", "bridge", "tunnel", "junction", "width", "access", "ref", "service", "area"], errors='ignore')
    be["geometry"] = be.apply(lambda x: shapely.wkt.loads(x.geometry), axis = 1)
    be = gpd.GeoDataFrame(be, geometry = "geometry") 

    # car edges
    ce = pd.read_csv(f'./src/data/{city_name}/{city_name}_carall_edges.csv').drop(columns = ["key", "lanes", "name", "highway", "maxspeed", "bridge", "tunnel", "junction", "width", "access", "ref", "service"], errors='ignore')
    ce["geometry"] = ce.apply(lambda x: shapely.wkt.loads(x.geometry), axis = 1)
    ce = gpd.GeoDataFrame(ce, geometry = "geometry") 

    # bike nodes
    bn = pd.read_csv(f'./src/data/{city_name}/{city_name}_biketrack_nodes.csv').drop(columns = ["highway", "ref"], errors='ignore')
    bn["geometry"] = bn.apply(lambda x: shapely.wkt.loads(x.geometry), axis = 1)
    bn = gpd.GeoDataFrame(bn, geometry = "geometry")

    # car nodes
    cn = pd.read_csv(f'./src/data/{city_name}/{city_name}_carall_nodes.csv').drop(columns = ["highway", "ref"], errors='ignore')
    cn["geometry"] = cn.apply(lambda x: shapely.wkt.loads(x.geometry), axis = 1)
    cn = gpd.GeoDataFrame(cn, geometry = "geometry")

    # AN: dataframe of ALL NODES

    # merge all nodes to one dataframe, 
    an = pd.merge(bn, cn, how = "outer", indicator = True) # merging
    an["type"] = an["_merge"].cat.rename_categories(["bike", "car", "multi"]) # adding info on type
    an = an.drop(columns = "_merge")
    an = an.sort_values(by = "osmid").reset_index(drop = True) # sort by osmid
    # make attribute dictionary with type and geocoordinates for each node
    an["attr_dict"] = an.apply(lambda x: make_attr_dict(category_node = x.type, coord = x.geometry), axis = 1) # add attr_dict

    # AE: dataframe of ALL EDGES

    # make df with all edges (ae) to pass it to nx

    # add edge ids (strings with "id1, id2" sorted (id1 < id2))
    be["edge_id"] = be.apply(lambda x: str(sorted([x["u"], x["v"]])), axis = 1)
    ce["edge_id"] = ce.apply(lambda x: str(sorted([x["u"], x["v"]])), axis = 1)
    # (edge ids are set as strings; converting back: with "from ast import literal_eval" fct)
    # finding duplicates by ["osmid", "oneway", "edge_id", "length"]

    # simplifying network into undirected - beu and ceu contain the "undirected" edges
    # (removing all parallel edges)

    beu = be.drop_duplicates(subset = ["osmid", "oneway", "edge_id", "length"],
                    keep = "first",
                    inplace = False,
                    ignore_index = True).copy()
    ceu = ce.drop_duplicates(subset = ["osmid", "oneway", "edge_id", "length"],
                    keep = "first",
                    inplace = False,
                    ignore_index = True).copy()

    # add type info prior to merging
    beu["type"] = "bike"
    ceu["type"] = "car"

    # concatenate
    ae = pd.concat([beu, ceu]).reset_index(drop = True)

    # change type for "multi" for edges that appear in both sets
    ae.loc[ae.duplicated(subset = ["u", "v", "osmid", "oneway", "length", "edge_id"], keep = False), "type"] = "multi"

    # remove duplicates
    ae = ae.drop_duplicates(subset = ["u", "v", "osmid", "oneway", "length", "edge_id", "type"], 
                            keep = "first",
                            ignore_index = True, 
                            inplace = False)

    ae_tokeep = ae[ae.duplicated("edge_id", keep = False) & (ae["type"]=="bike")].index
    ae_todrop = ae[ae.duplicated("edge_id", keep = False) & (ae["type"] == "car")].index

    ae.loc[ae_tokeep, "type"] = "multi"
    ae = ae.drop(ae_todrop, errors='ignore')

    # add attribute dictionary (for nx)
    ae["attr_dict"] = ae.apply(lambda x: make_attr_dict(length = x.length, 
                                                        category_edge = x.type,
                                                        edge_id = x.edge_id,
                                                        coord = x.geometry,
                                                        intnodes = []), # intnodes attribute: for storing simplification info on interstitial nodes 
                                axis = 1)

    # sort by "left" node (id1 < id2 - to control order of tuple keys in nx)
    ae["order"] = ae.apply(lambda x: np.min([x["u"], x["v"]]), axis = 1)
    ae = ae.sort_values(by = "order").reset_index(drop = True)
    ae["orig"] = ae.apply(lambda x: np.min([x["u"], x["v"]]), axis = 1)
    ae["dest"] = ae.apply(lambda x: np.max([x["u"], x["v"]]), axis = 1)
    ae = ae.drop(columns = ["order", "u", "v"], errors='ignore') # instead of "u" and "v",
    # we will use "origin" and "destination" where osmid(origin) < osmid (destination)!
    return ae, an


def create_nx_objects(city_name, all_edges, all_nodes) -> nx.Graph:
    # create subfolder to save pickle files as results
    if not os.path.exists(f"./src/data/{city_name}/pickle"):
        os.mkdir(f"./src/data/{city_name}/pickle")

    # CREATE NX OBJECTS

    # make multinetwork containing ALL edges
    mnw = nx.Graph()
    mnw.add_nodes_from(all_nodes.loc[:,["osmid", "attr_dict"]].itertuples(index = False))
    mnw.add_edges_from(all_edges.loc[:,["orig", "dest", "attr_dict"]].itertuples(index = False))

    # save to pickle ("original" nw = non-simplified, with disconnected components)
    nx.readwrite.gpickle.write_gpickle(mnw, f"./src/data/{city_name}/pickle/mnw.gpickle")

    # KEEP ONLY LARGEST CONNECTED COMPONENT

    # make list of connected components
    cd_nodeset = []

    for comp in nx.connected_components(mnw):
        
        cd_nodeset = cd_nodeset + [comp]
        
    n = len(cd_nodeset)
        
    LOG.info("number of disconnected components on mnw: " + str(n))

    cd_size = [None]*n
    cd_network = [None]*n
    cd_coord_dict = [None]*n
    cd_coord_list = [None]*n
    cd_types = [None]*n

    for i in range(n):
        cd_size[i] = len(cd_nodeset[i])
        cd_network[i] = nx.subgraph(mnw, cd_nodeset[i])
        cd_coord_dict[i] = nx.get_edge_attributes(cd_network[i], "coord")
        cd_coord_list[i] = [cd_coord_dict[i][key] for key in cd_coord_dict[i].keys()]
        cd_types[i] = nx.get_edge_attributes(cd_network[i], "category_edge")

    # make df with info on connected components
    comps = pd.DataFrame({
        'nodeset': cd_nodeset, 
        'size': cd_size,
        'network': cd_network,
        'coord': cd_coord_list,
        'type': cd_types})

    del(cd_nodeset, cd_size, cd_network, cd_coord_list, cd_types, cd_coord_dict)

    # lcc is the size of the largest connected component
    lcc = np.max(comps["size"])

    LOG.info("size of lcc: " + str(lcc))

    comps = comps.sort_values(by = "size", ascending = False).reset_index(drop = True)

    # DEFINE MNWL as largest connected component
    # (drop all others)
    mnwl_nodes = comps["nodeset"][0]
    mnwl_edges = all_edges.loc[all_edges.apply(lambda x: x.orig in mnwl_nodes, axis = 1),:].copy().reset_index(drop = True)
    mnwl = nx.subgraph(mnw, mnwl_nodes)

    # save as pickle ("original" nw = non-simplified, but only LCC)
    nx.readwrite.gpickle.write_gpickle(mnwl, f"./src/data/{city_name}/pickle/mnwl.gpickle")


    # make a copy of mnwl - H will be simplified and manipulated throughout while loop
    H = mnwl.copy()

    # set parameters for the while loop
    simplify_further = True
    run = 0

    # make dictionary of edge attributes of mnwl
    mnwl_typedict = nx.get_edge_attributes(mnwl, "category_edge")

    # loop runs while there are interstitial nodes on the nw
    while simplify_further:
        
        run += 1
        LOG.info("Run " + str(run) + ", " + time.ctime())
        
        # get all nodes from nw
        points_all_list = sorted(list(H.nodes))

        # get all node degrees
        degrees_all_list = [None]*len(points_all_list)
        for i in range(len(points_all_list)):
            degrees_all_list[i] = H.degree(points_all_list[i])

        # make df with node + degree info + remove (T/F) + types (of incident edges)
        pointsall = pd.DataFrame({
            "osmid": points_all_list, 
            "d": degrees_all_list, 
            "remove": None, 
            "types": None})
        
        # get edge attributes (of CURRENT nw) as dict
        catdict = nx.get_edge_attributes(H, "category_edge")
        
        # get edge type information (car/bike/multi) from attribute dictionary
        pointsall["types"] = pointsall.apply(lambda x: 
                                            [ catdict[tuple(sorted(edge))] for edge in H.edges(x.osmid) ], 
                                            axis = 1)

        # split df in "endpoints" and d2 nodes
        pointsend = pointsall[pointsall["d"]!=2].copy().reset_index(drop = True)
        pointsd2 = pointsall[pointsall["d"]==2].copy().reset_index(drop = True)

        # non-d2 nodes: all of them are remove=False (to keep)
        pointsend["remove"] = False
        # d2 nodes: the ones that have same 2 edge types incident are remove=True
        pointsd2["remove"] = pointsd2.apply(lambda x: x.types[0]==x.types[1], axis = 1)

        # final result: 2 dfs - nodes_final and nodes_interstitial

        # nodes_final = nodes to keep (either they have d!=2 or they have d==2 but 2 different edge types)
        nodes_final = pd.concat([pointsend, pointsd2[pointsd2["remove"]==False].copy()]).reset_index(drop = True)

        # nodes_interstitial = nodes to remove (d2 nodes with same 2 edge types incident)
        nodes_interstitial = pointsd2[pointsd2["remove"]==True].copy().reset_index(drop = True)
        nodes_interstitial["types"] = nodes_interstitial.apply(lambda x: x.types[0], axis = 1) # remove second-edge info (is same as first)

        del(pointsall, catdict, degrees_all_list, points_all_list, pointsend, pointsd2)

        # save info about endpoint/interstitial to node attributes on mnwl
        for i in range(len(nodes_interstitial)):
            H.nodes[nodes_interstitial.loc[i, "osmid"]]["category_point"] = "int"
        for i in range(len(nodes_final)):
            H.nodes[nodes_final.loc[i, "osmid"]]["category_point"] = "end"

        # make df with interstitial edges
        eint = nodes_interstitial.copy() 
        eint["orig"] = eint.apply(lambda x: sorted([n for n in H.neighbors(x.osmid)])[0], axis = 1)
        eint["dest"] = eint.apply(lambda x: sorted([n for n in H.neighbors(x.osmid)])[1], axis = 1)

        # add info on edge lengths
        lendict = nx.get_edge_attributes(H, "length")
        eint["length_new"] = eint.apply(lambda x: 
                                        np.sum(
                                            [lendict[tuple(sorted(edge))] for edge in H.edges(x.osmid)]
                                        ), 
                                        axis = 1)

        stack = list(np.unique(eint["osmid"]))
        
        Hprior = H.copy() # make a copy of the nw in each simplification step
        # to use for checking for neighbours for removing from stack
        
        # interstitial nodes dictionary - to keep track of nodes that are removed by "while stack"
        intnodesdict = nx.get_edge_attributes(H, "intnodes")
        # edge coordinate dictionary - to merge linestrings of aggregated edges
        edgecoorddict = nx.get_edge_attributes(H, "coord")
        
        while stack:

            mynode = stack.pop()
            
            for n in nx.neighbors(Hprior, mynode): # remove neighbors from ORIGINAL nw
                if n in stack:
                    stack.remove(n)
                    #LOG.info("removed "+ str(n))
                    
            # u and v are the neighbors of "mynode"
            u = eint.loc[eint["osmid"]==mynode]["orig"].values[0]
            v = eint.loc[eint["osmid"]==mynode]["dest"].values[0]
            
            # counter (to break out of loop if it is not increased)
            nodes_removed = 0
            
            if (u,v) not in H.edges: # only if neighbors are not neighbors themselves - 
                # to avoid roundabouts from disappearing
                
                # get info on interstitional nodes (for deriving edge coordinates later on)
                myintnodes = [intnodesdict[tuple(sorted(edge))] for edge in H.edges(mynode)]
                myintnodes.append([mynode])
                myintnodes = [x for x in list(itertools.chain.from_iterable(myintnodes)) if x]
                
                H.add_edge(u_of_edge = u,
                            v_of_edge = v,
                            length = eint.loc[eint["osmid"]==mynode]["length_new"].values[0],
                            category_edge = eint.loc[eint["osmid"]==mynode]["types"].values[0],
                            intnodes = myintnodes,
                            edge_id = str(sorted([u, v])),
                            coord = shapely.ops.linemerge( [ edgecoorddict[tuple(sorted([u,mynode]))],
                                                            edgecoorddict[tuple(sorted([v,mynode]))] ]
                                                        ) 
                        )

                H.remove_node(mynode)
                nodes_removed += 1
        
        if nodes_removed == 0:
            
            simplify_further = False # to break out of loop
                    
            # save simplified network to H gpickle
            nx.readwrite.gpickle.write_gpickle(H, f"./src/data/{city_name}/pickle/H.gpickle") 
            
            LOG.info("Done")
    return H


def transform_nx_and_save(city_name, H) -> None:
    # make "bikeable" network from H (excluding car edges)
    bikeable_nodes = [node for node in H.nodes if H.nodes[node]["category_node"]!="car"]
    H_noncar_induced = H.subgraph(bikeable_nodes).copy() 
    # induced subgraph - still contains the car edges that lie between multi nodes; - exclude them:
    banw = H_noncar_induced.copy()
    banw.remove_edges_from([edge for edge in banw.edges if banw.edges[edge]["category_edge"]=="car"])
    nx.readwrite.gpickle.write_gpickle(banw, f"./src/data/{city_name}/pickle/B.gpickle") 

    # conversion to igraph
    h = ig.Graph.from_networkx(H)
    h.write_pickle(f"./src/data/{city_name}/pickle/h.pickle")
    b = ig.Graph.from_networkx(banw)
    b.write_pickle(f"./src/data/{city_name}/pickle/b.pickle")
    # to read in again: Graph.Read_Pickle()

    # eids: "conversion table" for edge ids from igraph to nx 
    eids_nx = [tuple(sorted(literal_eval(h.es(i)["edge_id"][0]))) for i in range(len(h.es))]
    eids_ig = [i for i in range(len(h.es))]
    eids_conv = pd.DataFrame({"nx": eids_nx, "ig": eids_ig})

    # nids: "conversion table" for node ids from igraph to nx
    nids_nx = [h.vs(i)["_nx_name"][0] for i in range(len(h.vs))]
    nids_ig = [i for i in range(len(h.vs))]
    nids_conv = pd.DataFrame({"nx": nids_nx, "ig": nids_ig})

    eids_conv.to_pickle(f"./src/data/{city_name}/pickle/eids_conv.pickle")
    nids_conv.to_pickle(f"./src/data/{city_name}/pickle/nids_conv.pickle")

    # ---


    # extract edge and node attributes as dictionaries

    # tnd = nx.get_node_attributes(H, "category_node") # type of nodes dictionary tnd
    # ted = nx.get_edge_attributes(H, "category_edge") # type of edges dictionary tnd
    # led = nx.get_edge_attributes(H, "length") # length of edges dictionary led
    # cnd = nx.get_node_attributes(H, "coord") # coordinates of nodes dictionary cnd
    # ced = nx.get_edge_attributes(H, "coord") # coordinates of edges dictionary ced

    # make data frame of ebc with:
    ebc = pd.DataFrame({"edge_ig": [e.index for e in h.es]}) # igraph edge ID
    ebc["edge_nx"] = ebc.apply(lambda x: tuple(literal_eval(h.es[x.edge_ig]["edge_id"])), axis = 1) # nx edge ID
    ebc["length"] = ebc.apply(lambda x: h.es[x.edge_ig]["length"], axis = 1) # length in meters

    # compute ebcs:
    ebc["ebc_inf"] = h.edge_betweenness(directed = False, cutoff = None, weights = "length") # "standard" ebc
    ebc["ebc_lambda"] = h.edge_betweenness(directed = False, cutoff = 2500, weights = "length") # ebc only including *paths* below 2500m
    ebc.to_pickle(f"./src/data/{city_name}/pickle/ebc.pickle")