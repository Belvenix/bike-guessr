# system
import os
from pathlib import Path
import time
from datetime import date
import pickle

# data
import math
import numpy as np
import pandas as pd
import random
import itertools
from collections import Counter
from ast import literal_eval # to convert str into lists (imported osmid data)

# network
import networkx as nx
import igraph as ig

# geo
import geopandas as gpd
import shapely

# plot
import matplotlib.pyplot as plt
import folium
from folium.features import DivIcon
from IPython.display import display


# parameters for plotting

plotparam = {"dpi": 96}

# create basemaps dict for folium tile layers
basemaps = {
    'Google Maps': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Maps',
                overlay = True,
                control = True
        ),
    'Google Satellite': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Satellite',
                overlay = True,
                control = True
        ),
'st_lite_op3': folium.TileLayer(
        tiles = 'https://stamen-tiles.a.ssl.fastly.net/toner-lite/{z}/{x}/{y}.png',
        attr = 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name = 'Stamen Toner Lite O3',
        overlay = True,
        control = True,
        opacity = 0.3),
    'osm': folium.TileLayer(
        tiles = "openstreetmap", 
        name = "OpenStreetMap",
        control = True, 
        overlay = True),
    'white_background': folium.TileLayer(
        tiles = 'https://api.mapbox.com/styles/v1/krktalilu/ckrdjkf0r2jt217qyoai4ndws/tiles/256/{z}/{x}/{y}@2x?access_token=pk.eyJ1Ijoia3JrdGFsaWx1IiwiYSI6ImNrcmRqMXdycTB3NG8yb3BlcGpiM2JkczUifQ.gEfOn5ttzfH5BQTjqXMs3w',
        name = "White background",
        attr = 'Mapbox',
        control = True,
        overlay = True
        )

}


# IMPORT OBJECTS FROM PREVIOUS STEPS

def import_identify_prioritize_data(city_name):

    H = nx.read_gpickle(f"./src/data/{city_name}/pickle/H.gpickle")
    h = ig.Graph.Read_Pickle(f"./src/data/{city_name}/pickle/h.pickle")

    ebc = pd.read_pickle(f"./src/data/{city_name}/pickle/ebc.pickle")
    mygaps = pd.read_pickle(f"./src/data/{city_name}/pickle/mygaps.pickle")

    # NX attribute dicts (for plotting)
    ced = nx.get_edge_attributes(H, "coord") # coordinates of edges dictionary ced
    ted = nx.get_edge_attributes(H, "category_edge") # type of edges dictionary ted
    tnd = nx.get_node_attributes(H, "category_node") # type of nodes dictionary tnd
    cnd = nx.get_node_attributes(H, "coord") # coordinates of nodes dictionary cnd

    return mygaps, H, h, ebc, ced, ted


def decluster(city_name, mygaps, h, ebc, ced, B_cutoff = 15300) -> pd.DataFrame:
    # make subfolder for storing output
    if not os.path.exists(f"./src/analysis/{city_name}/"):
        os.makedirs(f"./src/analysis/{city_name}/")
    if os.path.exists(f"./src/data/{city_name}/pickle/gaps_declustered.pickle"):
        gap_dec = pd.read_pickle(f"./src/data/{city_name}/pickle/gaps_declustered.pickle")
    else:
        mygaps = mygaps[mygaps["B"]>=B_cutoff].reset_index(drop = True)

        # ADDING EBC VALUES / RANKING METRICS TO EACH EDGE in network "f" (which will be used for declustering)
        f = h.copy()
        for i in range(len(ebc)):
            f.es[i]["ebc"] = ebc.loc[i, "ebc_lambda"]
            

        ### make a subgraph of f that contains only overlapping gaps

        my_edges = list(set([item for sublist in mygaps["path"] for item in sublist]))
        c = f.copy()
        c = c.subgraph_edges(edges = my_edges, 
                                delete_vertices=True)

        cl = c.decompose() ### cl contains disconnected components 

        gapranks = []
        gapcoords = []

        gapranks = []
        gapcoords = []
        gapedgeids = []

        # looping through disconnected components
        from tqdm import tqdm
        from datetime import datetime
        for comp in tqdm(range(len(cl)), desc=f'{datetime.now().isoformat()} Declustering {city_name} - disconnected components loop', total=len(cl)):
            
            mc = cl[comp].copy() # mc: my component (current)

            #### decluster component:

            while len(mc.es) > 0:

                nodestack = [node.index for node in mc.vs() if mc.vs[node.index]["category_node"]=="multi" and mc.degree(node.index)!=2]
                nodecomb = [comb for comb in itertools.combinations(nodestack, 2)] ### all possible OD combis on that cluster
                sp = [] ### list of shortest paths between d1 multi nodes on that cluster:
                for mycomb in nodecomb:

                    gsp = mc.get_shortest_paths(mycomb[0], 
                                                        mycomb[1],
                                                        weights = "length", 
                                                        mode = "out", 
                                                        output = "epath")[0]
                    if gsp:
                        sp.append(gsp)

                ### compute metrics for each path:
                if not sp:
                    break

                lens = []
                cycs = []
                
                for p in sp:
                    lens += [np.sum([mc.es[e]["length"] for e in p])]
                    cycs += [np.sum([mc.es[e]["length"]*mc.es[e]["ebc"] for e in p])]
                    
                norms = list(np.array(cycs)/np.array(lens))
                maxpath = sp[norms.index(max(norms))]
                gapranks.append(np.round(max(norms), 2))
                
                gapcoord_current = []
                edgeids_current = []
                
                
                for e in maxpath:
                    edge_id = tuple(sorted(literal_eval(mc.es[e]["edge_id"])))
                    edge_coords = [(c[1], c[0]) for c in ced[tuple(sorted(edge_id))].coords]
                    gapcoord_current.append(edge_coords)
                    edgeids_current.append(edge_id)
                    
                mc.delete_edges(maxpath)
                gapcoords.append(gapcoord_current)
                gapedgeids.append(edgeids_current)


        gap_dec = pd.DataFrame({"rank": gapranks, "coord": gapcoords, "id": gapedgeids})
        gap_dec = gap_dec.sort_values(by = "rank", ascending = False).reset_index(drop = True)
        gap_dec = gap_dec[gap_dec["rank"]>=B_cutoff].copy().reset_index(drop = True)

        gap_dec.to_pickle(f"./src/data/{city_name}/pickle/gaps_declustered.pickle")

        ### export to csv to start manual analysis
        gap_dec["address"] = None
        gap_dec["class"] = None
        gap_dec["comments"] = None
        gap_dec[["rank", "address", "class", "comments"]].to_csv(path_or_buf = f"./src/analysis/{city_name}/gaps_declustered_table.csv", 
                                                                sep = ";", 
                                                                index = True, 
                                                                encoding = "utf-8", 
                                                                decimal = ",")
    return gap_dec


def plot_folium_declustered_gaps(city_name, H, ced, ted, gap_dec, city_coords=[55.6761, 12.5683]):

    ### SET UP MAP WITH FOLIUM

    # make map object

    # coopenhagen coordinates
    mycity_coord = city_coords

    m = folium.Map(location = mycity_coord, zoom_start = 13, tiles = None) 

    # add tile layers to m
    for key in ["osm", "st_lite_op3", "white_background", "Google Satellite"]:
        basemaps[key].add_to(m)
        
    ##### plot car, onlybike, and multinetworks #####
    snw = folium.FeatureGroup(name = "All streets", show = True)
    bnw = folium.FeatureGroup(name = "Bikeable", show = True) # bikeable edges 

    sloc = []
    bloc = []

    for edge in H.edges:
        myloc = [(c[1], c[0]) for c in ced[tuple(sorted(edge))].coords]
        sloc.append(myloc)
        if ted[edge] != "car":
            bloc.append(myloc)
            
    snw_line = folium.PolyLine(locations = sloc, weight = 2, color = "#dadada").add_to(snw)
    bnw_line = folium.PolyLine(locations = bloc, weight = 3, color = "black").add_to(bnw)

    snw.add_to(m)
    bnw.add_to(m)


    #### ADD TO MAP 


    gaps_fg = folium.FeatureGroup("Declustered gaps", show = True)

    for locs in gap_dec["coord"]:
        myline = folium.PolyLine(locations = locs,
                                    weight = 6,
                                    color = "black").add_to(gaps_fg) # black border
        # + randomly colored gaps (not classified yet):
        myline = folium.PolyLine(locations = locs,
                                    weight = 4,
                                    color = random.choice(["#33FFDA", "#0AB023", "#B00A60", "#0A3CB0"])).add_to(gaps_fg)

    gaps_fg.add_to(m)


    my_fg_nr = folium.FeatureGroup("Gap numbers", show = False)

    # Add pop-ups with gap number

    for i in range(len(gap_dec)):
        
        my_marker = folium.Marker(location = gap_dec.loc[i]["coord"][0][0], 
                                        popup = "Gap " + str(i)
                                ).add_to(my_fg_nr)
                    
    my_fg_nr.add_to(m)

    ### ADD LAYER CONTROL AND SAVE / DISPLAY FOLIUM MAP
    folium.LayerControl().add_to(m)  

    m.save(f"./src/analysis/{city_name}/gaps_declustered_plot.html")


def manual_classify(city_name):
    # import manually classified gaps
    gaps_class = pd.read_csv(f"./src/analysis/{city_name}/gaps_classified_table.csv", 
                            index_col = 0,
                            sep = ";",  
                            decimal = ",",
                            encoding = "utf-8")# , sep = ";", decimal = ",", index_col = 0)

    # import gap info (coords, edge ids, rank)
    gaps_info = pd.read_pickle(f"./src/data/{city_name}/pickle/gaps_declustered.pickle")

    # merge gaps_class and gap_info to get all data on classified gaps
    gaps = pd.merge(gaps_class, gaps_info, how = "outer", indicator = False, on = "rank")

    # drop unclassified gaps (that were not confirmed as gaps)
    gaps_final = gaps.drop(gaps[gaps["class"].isnull()].index, errors='ignore').reset_index(drop=True)
    gaps_final["gapnumber"] = gaps_final.index + 1
    gaps_final.to_pickle(f"./src/data/{city_name}/pickle/gaps_final.pickle") # save final gaps to pickle