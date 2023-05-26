from pathlib import Path

import dgl
import osmnx as ox

MAIN_DIR = './src/road_embedding_gnn/'

validation_line_graphs = dgl.load_graphs(f'{MAIN_DIR}docker_data/data/data_transformed/validation.bin')[0]

print([f'dgl(nodes: {g.num_nodes()}, edges: {g.num_edges()})' for g in validation_line_graphs][0])

found_files = list(Path(f'{MAIN_DIR}docker_data/data/data_val/').glob('*.xml'))

validation_graphs = [ox.io.load_graphml(path) for path in found_files]

print([f'nx(nodes: {g.number_of_nodes()}, edges: {g.number_of_edges()})' for g in validation_graphs][0])


validation_nx2dgl_graphs = [dgl.from_networkx(g) for g in validation_graphs]

print([f'nx2dgl(nodes: {g.num_nodes()}, edges: {g.num_edges()})' for g in validation_nx2dgl_graphs][0])


validation_nx2dgl_line_graphs = [dgl.line_graph(g) for g in validation_nx2dgl_graphs]

print([f'nx2dgl_line(nodes: {g.num_nodes()}, edges: {g.num_edges()})' for g in validation_nx2dgl_line_graphs][0])