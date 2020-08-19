import sys, os
from pprint import pprint
import networkx as nx
import numpy as np

sys.path.append("../credit_scoring")

from credit_scoring import CreditScoring

data_directory = '.'

#files = ['products.csv', 'areas.csv', 'districts.csv']
files = ['products_tiny.csv', 'districts_tiny_no_dup.csv'] #, 'areas_tiny_no_dup.csv']

file_paths = [os.path.join(data_directory, f) for f in files]

personal_file = os.path.join(data_directory, 'defaulters_list_tiny.txt')

mlcs = CreditScoring(file_paths, personal_file, verbose=True, check_common_nodes_in_all_layers = False)

mlcs.print_stats()

print('Common nodes rankings')
pprint(mlcs.common_nodes_rankings)

for i, l in enumerate(mlcs.layer_specific_node_rankings):

    print('Rankings for layer ', i)

    pprint(l)

sys.exit()
print('--- MultiLayer info ---')
layerdict = {}
print('1. nodes and names...')
for node, node_dict in mlcs.multilevel_graph.nodes(data=True):
    print(node, node_dict['name'])
    layerdict[node] = node_dict['name']
print('2. edges')
for edge in mlcs.multilevel_graph.edges():
    print(edge, layerdict[edge[0]], layerdict[edge[1]])

print('Ordering into to adj matrix is acc. to G.nodes(), that follow')
print(mlcs.multilevel_graph.nodes())
print('Adj matrix: ')
adj_matrix = nx.adjacency_matrix(mlcs.multilevel_graph)
adj_matrix.maxprint = np.inf
print(adj_matrix)

print('Sum of elements in the adj. matrix: ', adj_matrix.sum())

numpy_array = adj_matrix.toarray()
np.set_printoptions(threshold=sys.maxsize)
print(numpy_array)

print(mlcs.pers_matrix.toarray())
# now checking the adj matrix by code ;)

num_edges_found = 0
edges_not_in_matrix = []

for edge in mlcs.multilevel_graph.edges():

    if adj_matrix[edge[0], edge[1]] == 1:
        num_edges_found += 1
    else:
        edges_not_in_matrix.append((edge[0], edge[1]))

    if adj_matrix[edge[1], edge[0]] == 1:
        num_edges_found += 1
    else:
        edges_not_in_matrix.append((edge[1], edge[0]))

print('Number of edges found: ', num_edges_found)
print('Edges not found: ')
pprint(edges_not_in_matrix)

print('person matrix\n', mlcs.pers_matrix)
print('defaulter indexes\n', mlcs.defaulter_indices)

print('Now printing the same for each layer...')

for l in mlcs.multilevel_graph.list_of_layers:

    print('--- Layer ', l, 'info ---')

    layerdict = {}

    print('1. nodes and names...')

    for node, node_dict in l.nodes(data=True):

        print(node, node_dict['name'])
        layerdict[node] = node_dict['name']

    print('2. edges')

    for edge in l.edges():

        if l.name == 'districts_tiny_no_dup':
            print(edge, layerdict[edge[0]], layerdict[edge[1]], edge[0] - 19, edge[1] - 19)
        else:
            print(edge, layerdict[edge[0]], layerdict[edge[1]])
    
    print('Ordering into to adj matrix is acc. to G.nodes(), that follow')
    print(l.nodes())

    print('Adj matrix for layer: ', l)

    adj_matrix = nx.adjacency_matrix(l)

    #adj_matrix.maxprint = np.inf

    print(adj_matrix)

for i, l in enumerate(mlcs.layer_specific_node_rankings):

    print('Rankings for layer ', i)

    pprint(l)

#    for n, nd in l.nodes(data=True):
#    
#        print(n,nd['name'])
#print('EDGES...')
#
#for edge in mlcs.multilevel_graph.edges():
#    print(edge)

#print(mlcs.multilevel_graph)
#
#mlcs.supra_transition_matrix
#
#print(type(mlcs.supra_transition_matrix))
