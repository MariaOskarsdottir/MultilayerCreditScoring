import csv, os
from itertools import permutations
import numpy as np
import networkx as nx
import multinetx as mx
from scipy.sparse.linalg import eigs
from scipy.sparse import lil_matrix, dok_matrix
from sklearn.preprocessing import normalize

#TODO: debug
import time

class Timer:

    def __init__(self):
        self.start_time = time.time()
        self.increment_time = None

    def report_time(self, message = None):

        if message:
            print(message)

        if not self.increment_time:
            print("--- {:6.1f} sec. ---".format(time.time() - self.start_time))

        else:
            print("--- {:6.1f} / {:6.1f} sec. ---".format(time.time() - self.increment_time, time.time() - self.start_time))

        self.increment_time = time.time()

    
class CreditScoring:

    def __init__(self, layer_files, personal_matrix_file, alpha = 0.85):
        """    
        Attributes
        ----------
        layer_files : a list of paths to csv files 
            for layer construciton
        """
        # TODO: debug
        self.timer = Timer()

        # list containing networkX graphs for each layer
        self.layers = []
        
        for f in layer_files:
    
            # if other files than csv are present, skip those
            if f[-4:] != '.csv':
                continue
                #TODO: Or should we throw an exception?       

            self.layers.append(self.create_layer_from_csv(f, self.number_nodes_in_list()))

        # extract the sets of nodes to be interconnected    
        self.interconnect_sets = []
    
        for layer in self.layers:
            self.interconnect_sets.append({n for n, d in layer.nodes(data=True) if d['bipartite']==0})

        # adds self.interconnections to the party
        self.wire_interconnections()

        N = self.number_nodes_in_list()

        adj_block = lil_matrix((N,N), dtype=np.int8)

        for indices in self.interconnections.values():
        
            # it might happen that a node to interconnect appears only on one layer
            if len(indices) == 1:
                continue

            # only the portion below the diagonal of the inter adj matrix is used in multinetX
            # use it to make the perform better (if not, to run at all!)
            indices.sort(reverse=True)

            pos = 0

            # if we have more than 2 layers the we want to process them all
            while (pos+2) <= len(indices):

                adj_block[indices[pos],indices[pos+1]] = 1

                pos += 1

        self.multilevel_graph = mx.MultilayerGraph(list_of_layers=self.layers, inter_adjacency_matrix=adj_block)
        self.timer.report_time('Multilevel graph created')

        self.supra_transition_matrix = normalize(mx.adjacency_matrix(self.multilevel_graph), norm='l1', axis=0)
        self.timer.report_time('Adj matrix col normalized')
        
        # adds self.pers_matrix and self.defaulter_indices to the party
        self.construct_persoal_matrix(personal_matrix_file)
        self.timer.report_time('Personal matrix created')

        self.supra_transition_matrix = alpha * self.supra_transition_matrix + (1 - alpha)/self.pers_matrix.sum() * self.pers_matrix
        self.timer.report_time('Supra trans matrix calculated')
        print()

        _, leading_eigenvectors = eigs(self.supra_transition_matrix, 1)

        # do we need to be conserned about img (complex numbers!)
        leading_eigenvector = leading_eigenvectors[:, 0].real

        self.leading_eigenvector_norm = leading_eigenvector / leading_eigenvector.sum()

        # adds self.common_nodes_rankings and self.layer_specific_node_rankings to the class namespace
        self.sample_rankings()
        self.timer.report_time('Done eig. calcs. and sampling the ranking dictionaries')
        print()


    def create_layer_from_csv(self, file_path, node_start_id = 0):
        """
        Creates a network layer from a csv file

        The first column in the csv file should hold names of nodes to be interconnected between layers, i.e. common nodes

        Attributes
        ----------
        file_path : str
            path to file, containing 2 columns, to be used in network layer construction
        node_start_id : int
            first id to assign to a node created, defaults to 0
        """

        # conversion to int nodes of this graph will eventually be returned from current function
        g = nx.Graph()

        # first pass over the csv file only creates the nodes with the appropriate bipartite attribute
        # bipartite = 0 is used for nodes to be interconnected between layers
        with open(file_path, encoding='utf8') as f:
            csv_reader = csv.reader(f)

            externally_connected = [] #what about duplicates, are they to be expected? yes so this is ok I think.
            internally_connected = set()

            for row in csv_reader:

                ext_node_str_id = row[0].strip()

                externally_connected.append(ext_node_str_id)
                internally_connected.add(row[1].strip())

        g.add_nodes_from(externally_connected, bipartite=0)
        g.add_nodes_from(internally_connected, bipartite=1)


        # second pass creates a list of edges.
        with open(file_path, encoding='utf8') as f:
            csv_reader = csv.reader(f)

            # get the data as list of tuples
            edges = [(row[0].strip(), row[1].strip()) for row in csv_reader]

        g.add_edges_from(edges)

        # add name to the layer         
        name = os.path.splitext(os.path.basename(file_path))[0]
        g.name = name

        # abandon labels for ids, here we are using the default value for the ordering parameter
        return nx.convert_node_labels_to_integers(g, first_label=node_start_id, ordering='default', label_attribute='name')


    def number_nodes_in_list(self):
        """
        Returns the total count of nodes in the list of networkX layers 

        """

        node_count = 0

        for layer in self.layers:
            node_count += layer.number_of_nodes()

        return node_count


    def wire_interconnections(self):
        """
        Retruns a descriptive datastrcuture (a dict of lists) describing how the nodes are to be interconnected.
        The keys of the dict are node labels and the values are lists containing indices of where 
        those nodes are found in the multilayer network structure 
        """

        self.interconnections = {}

        for n,interconnect_set in enumerate(self.interconnect_sets):
            for node_id in interconnect_set:

                name = self.layers[n].nodes[node_id]['name']

                if name not in self.interconnections:
                    self.interconnections[name] = [node_id]
                else:
                    self.interconnections[name].append(node_id)


    def construct_persoal_matrix(self, file_name):
        
        defaulters = []
        
        #TODO: HERE WE HAVE ADOPTED TO THE R FORMAT BUT WE MUST BE MORE GENERAL THAN THAT --> SIMPLE LIST OF DEFAULTERS
        with open(file_name) as f:
            csv_reader = csv.reader(f)

            for row in csv_reader:

                if row[1] == '1':
                    defaulters.append(row[0].strip())

        self.defaulter_indices = [index_list for (ident, index_list) in self.interconnections.items() if ident in defaulters]

        n = self.multilevel_graph.num_nodes

        self.pers_matrix = dok_matrix((n, n), dtype=np.int8)

        for indexes in self.defaulter_indices:

            # are there better ways to populate all indexes than gettin all 2 permutations of the doubled lists thrown into a set to remove duplicates
            for i,j in set(permutations(indexes + indexes, 2)):

                self.pers_matrix[i,j] = 1


    def print_stats(self):

        padding_1 = '{:12} {:^15} {:^15} {:^15} {:^15}'
        padding_2 = '{:40} {:^15} {:^15}'

        print(padding_1.format('Layer name', 'Total nodes', 'Common nodes', 'Special nodes', 'Number of edges'))

        for l in self.layers:
    
            number_of_zero_bipatrite_nodes = len({n for n, d in l.nodes(data=True) if d['bipartite']==0})
            number_of_one_bipartite_nodes = len({n for n, d in l.nodes(data=True) if d['bipartite']==1})
    
            print(padding_1.format(l.name, l.number_of_nodes(), number_of_zero_bipatrite_nodes, number_of_one_bipartite_nodes, l.number_of_edges()))

        if self.multilevel_graph:
            print()
            print(padding_2.format('Multi layer name', 'Number of nodes', 'Number of edges'))
            print(padding_2.format(self.multilevel_graph.name, self.multilevel_graph.number_of_nodes(), self.multilevel_graph.number_of_edges()))

        print()

    def sample_rankings(self):

        # we are not concerned about ordering of the nodes in the networks 
        # so we are going to return the results as dictionaries

        # first the dict for common nodes
        self.common_nodes_rankings = {}

        for node_ident, indices in self.interconnections.items():

            rank_sum = 0

            for i in indices:

                rank_sum += self.leading_eigenvector_norm[i]

            self.common_nodes_rankings[node_ident] = rank_sum

        print('Length of common nodes ranking dict: ', len(self.common_nodes_rankings))

        # then we propose a list of dictionaries for the rankings of specific nodes in each layer
        # (not knowing how many they are beforehand)

        self.layer_specific_node_rankings = []

        for layer in self.multilevel_graph.list_of_layers:

            layer_ranking_dict = {}

            # get the bipartite=1 set of nodes on current layer
            
            for n, d in layer.nodes(data=True):
                if d['bipartite'] == 1:
                    layer_ranking_dict[d['name']] = self.leading_eigenvector_norm[n]


            self.layer_specific_node_rankings.append(layer_ranking_dict)

        for i, d in enumerate(self.layer_specific_node_rankings):

            print('Number records in dictionary of specific node rankins for layer {}: {}'.format(i, len(d)))

        print()
