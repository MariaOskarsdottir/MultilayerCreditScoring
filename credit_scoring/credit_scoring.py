try:
    import multinetx as mx
except ImportError:
    raise ImportError("multinetx is required - see instructions for setup")

import csv, os
from itertools import permutations
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigs
from scipy.sparse import lil_matrix, dok_matrix
from sklearn.preprocessing import normalize

import time
from pprint import pprint

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

    def __init__(self, layer_files, personal_matrix_file, alpha = 0.85, csv_delimiter = ',', check_common_nodes_in_all_layers = True, verbose = False):
        """    
        Attributes
        ----------
        layer_files : a list of paths to csv files 
            for layer construciton
        personal_matrix_file : path to a text file 
            listing all the defaulters
        alpha : ratio 0<alpha<= 1
            determines the influence of the personal matrix when calculationg the supra transition matrix. Defaults to 0.85
        csv_delimiter : defaults to , (comma)
            if other symbol is used it can be set here.
        check_common_nodes_in_all_layers : boolean defaults to True
            controls if a test for all common nodes are existing in all layers is included. If this test is run and single common
            node are missing in a single layer a n exception is raised
        verbose : boolean defualts to False
            If set to True some extra information is printed to the console. Mostly concerning running times of the tasks performed. 
        """
        
        self.verbose = verbose
        self.csv_delimiter = csv_delimiter
        self.timer = Timer()

        # list containing networkX graphs for each layer
        self.layers = []
        
        for f in layer_files:
    
            # if other files than csv are present, stop execution and inform user
            if f[-4:] != '.csv':
                raise Exception('Input file not of type csv!')       

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
        
            # optionally checking if all common nodes are present in all layers 
            if check_common_nodes_in_all_layers and len(indices) != len(self.layers):
                raise Exception('Common node present only in subset of layers. Current indices: ', indices)
            
            # it might happen that a node to interconnect appears only on one layer
            if len(indices) == 1:
                continue

            # only the portion below the diagonal of the inter adj matrix is used in multinetX
            # use it to make the perform better (if not, to run at all!)
            indices.sort(reverse=True)

            for i,j in permutations(indices, 2):

                if i > j:
                    adj_block[i, j] = 1


        self.multilevel_graph = mx.MultilayerGraph(list_of_layers=self.layers, inter_adjacency_matrix=adj_block)
        if self.verbose: self.timer.report_time('Multilevel graph created')

        # here we are conserned if the order of nodes into the adjancy matrix is correct and always the same.
        # this site is telling us that his is not an issue if python 3.6 or newer is used: 
        # https://networkx.github.io/documentation/stable/reference/classes/ordered.html

        nodes = list(self.multilevel_graph.nodes())
        
        if not all(nodes[i] < nodes[i + 1] for i in range(len(nodes)-1)):
            raise Exception("Nodes not in proper order for Adjacency matrix extraction") 

        # column normalized adj matrix
        self.supra_transition_matrix = normalize(nx.to_scipy_sparse_matrix(self.multilevel_graph, dtype=np.int8), norm='l1', axis=0)        
        if self.verbose: self.timer.report_time('Adj matrix col normalized')
        
        # adds self.pers_matrix and self.defaulter_indices to the party
        self.construct_persoal_matrix(personal_matrix_file)
        if self.verbose: self.timer.report_time('Personal matrix created')

        self.supra_transition_matrix = alpha * self.supra_transition_matrix + (1 - alpha)/self.pers_matrix.sum() * self.pers_matrix
        if self.verbose: self.timer.report_time('Supra trans matrix calculated')

        _, leading_eigenvectors = eigs(self.supra_transition_matrix, 1)

        # do we need to be conserned about img (complex numbers!)
        leading_eigenvector = leading_eigenvectors[:, 0].real
        
        # normalize the eigenvector
        self.leading_eigenvector_norm = leading_eigenvector / leading_eigenvector.sum() 

        # adds self.common_nodes_rankings and self.layer_specific_node_rankings to the class namespace
        self.sample_rankings()
        if self.verbose: self.timer.report_time('Done eig. calcs. and sampling the ranking dictionaries')

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
            csv_reader = csv.reader(f, delimiter = self.csv_delimiter)

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
            csv_reader = csv.reader(f, delimiter = self.csv_delimiter)

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
        """
        from a simple list of defaulter names in a text file, this function populates the personal matrix, self.pers_matrix 
        """
        
        with open(file_name) as f:
            defaulters = [row.strip() for row in f.readlines()]

        self.defaulter_indices = [index_list for (ident, index_list) in self.interconnections.items() if ident in defaulters]

        n = self.multilevel_graph.num_nodes

        self.pers_matrix = dok_matrix((n, n), dtype=np.int8)

        for indexes in self.defaulter_indices:

            # are there better ways to populate all indexes than gettin all 2 permutations of the doubled lists thrown into a set to remove duplicates
            for i,j in set(permutations(indexes + indexes, 2)):

                self.pers_matrix[i,j] = 1

    def print_stats(self):
        """
        Prints information on number of nodes of different types and endes in each layer and in the multilayer graph
        """

        len_layer_name = [len(x.name) for x in self.layers]
        
        len_longest_lname = max(len_layer_name)

        padding_1 = '{:' + str(len_longest_lname) + '} {:^15} {:^15} {:^15} {:^15}'
        padding_2 = '{:' + str(len(self.multilevel_graph.name)) + '} {:^15} {:^15}'

        print()
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
        """
        Calculates the final result, the ranking of nodes in the network. Ranings of common nodes are found in
        the dict self.common_nodes_rankings. For the specific nodes the list self.layer_specific_node_rankings has
        items (as dictionaries) for each layer of the multilayer grap.
        """

        maxValue = 0

        # first the dict for common nodes
        common_nodes_rankings = {}

        for node_ident, indices in self.interconnections.items():

            rank_sum = 0

            for i in indices:

                rank_sum += self.leading_eigenvector_norm[i]

            common_nodes_rankings[node_ident] = rank_sum

            if rank_sum > maxValue:
                maxValue = rank_sum

        if self.verbose: 
            print()
            print('Length of common nodes ranking dict: ', len(common_nodes_rankings))


        # then we propose a list of dictionaries for the rankings of specific nodes in each layer
        # (not knowing how many they are beforehand)

        layer_specific_node_rankings = []

        for layer in self.multilevel_graph.list_of_layers:

            layer_ranking_dict = {}

            # get the bipartite=1 set of nodes on current layer
            
            for n, d in layer.nodes(data=True):
                if d['bipartite'] == 1:
                    #layer_ranking_dict[d['name']] = self.leading_eigenvector_norm[n]
                    v = self.leading_eigenvector_norm[n]
                    layer_ranking_dict[d['name']] = v
                    if v > maxValue:
                        maxValue = v

            layer_specific_node_rankings.append(layer_ranking_dict)

        if self.verbose:
            for i, d in enumerate(layer_specific_node_rankings):
                print('Number records in dictionary of specific node rankins for layer {}: {}'.format(i, len(d)))
            print()

        # finally we must normalize all values to have the maximum as 1
        self.common_nodes_rankings = {key : value / maxValue for (key, value) in common_nodes_rankings.items()}
        self.layer_specific_node_rankings = []
        for un_norm_dict in layer_specific_node_rankings:
            self.layer_specific_node_rankings.append({key : value / maxValue for (key, value) in un_norm_dict.items()})