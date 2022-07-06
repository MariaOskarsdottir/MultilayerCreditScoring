import csv
import os
from itertools import permutations
import numpy as np
from igraph import Graph
from functools import reduce
import pandas as pd
from operator import and_
from scipy.sparse.linalg import eigs
from scipy.sparse import lil_matrix, dok_matrix
from sklearn.preprocessing import normalize
from collections import defaultdict
import time
from pprint import pprint
from itertools import permutations
from typing import List


class MultiLayerRanker:

    def __init__(self, layer_files, common_nodes_file, personal_file, biderectional=False, sparse=True):
        """Initializer for Multi Layer Ranker

        Args:
            layer_files (list): list of layer files 
            common_nodes_file (str): csv file to common nodes 
            personal_file (str): file to create personal matrix 
            biderectional (bool, optional): wheter edges are biderectional or not. Defaults to False.
            sparse (bool, optional): use sparse or desnse matrix. Defaults to True.
        """
        assert reduce(
            and_, [f[-5:] == '.ncol' for f in layer_files]), "File not in ncol format"
        self.files = layer_files
        self.filesNoExt = [f.strip(".ncol")for f in self.files]
        self.sparse = sparse
        self.personal_file = personal_file
        self.labs = []
        self.common_nodes = {}
        self.common = pd.read_csv(common_nodes_file, header=None)
        self.gs = {}
        # create independant layers
        for f in self.files:
            g = Graph()
            i = 0

            for _, node in self.common.iterrows():
                g.add_vertex(f'{node[0]}')
                self.common_nodes[f'{node[0]}'] = i
                i += 1

            with open(f) as graphFile:
                for line in graphFile:
                    u, v = line.split()
                    for n in [u, v]:
                        if not len(g.vs.select(name=f'{n}')):
                            g.add_vertex(n)
                    g.add_edge(u, v)
            self.gs[f.strip('.ncol')] = g
        self.ns = {f: self.gs[f].vcount() for f in self.filesNoExt}
        self.N = sum([self.ns.get(i) for i in self.ns])
        self.startingInd = {}
        self.layers = []
        self.bounds = {}
        if sparse:
            self.adj_matrix = lil_matrix((self.N, self.N), dtype=np.int8)
            self.personal = lil_matrix((self.N, self.N), dtype=np.int8)
        else:
            self.adj_matrix = np.zeros((self.N, self.N))
            self.personal = np.zeros((self.N, self.N))
        # build adj matrix
        self.buildAdjMatrix(biderectional)
        # build personal matrix
        self.construct_personal_matrix()

    def makeBounds(self):
        """
        Helper function to create the bounds of each layer in the adj Matrix
        """
        for i, f in enumerate(reversed(self.filesNoExt)):

            if i == 0:
                prev = f
                self.bounds[f] = (self.startingInd[f], self.N - 1)
                continue

            self.bounds[f] = (self.startingInd[f], self.startingInd[prev]-1)
            prev = f

    def pageRank(self, alpha=.85):
        """
        General personalized page rank given the adjacency matrix, personal matrix and alpha score

        Args:
            alpha (int): page rank exploration parameter, defaults to .85 

        Returns:
            array: leading eigen vector corresponding to the rank of each node 
        """

        matrix = self.adj_matrix

        matrix = normalize(matrix, norm='l1', axis=0)

        matrix = alpha * matrix + (1 - alpha) / \
            self.personal.sum() * self.personal

        self.supra = matrix
        _, leading_eigenvectors = eigs(matrix, 1)

        # do we need to be conserned about img (complex numbers!)
        leading_eigenvector = leading_eigenvectors[:, 0].real
        # normalize the eigenvector
        self.leading_eigenvector_norm = leading_eigenvector / leading_eigenvector.sum()

        return self.leading_eigenvector_norm

    def getGraph(self, ind):
        """ Given an index of the adjacency matrix returns what graph it belongs to 

        Args:
            ind (int): adj matrix index 

        Returns:
            graph: returns the corresponding graph
        """
        '''
       
        '''
        for g in self.bounds:
            low, up = self.bounds[g]

            if ind <= up and low <= ind:
                return g

    def formattedRanks(self, eigenVects):
        """formats eigen vector to display with corresponding node labels

        Returns:
            dict: ranked eigenvectors and their labels
        """
        rankings = defaultdict(dict)
        for i, val in enumerate(eigenVects):
            g = self.getGraph(i)
            node = self.ns[g] - (self.bounds[g][1] - i) - 1
            name = self.gs[g].vs[node]['name']
            rankings[g][name] = val
        return rankings

    def getLabels(self):
        """gets labels for each graph

        Returns:
            list: ordered list of labels 
        """
        labs = []
        for g in self.gs:
            for node in self.gs[g].vs:
                labs.append(node['name'])
        return labs

    def adjDF(self, matrix, f=None):
        """Creates a df of the adj matrix or personal matrix with corresponding node labels

        Args:
            matrix (dense matrix): an adj matrix or personal matrix to transform
            f (str):Optional if you wish to write the df to an output csv

        Returns:
           pandas df 
        """
        if not self.labs:
            self.labs = self.getLabels()
        nodes = pd.DataFrame(data=matrix, columns=self.labs, index=self.labs)
        if f: nodes.to_csv(f)
        return nodes

    def buildAdjMatrix(self, bidirectional):
        """Creates adj matrix 

        Args:
            bidirectional (bool): wheter the edges are directed or undirected
        """
        n = 0

        '''
            fills up adjacency matrix for inter graph connections using Igraph methods
            '''
        for f, g in zip(self.filesNoExt, self.gs):
            for edge in self.gs[g].get_edgelist():
                self.adj_matrix[self.gs[f].vs.find(
                    edge[0]).index + n, self.gs[f].vs.find(edge[1]).index + n] = 1
            self.startingInd[f] = n
            n += self.ns[f]
        if bidirectional:
            self.adj_matrix += self.adj_matrix.T
        '''
            Makes bounds for when a graph starts and ends
            '''
        self.makeBounds()

        '''
            Fills up intra layer edges
            '''
        for com in self.common_nodes:
            for f1, f2 in permutations(self.filesNoExt, 2):
                s1 = self.startingInd[f1]
                s2 = self.startingInd[f2]
                ind = self.common_nodes[com]
                self.adj_matrix[ind + s1, ind + s2] = 1

    def construct_personal_matrix(self):
        """
        Constructs personal matrix 
        """
        personalDF = pd.read_csv(self.personal_file, dtype=str, header=None)

        for _, row in personalDF.iterrows():
            r = row[row.notna()]
            assert (
                len(r) == 1), f"Incorrect input format at row {r} in file {self.personal}"
            for graph in self.gs:
                ind = self.gs[graph].vs.find(f'{r[0]}').index
                start = self.startingInd[graph]
                _, cs = self.adj_matrix[start + ind, :].nonzero()
                self.personal[ind + start, start + ind] = 1
                for i in cs:
                    g = self.getGraph(i)
                    g2 = self.getGraph(start + ind)
                    if g != g2:
                        self.personal[ind + start, i] = 1



class MultiLayerRanker:

    def __init__(self, layer_files, common_nodes_file, personal_file, biderectional=False, sparse=True):
        """Initializer for Multi Layer Ranker

        Args:
            layer_files (list): list of layer files 
            common_nodes_file (str): csv file to common nodes 
            personal_file (str): file to create personal matrix 
            biderectional (bool, optional): wheter edges are biderectional or not. Defaults to False.
            sparse (bool, optional): use sparse or desnse matrix. Defaults to True.
        """
        assert reduce(
            and_, [f[-5:] == '.ncol' for f in layer_files]), "File not in ncol format"
        self.files = layer_files
        self.filesNoExt = [f.strip(".ncol")for f in self.files]
        self.sparse = sparse
        self.personal_file = personal_file
        self.labs = []
        self.common_nodes = {}
        self.common = pd.read_csv(common_nodes_file, header=None)
        self.gs = {}
        # create independant layers
        for f in self.files:
            g = Graph()
            i = 0

            for _, node in self.common.iterrows():
                g.add_vertex(f'{node[0]}')
                self.common_nodes[f'{node[0]}'] = i
                i += 1

            with open(f) as graphFile:
                for line in graphFile:
                    u, v = line.split()
                    for n in [u, v]:
                        if not len(g.vs.select(name=f'{n}')):
                            g.add_vertex(n)
                    g.add_edge(u, v)
            self.gs[f.strip('.ncol')] = g
        self.ns = {f: self.gs[f].vcount() for f in self.filesNoExt}
        self.N = sum([self.ns.get(i) for i in self.ns])
        self.startingInd = {}
        self.layers = []
        self.bounds = {}
        if sparse:
            self.adj_matrix = lil_matrix((self.N, self.N), dtype=np.int8)
            self.personal = lil_matrix((self.N, self.N), dtype=np.int8)
        else:
            self.adj_matrix = np.zeros((self.N, self.N))
            self.personal = np.zeros((self.N, self.N))
        # build adj matrix
        self.buildAdjMatrix(biderectional)
        # build personal matrix
        self.construct_personal_matrix()

    def makeBounds(self):
        """
        Helper function to create the bounds of each layer in the adj Matrix
        """
        for i, f in enumerate(reversed(self.filesNoExt)):

            if i == 0:
                prev = f
                self.bounds[f] = (self.startingInd[f], self.N - 1)
                continue

            self.bounds[f] = (self.startingInd[f], self.startingInd[prev]-1)
            prev = f

    def pageRank(self, alpha=.85):
        """
        General personalized page rank given the adjacency matrix, personal matrix and alpha score

        Args:
            alpha (int): page rank exploration parameter, defaults to .85 

        Returns:
            array: leading eigen vector corresponding to the rank of each node 
        """

        matrix = self.adj_matrix

        matrix = normalize(matrix, norm='l1', axis=0)

        matrix = alpha * matrix + (1 - alpha) / \
            self.personal.sum() * self.personal

        self.supra = matrix
        _, leading_eigenvectors = eigs(matrix, 1)

        # do we need to be conserned about img (complex numbers!)
        leading_eigenvector = leading_eigenvectors[:, 0].real
        # normalize the eigenvector
        self.leading_eigenvector_norm = leading_eigenvector / leading_eigenvector.sum()

        return self.leading_eigenvector_norm

    def getGraph(self, ind):
        """ Given an index of the adjacency matrix returns what graph it belongs to 

        Args:
            ind (int): adj matrix index 

        Returns:
            graph: returns the corresponding graph
        """
        '''
       
        '''
        for g in self.bounds:
            low, up = self.bounds[g]

            if ind <= up and low <= ind:
                return g

    def formattedRanks(self, eigenVects):
        """formats eigen vector to display with corresponding node labels

        Returns:
            dict: ranked eigenvectors and their labels
        """
        rankings = defaultdict(dict)
        for i, val in enumerate(eigenVects):
            g = self.getGraph(i)
            node = self.ns[g] - (self.bounds[g][1] - i) - 1
            name = self.gs[g].vs[node]['name']
            rankings[g][name] = val
        return rankings

    def getLabels(self):
        """gets labels for each graph

        Returns:
            list: ordered list of labels 
        """
        labs = []
        for g in self.gs:
            for node in self.gs[g].vs:
                labs.append(node['name'])
        return labs

    def toDf(self, matrix, f=None):
        """Creates a df of the adj matrix or personal matrix with corresponding node labels

        Args:
            matrix (dense matrix): an adj matrix or personal matrix to transform
            f (str):Optional if you wish to write the df to an output csv

        Returns:
           pandas df 
        """
        if issparse(matrix):
            matrix = matrix.toarray()
        if not self.labs:
            self.labs = self.getLabels()
        nodes = pd.DataFrame(data=matrix, columns=self.labs, index=self.labs)
        if f: nodes.to_csv(f)
        return nodes

    def buildAdjMatrix(self, bidirectional):
        """Creates adj matrix 

        Args:
            bidirectional (bool): wheter the edges are directed or undirected
        """
        n = 0

        '''
            fills up adjacency matrix for inter graph connections using Igraph methods
            '''
        for f, g in zip(self.filesNoExt, self.gs):
            for edge in self.gs[g].get_edgelist():
                self.adj_matrix[self.gs[f].vs.find(
                    edge[0]).index + n, self.gs[f].vs.find(edge[1]).index + n] = 1
            self.startingInd[f] = n
            n += self.ns[f]
        if bidirectional:
            self.adj_matrix += self.adj_matrix.T
        '''
            Makes bounds for when a graph starts and ends
            '''
        self.makeBounds()

        '''
            Fills up intra layer edges
            '''
        for com in self.common_nodes:
            for f1, f2 in permutations(self.filesNoExt, 2):
                s1 = self.startingInd[f1]
                s2 = self.startingInd[f2]
                ind = self.common_nodes[com]
                self.adj_matrix[ind + s1, ind + s2] = 1

    def construct_personal_matrix(self):
        """
        Constructs personal matrix 
        """
        personalDF = pd.read_csv(self.personal_file, dtype=str, header=None)

        for _, row in personalDF.iterrows():
            r = row[row.notna()]
            assert (
                len(r) == 1), f"Incorrect input format at row {r} in file {self.personal}"
            for graph in self.gs:
                ind = self.gs[graph].vs.find(f'{r[0]}').index
                start = self.startingInd[graph]
                _, cs = self.adj_matrix[start + ind, :].nonzero()
                self.personal[ind + start, start + ind] = 1
                for i in cs:
                    g = self.getGraph(i)
                    g2 = self.getGraph(start + ind)
                    if g != g2:
                        self.personal[ind + start, i] = 1

