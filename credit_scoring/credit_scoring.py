import csv, os
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

    def __init__(self, layer_files, alpha = 0.85,sparse = True,  verbose = False, common_nodes = None):
        assert reduce(and_ , [f[-5:] == '.ncol' for f in layer_files]), "File not in ncol format"   
        self.files = layer_files
        self.filesNoExt = [f.strip(".ncol")for f in self.files]
        self.verbose = verbose
        self.timer = Timer()
        self.sparse = sparse
        self.labs = []
        if common_nodes: 
            common = pd.read_csv(common_nodes,header = None)
            self.gs = {}
            for f in self.files: 
                g = Graph()
                for _,node in common.iterrows(): 
                    g.add_vertex(f'{node[0]}')
                with open(f) as graphFile: 
                    for line in graphFile:
                        u, v= line.split()
                        for n in [u,v]:
                            if not len(g.vs.select(name=f'{n}')):
                                g.add_vertex(n)
                        g.add_edge(u,v)
                self.gs[f.strip('.ncol')] = g 
            
        else: 
            self.gs = { f.strip('.ncol') : Graph.Read_Ncol(f) for f in self.files}

        self.ns = { f: self.gs[f].vcount() for f in self.filesNoExt }
        self.N = sum([self.ns.get(i) for i in self.ns])
        self.startingInd = {}
        self.layers = []
        self.bounds = {}


       
        if sparse: 
            self.adj_matrix = lil_matrix((self.N,self.N), dtype=np.int8)
            self.personal = lil_matrix((self.N,self.N), dtype=np.int8)
        else: 
            self.adj_matrix = np.zeros((self.N,self.N))
            self.personal = np.zeros((self.N,self.N))


    def makeBounds(self): 
        for i,f in enumerate(reversed(self.filesNoExt)): 

            if i == 0: 
                prev = f
                self.bounds[f] = (self.startingInd[f],self.N - 1)
                continue

            self.bounds[f] = (self.startingInd[f],self.startingInd[prev]-1)
            prev = f



    def buildAdjMatrix(self,bidirectional,intraFIle): 
        intraDF = pd.read_csv(intraFIle)
        n=0
        '''
        fills up adjacency matrix for inter graph connections using Igraph methods
        
        '''
        for f,g in zip(self.filesNoExt,self.gs): 
            for edge in self.gs[g].get_edgelist(): 
                self.adj_matrix[self.gs[f].vs.find(edge[0]).index + n ,self.gs[f].vs.find(edge[1]).index  + n ] = 1 
            self.startingInd[f] = n
            n+= self.ns[f]

        '''
        Makes bounds for when a graph starts and ends
        '''
        self.makeBounds()

        '''
        Fills up intra layer edges
        '''
        for _,row in intraDF.iterrows():
            '''
            Grab edge ids from both graphs
            Then input into the correct column based on origin node 
            '''
            r = row[row.notna()]
            assert (len(r) == 3) , f"Incorrect input format at row {r} in file {intraFIle}"
            e1,e2,d = r 
            ind = r.index
            e1G = self.gs.get(ind[0])
            e2G = self.gs.get(ind[1])

            e1Ind = e1G.vs.find(f'{e1}').index
            e2Ind = e2G.vs.find(f'{e2}').index
            if d == 1 or d == 0 : 
                self.adj_matrix[ e1Ind + self.startingInd[ind[0]],e2Ind + self.startingInd[ind[1]]] = 1 
            elif d == -1: 
                self.adj_matrix[e1Ind + self.startingInd[ind[1]] , e1Ind + self.startingInd[ind[0]]] = 1 
        if bidirectional: self.adj_matrix+= self.adj_matrix.T
        return self.adj_matrix

    def pageRank(self,alpha,matrix,personal): 
        '''
        General personalized page rank given the adjacency matrix, personal matrix and alpha score
        
        '''
        matrix = normalize( matrix, norm='l1', axis=0   )        
        if self.verbose: self.timer.report_time('Adj matrix col normalized')
         

        matrix = alpha * matrix + (1 - alpha)/personal.sum() * personal
       
        if self.verbose: self.timer.report_time('Supra trans matrix calculated')

        self.supra = matrix
        _, leading_eigenvectors = eigs(matrix, 1)


        # do we need to be conserned about img (complex numbers!)
        leading_eigenvector = leading_eigenvectors[:, 0].real
        # normalize the eigenvector
        self.leading_eigenvector_norm = leading_eigenvector / leading_eigenvector.sum() 
        pprint.pprint(f'{leading_eigenvector=}')
        print(leading_eigenvector.sum() )
        return self.leading_eigenvector_norm 

  

    
    def getGraph(self , ind):
        '''
        Given an index of the adjacency matrix returns what graph it belongs to 
        '''
        for g in self.bounds:
            low , up = self.bounds[g]

            if ind <= up and low <= ind : 
                return g


    def rank(self,eigenVects):
        '''
        gathers node ids using their index in the leading eigenvector to return final page rank scores
        '''
        rankings = defaultdict(dict)
        for i,val in enumerate(eigenVects):
            g = self.getGraph(i)
            node = self.ns[g] - (self.bounds[g][1] - i  ) - 1
            name = self.gs[g].vs[node]['name']
            rankings[g][name] = val
        return rankings


    def getLabels(self):
        labs = [ ]
        for g in self.gs:
            for node in self.gs[g].vs:
                labs.append(node['name'])
        return labs


    def writeCSV(self,matrix,f): 
        if not self.labs:
            self.labs = self.getLabels()
        
        return pd.DataFrame(data = matrix, columns= self.labs, index =self.labs ).to_csv(f)



    def construct_personal_matrix(self, file_name, labels = None):
        '''
        constructs personal matrix given a csv file 
        csv file format: 

        l1,l2,l3,d
        1,,1,1
    
        Important to have null/missing value in the layer that is not getting an edge added
        the above represents a directed edge from l1 node 1 -> l2 node 1 

        To represent l2 node 1 -> l1 node 1 we can simply do that by indicating a negative direction in the direction column: 

        l1,l2,l3,d
        1,,1,-1

        '''
        labels = self.getLabels()
        personalDF = pd.read_csv(file_name,dtype = str)
        for _,row in personalDF.iterrows():
            r = row[row.notna()]

            assert (len(r) == 1) , f"Incorrect input format at row {r} in file {file_name}"
            lab = r.index
            ind  = self.gs[lab[0]].vs.find(f'{r[0]}').index
            start = self.startingInd[lab[0]]
            _,cs = self.adj_matrix[ start + ind , :  ].nonzero()
            self.personal[ind + start , start + ind ] = 1
            t = [labels[i] for i in cs]

            for i in  cs: 
                g = self.getGraph(i)
                g2 = self.getGraph(start + ind)
                if g != g2:
                    diag = self.ns[g] - (self.bounds[g][1] - i  ) - 1
                    
                    self.personal[start + diag,i] += 1
        return self.personal
    def construct_personal_matrix2(self, file_name, labels = None):
        '''
        constructs personal matrix given a csv file 
        csv file format: 

        l1,l2,l3,d
        1,,1,1
    
        Important to have null/missing value in the layer that is not getting an edge added
        the above represents a directed edge from l1 node 1 -> l2 node 1 

        To represent l2 node 1 -> l1 node 1 we can simply do that by indicating a negative direction in the direction column: 

        l1,l2,l3,d
        1,,1,-1

        '''
        labels = self.getLabels()
        personalDF = pd.read_csv(file_name,dtype = str)
        for _,row in personalDF.iterrows():
            r = row[row.notna()]

            assert (len(r) == 1) , f"Incorrect input format at row {r} in file {file_name}"
            lab = r.index
            ind  = self.gs[lab[0]].vs.find(f'{r[0]}').index
            start = self.startingInd[lab[0]]
            _,cs = self.adj_matrix[ start + ind ,:].nonzero()
            self.personal[ind + start , start + ind ] = 1
            t = [labels[i] for i in cs]

            for i in  cs: 
                g = self.getGraph(i)
                g2 = self.getGraph(start + ind)
                if g != g2:
                    diag = self.ns[g] - (self.bounds[g][1] - i  ) - 1
                    # interest = labels[ind + start ]

                    # print(f'{r[0]=} {cs }{t=} {start=} {diag =} {diag + start=} {labels[i]=} {labels[start + diag]=}  {interest=} '); 
                    self.personal[ind + start,i] += 1
        return self.personal 