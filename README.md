# MultilayerCreditScoring

This repository/package includes a python script that implements the MultilayerCreditScoring (MCS) algorithim presented in the paper [Evolution of Credit Risk Using a Personalized Pagerank Algorithm for Multilayer Networks](https://arxiv.org/abs/2005.12418).

# Installation

```
pip install multilayer-credit-scoring
```

Since one dependancy, **multinetx** is not a published PyPI package, this additional step is needed so everything runs smoothly.

```
pip install git+https://github.com/nkoub/multinetx.git
```

# Input instructions

Each layer in the multilayer network requires it's own .ncol file with the appropriate [ncol](http://lgl.sourceforge.net) file format.

Example ncol layer file (layer1.ncol):

```
CommonNodeA SpecificNodeA
CommonNodeB SpecificNodeA
CommonNodeC SpecificNodeB
CommonNodeD SpecificNodeC
```

The inter-layer connections can be built using a csv file with the following format: 

Example inter-layer connection file; 
```
FilenameForLayer1,FilenameForLayer2,FilenameForLayer3,d 
Layer1NodeName,Layer2NodeName,,1
,Layer2NodeName,Laye31NodeName,1
Layer1NodeName,,Layer3NodeName,-1
```

The headers of the csv file MUST MATCH the name of the file of the corresponding layer, ie: if you have a layer1.ncol file, its appropriate name on the inter-layer file would be layer1. Each edge is built individually, so the only non null entries in the csv row should be the names of the edges in the files which you are trying to connect and their direction(d header). A direction of 1 build a link left -> right, layer1 -> layer2, while a -1 tag specifies layer2 -> layer1. 

Having a list of paths to such files and a file listing the defaulters for the Personilazation matrix, enables us to calculate the rankings like so:

Common Nodes (**Optional**)

If your layers have common nodes, you may create a common_nodes.csv file with the following format. Note this input expects **all layers** to have these nodes. 

```csv
nodeName1
nodeName2
nodeName3
```

# Usage 
```python

from credit_scoring improt MLN

mln = MLN( layer_files=['layer1.ncol','layer2.ncol'], alpha = 0.85,sparse = True, common_nodes= 'common.csv')


```
`sparse`: Indicates whether or not you want to use sparse matrices for computations or simply use dense ones. 
`common_nodes`: Optional, defaults to None. Specifies the common_nodes file 

Returns a multilayer network

```python
matrix = mln.buildAdjMatrix(intraFIle= 'intra.csv', bidirectional = True)
```
`bidrectional`: Indicates wheter the node links are directed or undirected

Outputs a Multilayer adjecency matrix as a scipy sparse amtrix or a dense numpy array

To Construct the personal matrix : 

```python
personal = mln.construct_personal_matrix('personal.csv')
```

To run page rank with an appropriate personal matrix and adjecency matrix; 

```
vect = ranker.pageRank(alpha = .85,matrix= matrix, personal = personal)
```
`alpha`: Defaults to the value `0.85`. It is the exploration rate used in the personalized pagerank algorithm.
`adj_matrix`: corresponding adjecency matrix to use
`personal`: corresponding personal matrix to use

returns leading eigen vector 

To view the rankings in a json format: 
```python
mlp.view(vect = vect)
```
`vect`: the leading eigenvector computed

sample output: 
```json
 {
    'LAYER1': 
            {
                'NODE1': -7,
                'NODE2': -4},
    'LAYER2': 
             {
                'NODE1': 5,
                'NODE2': -1,
             }
}

```



