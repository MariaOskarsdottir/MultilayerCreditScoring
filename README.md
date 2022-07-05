# MuLP

This repository/package includes a python script that implements the MultilayerCreditScoring (MCS) algorithm presented in the paper [Evolution of Credit Risk Using a Personalized Pagerank Algorithm for Multilayer Networks](https://arxiv.org/abs/2005.12418).

# Installation

```
pip install MuLP
```

# Input instructions

There are three primary input files: 

* Individual layer files (.ncol)
* Common Nodes file (csv)
* Personal Node file (csv)

Each layer in the multilayer network requires its own .ncol file with the appropriate [ncol file format](http://lgl.sourceforge.net).

Example ncol layer file (.ncol):

```
CommonNodeA SpecificNodeA
CommonNodeB SpecificNodeA
CommonNodeC SpecificNodeB
CommonNodeD SpecificNodeC
```

The inter-layer connections are only allowed between common nodes as to follow the structure layed out by Óskarsdóttir & Bravo. Due to this one must specify what the common nodes are in the following format:

Example input file(.csv): 
```
CommonNode1
CommonNode2
CommonNode3
```
To construct the personal matrix one must specify the influence (or personal) nodes in the following format: 

Example input file(.csv): 

```
InfluentialNode1
InfluentialNode2
InfluentialNode3
```

# Usage 

### Multilayer Network Initialization
To create a Multilayer Network the following arguments are available: 

```layer_files (list)```: list of layer files 

```common_nodes_file (str)```: csv file to common nodes 

```personal_file (str)```: file to create personal matrix 

```biderectional (bool, optional)```: wheter edges are biderectional or not. Defaults to False.

```sparse (bool, optional)```: use sparse or desnse matrix. Defaults to True.

```python

from MultiLayerRanker import MultiLayerRanker
ranker = MultiLayerRanker(layer_files=['products.ncol','districts.ncol'],
                           common_nodes_file= './common.csv',
                           personal_file= './personal.csv' ,
                           biderectional=True,
                           sparse = True)
```
### Ranking

The ```rank``` method of the ```MultiLayerRanker``` class runs the 
MultiLayer Personalized Page Rank Algorithm. One can choose to run different experiments with varyin alphas by specifying it in the method call: 

```alpha (int,optional)```: page rank exploration parameter, defaults to .85  

```python
eigs = ranker.pageRank(alpha = .85)
```

This method returns the leading eigenvector corresponding to each node's rank. 

### Output Formatting

The ```formattedRanks``` method allows you to get the rankings with appropriate node labels in a dictionary format: x
 

```eigs (ndarray)```: corresponding eigenvector to format 

```python
ranker.formattedRanks(eigs)
```

The  ```adjDF``` method allows you to view format a personal or adjacency matrix with corresponding labels as a dataframe: 

```matrix (ndarray)``` : an adj matrix or personal matrix to transform

```f (str,optional)```: Optional if you wish to write the df to an output csv

```python 
#for persoanl matrix
personalDF = ranker.toDf(ranker.personal)
#for adj matrix
adjDf = ranker.toDf(ranker.matrix)
```






