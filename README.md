# MultilayerCreditScoring

This repository/package includes a python script that implements the MultilayerCreditScoring (MCS) algorithim presented in the paper [Evolution of Credit Risk Using a Personalized Pagerank Algorithm for Multilayer Networks](https://arxiv.org/abs/2005.12418).

# installation

```
pip install multilayer-credit-scoring
```

Since one dependancy, **multinetx** is not a published PyPI package, this additional step is needed so everything runs smoothly.

```
pip install git+https://github.com/nkoub/multinetx.git
```

# Usage instructions

What is *currently* supported regarding the format of the input data:

> The multilayer network can be built from 2 column csv files, one for each layer. Here the common nodes are arranged in the frist column and the nodes specific to the layer in question are in the latter column. The nodes present in each row of the file form an edge between them in the bipartite network generated.

Example csv layer file (layer1.csv):

```
CommonNodeA SpecificNodeA
CommonNodeB SpecificNodeA
CommonNodeC SpecificNodeB
CommonNodeD SpecificNodeC
```

Having a list of paths to such files and a file listing the defaulters for the Personilazation matrix enables us the calculate the rankings like so:

```python

from credit_scoring improt CreditScoring

mlcs = CreditScoring(['./layer1.csv', './layer2.csv'], './defaulters.txt', alpha = 0.85)

mlcs.print_stats()

```
The `alpha` parameter to the CreditScoring constructor is an optional one and defaults to the value 0.85. It is used in the personalized pagerank algorithm.

`print_stats()` prints some counting figures for nodes and links of the networks generated.

To access the results, query the following parameters of the CreditScoring class instance:

```common_nodes_rankings``` gives a dictionary from common node identifiers (as seen the csv layer files) to the aggregated rankings.
```layer_specific_node_rankings``` gives a list of dictionaries (one for each layer) for the rankings of the nodes specific to that layer.

REMAINING IN THE IMPLEMENTATION

+ Network_type, defaUlt is bipartite multilayer but mulitplex can also be specified !?! //TODO: check this further.
+ Measures to ensure the same set of common nodes in each layer... if needed.
+ if we want to turn of the printouts, we could have a default verbose flag to the constructor.

