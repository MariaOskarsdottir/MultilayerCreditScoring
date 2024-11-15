import sys
from pprint import pprint
sys.path.append("../")
from MultiLayerRanker import MultiLayerRanker
ranker = MultiLayerRanker(layer_files=['products.ncol','districts.ncol'], common_nodes_file= './common.csv',personal_file= './personal.csv' ,biderectional=True)
eigs = ranker.pageRank(alpha = .85)
pprint(ranker.formattedRanks(eigs))