import sys, os

sys.path.append("../credit_scoring")

from credit_scoring import CreditScoring

data_directory = '.'

files = ['products.csv', 'areas.csv', 'districts.csv']

file_paths = [os.path.join(data_directory, f) for f in files]

personal_file = os.path.join(data_directory, 'll.csv')

mlcs = CreditScoring(file_paths, personal_file)

mlcs.print_stats()

#print(mlcs.multilevel_graph)
#
#mlcs.supra_transition_matrix
#
#print(type(mlcs.supra_transition_matrix))
