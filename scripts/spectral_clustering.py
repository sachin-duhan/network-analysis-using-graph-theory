import os
import copy
import json
from util import *

import dash
import dash_core_components as dcc
import dash_html_components as html

from sklearn.cluster import SpectralClustering
from sklearn.utils.validation import check_symmetric

data_dir = 'data'
my_username = 'k_xuanlim'

connections = {}
users = [file[:-4] for file in os.listdir(data_dir)]
users_num = len(users)
id_to_name = dict(zip([i for i in range(users_num)], users))
name_to_id = dict(zip(users, [i for i in range(users_num)]))

# get connections
for file in os.listdir(data_dir):
    
    f = open(os.path.join(data_dir, file))
    ls = []
    for line in f:
        ls.append(line.strip())
    f.close()

    username = file[:-4]
    uid = name_to_id[username]
    connections[uid] = []

    shared_friends = set(ls).intersection(users)
    hashed_ff = [name_to_id[friend] for friend in shared_friends]
    connections[uid] = hashed_ff

adjacencym = gen_adjacency_matrix(connections)
dist = floyd_warshall(copy.deepcopy(adjacencym), connections)

cluster_n = 10
sc = SpectralClustering(cluster_n, affinity='precomputed', n_init=100, assign_labels='discretize')
x = sc.fit_predict(dist)

clusters = [[] for _ in range(cluster_n)]
for i in range(len(x)):
    clusters[x[i]].append(i)

adjacencym[adjacencym == users_num + 1] = 0

cluster_names = ['cluster ' + str(i + 1) for i in range(len(clusters))]
fig = plot_network(adjacencym, clusters, cluster_names, id_to_name)
fig.show()