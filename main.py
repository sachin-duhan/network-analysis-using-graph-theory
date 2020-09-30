import os
import copy
import json
from util import *

import pandas as pd
import plotly.graph_objects as go

data_dir = 'masked'
my_username = 'k_xuanlim'

connections = {}
users = [file[:-4] for file in os.listdir(data_dir)]
users_num = len(users)
id_to_name = dict(zip([i for i in range(users_num)], users))
name_to_id = dict(zip(users, [i for i in range(users_num)]))

i_count_p1_p2 = []

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

    i_count_p1_p2.append((
            username, 
            round(len(shared_friends) / users_num * 100, 4),
            round(len(shared_friends) / len(ls) * 100, 4),
            len(ls)))

    hashed_ff = [name_to_id[friend] for friend in shared_friends]
    connections[uid] = hashed_ff

user_stats = pd.DataFrame(i_count_p1_p2)
user_stats.sort_values(by=[1], inplace=True)

user_stats.to_csv(os.path.join('output', 'user_stats.csv'))

fig = plot_users(user_stats)
fig.show()


def gen_result(merge_threshold, split_threshold, output_fname):

    cluster_no = users_num
    adjacencym = gen_adjacency_matrix(connections)
    dist = floyd_warshall(copy.deepcopy(adjacencym), connections)

    clusters = kmeans(dist, connections.keys(), cluster_no, merge_threshold, split_threshold)
    adjacencym[adjacencym == users_num + 1] = 0

    # name clusters as cluster i
    cluster_names = ['cluster ' + str(i + 1) for i in range(len(clusters))]

    # store result in dic
    res = {}
    res['my_username'] = my_username
    res['id_to_name'] = id_to_name
    res['adjacencym'] = adjacencym.tolist()
    res['clusters'] = clusters
    res['cluster_names'] = cluster_names

    res['cluster_size'] = []
    res['cluster_max'] = []
    res['cluster_min'] = []
    res['cluster_avg'] = []

    # analyse cluster
    for i, cluster in enumerate(clusters):

        cluster_dist = get_cluster_dist(dist, cluster)

        # basic stats: size, min, max, avg
        res['cluster_size'].append(len(cluster))
        res['cluster_max'].append(int(cluster_dist.max()))
        if len(cluster) > 1:
            res['cluster_min'].append(int(cluster_dist[cluster_dist > 0].min()))
            res['cluster_avg'].append(round(cluster_dist.sum() / (2 * sum(range(len(cluster)))), 4))
        else:
            res['cluster_min'].append(0)
            res['cluster_avg'].append(0)

    # closeness between clusters
    closeness = [[[None] * len(clusters)][0] for _ in range(len(clusters))]
    for i in range(len(clusters)):
        closeness[i][i] = 1
        for j in range(i + 1, len(clusters)):
            cluster_dist_ij = get_cluster_dist(dist, clusters[i] + clusters[j])
            dist_score_ij = sum(get_dist_score(cluster_dist_ij))
            if cluster_dist_ij.max() <= users_num:
                dist_score_i = sum(get_dist_score(get_cluster_dist(dist, clusters[i])))
                dist_score_j = sum(get_dist_score(get_cluster_dist(dist, clusters[j])))
                closeness[i][j] = 1 + round((dist_score_i + dist_score_j - dist_score_ij) / (dist_score_ij), 4)
                closeness[j][i] = closeness[i][j]
    
    res['closeness'] = closeness

    # save result as json file
    with open(os.path.join('output', output_fname), 'w') as f:
        json.dump(res, f)

    # show graphs
    fig = plot_network(adjacencym, clusters, cluster_names, id_to_name)
    fig.show(config={"displayModeBar": False, "showTips": False})

    fig = plot_clusters(res['cluster_size'], res['cluster_max'], res['cluster_min'], res['cluster_avg'], cluster_names)
    fig.show(config={"displayModeBar": False, "showTips": False})

    fig = plot_closeness(closeness, cluster_names)
    fig.show(config={"displayModeBar": False, "showTips": False})


gen_result(merge_threshold=4.5, split_threshold=4.7, output_fname='cluster1.json')
gen_result(merge_threshold=2.2, split_threshold=3.5, output_fname='cluster2.json')
