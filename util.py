import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import names
import random

BG_COLOR_WHITE = '#ffffff'
BG_COLOR_GRAY = '#e6e6e6'
LINE_COLOR = '#e6e6e6'

def generate_username():
    username = names.get_full_name().lower().replace(' ', '_')
    for _ in range(random.randint(0, 5)):
        username += str(random.randint(0, 9))
    return username
    

def gen_adjacency_matrix(connections):
    user_num = len(connections.keys())
    matrix = np.zeros((user_num, user_num), dtype='int64')
    matrix += user_num + 1
    for fid in connections.keys():
        matrix[fid][fid] = 0
        for ffid in connections[fid]:
            matrix[fid][ffid] = 1
    return matrix


def floyd_warshall(dist, connections):
    n = dist.shape[0]
    for k in range(n):
        for u in range(n):
            for v in range(n):
                dist[u][v] = min(dist[u][v], dist[u][k] + dist[k][v])
                dist[v][u] = dist[u][v]
    return dist


def get_cluster_dist(dist, cluster):
    cluster_dist = np.take(dist, cluster, axis=0)
    cluster_dist = np.take(cluster_dist, cluster, axis=1)
    return cluster_dist


def get_dist_score(cluster_dist):
    dist_score = cluster_dist.sum(axis=1) / len(cluster_dist)
    dist_score += cluster_dist.min(axis=1) * 0.7
    return dist_score


def get_centroids(clusters, dist):
    centroids = [0] * len(clusters)
    for i, cluster in enumerate(clusters):
        cluster_dist = get_cluster_dist(dist, cluster)
        dist_score = get_dist_score(cluster_dist)
        centroids[i] = cluster[np.where(dist_score == min(dist_score))[0][0]]
    return centroids


def merge_clusters(clusters, centroids, dist, threshold):
    merged = False
    for i in range(len(clusters)):

        min_loss = len(clusters) + 1
        min_j = -1

        for j in range(i + 1, len(clusters)):

            tmp = sum(np.take(dist[:, centroids[i]], clusters[j]))
            tmp = tmp / len(clusters[j])

            if tmp <= threshold and tmp < min_loss:
                min_j = j
                min_loss = tmp

        if min_j != -1:
            centroids.pop(min_j)
            clusters[i] += clusters.pop(min_j)
            merged = True

    return clusters, centroids, merged


def split_cluster(centroids, clusters, dist, threshold=4):
    for i, cluster in enumerate(clusters):
        cluster_dist = get_cluster_dist(dist, cluster)
        row, col = np.unravel_index(cluster_dist.argmax(), cluster_dist.shape)
        if cluster_dist[row, col] >= threshold or cluster_dist.mean() >= threshold/2:
            centroids.pop(i)
            centroids.append(cluster[row])
            centroids.append(cluster[col])
            return centroids, True
    return centroids, False


def assign_cluster(node, centroids, dist):
    min_centroids, min_dist = 0, dist[centroids[0]][node]
    for i in range(1, len(centroids)):
        if dist[node][centroids[i]] < min_dist:
            min_centroids = i
            min_dist = dist[centroids[i]][node]
    return min_centroids


def form_clusters(centroids, node_list, dist):
    clusters = [[] for i in range(len(centroids))]
    for node in node_list:
        i = assign_cluster(node, centroids, dist)
        clusters[i].append(node)
    return clusters


def converge(centroids, node_list, dist):
    for _ in range(3):
        clusters = form_clusters(centroids, node_list, dist)
        centroids = get_centroids(clusters, dist)
    return clusters, centroids


def kmeans(dist, node_list, cluster_no, merge_threshold, split_threshold):

    max_iter = 100
    clusters = [[node] for node in node_list]
    centroids = get_centroids(clusters, dist)

    merged = True
    
    for i in range(max_iter):

        if not merged:
            return clusters
        
        clusters, centroids, merged = merge_clusters(clusters, centroids, dist, merge_threshold)
        centroids = get_centroids(clusters, dist)

        split = True
        while split:
            centroids, split = split_cluster(centroids, clusters, dist, split_threshold)
            clusters, centroids = converge(centroids, node_list, dist)

    return clusters


def plot_network(adjacencym, clusters, legend_names, id_to_name, seed=None):

    marker_colors = px.colors.qualitative.T10 + px.colors.qualitative.Set2

    traces = []
    G = nx.from_numpy_matrix(adjacencym) 
    pos = nx.spring_layout(G, seed=seed)
    
    # edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.4, color='rgba(190, 190, 190, 0.5)'),
        hoverinfo='none',
        mode='lines',
        name='connection')

    traces.append(edge_trace)
    
    # node traces
    for i, cluster in enumerate(clusters):
        node_x = []
        node_y = []
        for node in cluster:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, 
            y=node_y,
            mode='markers',
            text=[id_to_name[uid] for uid in cluster],
            hovertemplate='%{text}<br>(' + legend_names[i] + ')<extra></extra>',
            marker=dict(color=marker_colors[i], size=7), 
            name=legend_names[i])

        traces.append(node_trace)

    # configure layout
    layout = go.Layout(
        font=dict(family='Georgia'),
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=go.layout.Margin(l=0, r=0, b=0, t=0),
        paper_bgcolor=BG_COLOR_WHITE,
        plot_bgcolor=BG_COLOR_WHITE)

    # create figure
    fig = go.Figure(data=traces, layout=layout)

    # fig.update_layout(plot_bgcolor='#ffffff')
    
    return fig


def plot_users(user_stats):

    fig = go.Figure()

    trace1 = go.Scatter(
        x=user_stats[0], 
        y=user_stats[1],
        mode='lines',
        text=user_stats[1],
        hoverinfo='x+y',
        name='Over my network size',
        line=dict(color='#3b738f'))

    trace2 = go.Scatter(
        x=user_stats[0], 
        y=user_stats[2],
        mode='lines',
        text=user_stats[2],
        hoverinfo='x+y',
        line=dict(color='#ee8080'),
        name="Over friend's network size")

    layout = go.Layout(
        font=dict(family='Georgia'),
        margin=go.layout.Margin(l=0, r=0, b=30, t=0),
        showlegend = True,
        hovermode  = 'x',
        paper_bgcolor=BG_COLOR_WHITE,
        plot_bgcolor=BG_COLOR_WHITE,

        xaxis = dict(
            title="User",
            showspikes=True,
            spikemode='across+toaxis',
            spikedash='solid',
            showline=True,
            showgrid=False, 
            linecolor=LINE_COLOR,
            showticklabels=False),

        yaxis = dict(
            title="Percentage",
            fixedrange=True,
            showline=True,
            showgrid=True, 
            gridcolor=LINE_COLOR,
            linecolor=LINE_COLOR,
            showticklabels=True),

        legend = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1))

    fig = go.Figure(data=[trace1, trace2], layout=layout)

    return fig


def plot_clusters(sizes, maxs, mins, avgs, cluster_names):

    cluster_ids = list(range(1, len(sizes) + 1))

    traces = []

    for i in cluster_ids:
        if mins[i - i] != maxs[i - 1]:
            traces.append(go.Scatter(
                hoverinfo='none',
                x=[i, i], y=[mins[i - 1], maxs[i - 1]],
                line=dict(width=2, color='gray'),
                mode='lines',
                showlegend=False))

    traces.append(go.Scatter(
        x=cluster_ids,
        y=avgs,
        mode="markers",
        showlegend=False,
        text = avgs,
        hovertemplate='Avg: %{text}<extra></extra>',
        marker=dict(
            color=sizes, 
            colorbar=dict(thickness=10, title='Size'), 
            coloraxis = "coloraxis",
            size=10, 
            showscale=True)))

    traces.append(go.Scatter(
        x=cluster_ids,
        y=mins,
        mode="markers",
        showlegend=False,
        text = mins,
        hovertemplate='Min: %{text}<extra></extra>',
        marker=dict(color=sizes, coloraxis = "coloraxis", size=10)))

    traces.append(go.Scatter(
        x=cluster_ids,
        y=maxs,
        mode="markers",
        showlegend=False,
        text = maxs,
        hovertemplate='Max: %{text}<extra></extra>',
        marker=dict(color=sizes, coloraxis = "coloraxis", size=10)))

    
    fig = go.Figure(data=traces)

    layout = go.Layout(
        font=dict(family='Georgia'),
        showlegend = True,
        hovermode  = 'x',
        coloraxis = {'colorscale':'teal'},
        xaxis = dict(
            title="Cluster",
            fixedrange=False,
            showspikes=False,
            spikemode='across+toaxis',
            showline=True,
            showgrid=False, 
            linecolor=LINE_COLOR,
            showticklabels=True),

        yaxis = dict(
            title="Shortest path between users",
            fixedrange=True,
            showline=True,
            showgrid=True, 
            gridcolor=LINE_COLOR,
            linecolor=LINE_COLOR,
            showticklabels=True),
            margin=go.layout.Margin(l=0, r=0, b=0, t=0),
            paper_bgcolor=BG_COLOR_WHITE,
            plot_bgcolor=BG_COLOR_WHITE)
    fig.update_layout(layout)

    return fig



def plot_closeness(closeness):

    cluster_ids = list(range(1, len(closeness) + 1))
    
    trace = go.Heatmap(
        z=closeness, 
        x=cluster_ids, 
        y=cluster_ids, 
        hovertemplate='cluster %{x}, cluster %{y}<br>Closeness: %{z} / 1<extra></extra>',
        coloraxis = "coloraxis",
        colorbar=dict(thickness=10, title='Deg'))
    
    layout = go.Layout(
        font=dict(family='Georgia'),
        xaxis=dict(showgrid=False, fixedrange=True, linecolor=LINE_COLOR, title='Cluster'),
        yaxis=dict(showgrid=False, fixedrange=True, linecolor=LINE_COLOR, title='Cluster'),
        margin=go.layout.Margin(l=0, r=0, b=0, t=0),
        paper_bgcolor=BG_COLOR_WHITE,
        plot_bgcolor=BG_COLOR_WHITE,
        coloraxis = {'colorscale':'teal'})

    fig = go.Figure(data=trace, layout=layout)

    return fig




























