# Network Analysis using Graph Theory

This project helps us to better understand our social media network by detecting and analysing clusters within our social media friends.

## Data collection
Follower and followee lists are collected using [Instaloader](https://instaloader.github.io). Two users are considered to be connected when they are following each other so that the graph is non-directional. My network only includes users whom I connect with.

## Analysis

### Cluster detection
Hierarchical k-means clustering is used to form clusters. Each user is initialised to be an independent cluster, then in each iteration, clusters that are close to each other are merged, and clusters having at least one pair of users with distance higher than some threshold are split into two. 

### Closeness between clusters
When two clusters are 'closer', they are more likely to be merged into one social group. Two clusters are considered to be closer when, the score of the cluster (see next section) formed when they are merged increase by a smaller percentage.

### Metric
Distance between two users is measured using the shortest path between users, i.e. the minimum number of profiles a user has to visit before the target user's profile can be reached.

User in a cluster is ranked by a combination of: 
- Mean distance to reach other users in the same cluster
- Minimum distance to reach any one of the user in the cluster

The user with the lowest score will be the centroid of the cluster.

Cluster is the total of scores of user within the cluster.

## Visualisation
The network graph is created using [NetworkX](https://networkx.github.io) and [Plotly](https://github.com/plotly).
