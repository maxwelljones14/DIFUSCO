import numpy as np
import networkx as nx


def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    x_root = find(parent, x)
    y_root = find(parent, y)

    if rank[x_root] < rank[y_root]:
        parent[x_root] = y_root
    elif rank[x_root] > rank[y_root]:
        parent[y_root] = x_root
    else:
        parent[y_root] = x_root
        rank[x_root] += 1

def kruskal(adj_matrix):
    n = adj_matrix.shape[0]
    parent = [i for i in range(n)]
    rank = [0] * n

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, adj_matrix[i][j]))

    edges.sort(key=lambda x: -x[2])  # Sort edges in descending order of weight

    tree_edges = []
    while len(tree_edges) < n - 1 and edges:
        u, v, weight = edges.pop(0)
        if find(parent, u) != find(parent, v):
            tree_edges.append((u, v))
            union(parent, rank, u, v)

    return np.array(tree_edges)

def max_n_minus_1_edges(adj_matrix):
    n = adj_matrix.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, adj_matrix[i][j]))
    
    weights = np.array([edge[2] for edge in edges])
    partition_idx = np.argpartition(weights, -n+1)[-n+1:]
    max_edges = [edges[i] for i in partition_idx]
    
    return np.array(max_edges)[:,:2]

def evaluate_mst(points, mst):
    mst_length = 0
    for i in range(mst.shape[0]):
            mst_length += np.linalg.norm(points[int(mst[i][0])] - points[int(mst[i][1])])
    return mst_length

def find_mincut(k, nodes_coord, distances, np_edge_index):
    """"
    k: # neighbors for each node
    nodes_coord: coordinates for each node
    """
    num_nodes = len(nodes_coord)
    G = nx.Graph()

    # Add nodes to the graph
    for i in range(num_nodes):
        G.add_node(i)

    #Reconstruct K-NN graph
    for i in range(np_edge_index.shape[1]):
            if np_edge_index[0][i] != np_edge_index[1][i]:  # Avoid self-loops
                G.add_edge(np_edge_index[0][i], np_edge_index[1][i], weight=distances[i])

    
    # Find the minimum cut
    cut_value, partition = nx.stoer_wagner(G)

    #Get mincut edges from partition
    mincut_edges = []
    partition_0, partition_1 = partition
    for i in partition_0:
        for j in partition_1:
            if G.has_edge(i, j):
                mincut_edges.append((i,j))

    return np.array(mincut_edges)

def find_mincut_approximation(heatmap, k, nodes_coord, np_edge_index):
    """
    First creates the KNN graph using k and nodes_coord, with weights retreived from heatmap
    Tries to find an approximate mincut solution by checking edge weights sequentially
    from high to low, until the graph is disconnected
    """
    num_nodes = len(nodes_coord)
    G = nx.Graph()

    # Add nodes to the graph
    for i in range(num_nodes):
        G.add_node(i)

    #Reconstruct K-NN graph
    # nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(nodes_coord)
    # _, indices = nbrs.kneighbors(nodes_coord)

    for i in range(np_edge_index.shape[1]):
            if np_edge_index[0][i] != np_edge_index[1][i]:  # Avoid self-loops
                G.add_edge(np_edge_index[0][i], np_edge_index[1][i], weight=heatmap[i])

    # Sort edges by weight in descending order
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

    cut_edges = []

    # Iterate through the sorted edges
    for edge in sorted_edges:
        G.remove_edge(*edge[:2])
        cut_edges.append((edge[0], edge[1]))

        # Check if the graph is still connected
        if not nx.is_connected(G):
            break

    return np.array(cut_edges)

def find_dijkstra_approximation(heatmap, k, nodes_coord, np_edge_index):
    """
    heatmap: nxn numpy array
    k: # neighbors for each node
    nodes_coord: coordinates for each node

    From the starting node, does DFS, traversing the edge with highest weight
    from the heatmap until it reaches the end node.
    """

    num_nodes = len(nodes_coord)
    G = nx.Graph()

    # Add nodes to the graph
    for i in range(num_nodes):
        G.add_node(i)

    # Reconstruct K-NN graph with weights from heatmap
    # kdt = KDTree(nodes_coord, leaf_size=30, metric='euclidean')
    # _, indices = kdt.query(nodes_coord, k=k, return_distance=True)

    # for i in range(num_nodes):
    #     for j in indices[i]:
    #         if i != j:  # Avoid self-loops
    #             G.add_edge(i, j, weight=heatmap[i][j])

    for i in range(np_edge_index.shape[1]):
            if np_edge_index[0][i] != np_edge_index[1][i]:  # Avoid self-loops
                G.add_edge(np_edge_index[0][i], np_edge_index[1][i], weight=heatmap[i])


    start_node = np.argmin(np.sum(nodes_coord, axis=1))  # bottom left
    end_node = np.argmax(np.sum(nodes_coord, axis=1))  # top right

    # Modified DFS to prioritize edges with highest weights
    def dfs_modified(node, visited, path):
        visited[node] = True
        path.append(node)

        if node == end_node:
            return True

        neighbors = list(G.neighbors(node))
        neighbors.sort(key=lambda x: G[node][x]['weight'], reverse=True)  # Sort neighbors by edge weight

        for neighbor in neighbors:
            if not visited[neighbor]:
                if dfs_modified(neighbor, visited, path):
                    return True

        path.pop()  # Remove node from path if it's a dead end
        return False

    # Initialize visited array and empty path
    visited = [False] * num_nodes
    path = []

    # Perform modified DFS
    dfs_modified(start_node, visited, path)

    # Reconstruct edges along the path
    path_edges = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        path_edges.append([u, v])

    return np.array(path_edges)

def find_dijkstra(k, nodes_coord, np_edge_index):
    """
    k: # neighbors for each node
    nodes_coord: coordinates for each node

    From starting node, uses dijkstra to find shortest path to end node
    """
    num_nodes = len(nodes_coord)
    G = nx.Graph()

    # Add nodes to the graph
    for i in range(num_nodes):
        G.add_node(i)

    # Reconstruct K-NN graph with weights from heatmap
    # kdt = KDTree(nodes_coord, leaf_size=30, metric='euclidean')
    # distances, indices = kdt.query(nodes_coord, k=k, return_distance=True)

    # for i in range(num_nodes):
    #     for j in indices[i]:
    #         if i != j:  # Avoid self-loops
    #             dist = distances[i][list(indices[i]).index(j)]
    #             G.add_edge(i, j, weight=dist)
    for i in range(np_edge_index.shape[1]):
        if np_edge_index[0][i] != np_edge_index[1][i]:  # Avoid self-loops
            G.add_edge(np_edge_index[0][i], np_edge_index[1][i], weight=heatmap[i])

    start_node = np.argmin(np.sum(nodes_coord, axis=1))  # bottom left
    end_node = np.argmax(np.sum(nodes_coord, axis=1))  # top right

    # Find shortest path using Dijkstra's algorithm
    shortest_path = nx.dijkstra_path(G, source=start_node, target=end_node)

    path_edges = []
    for i in range(len(shortest_path) - 1):
        u, v = shortest_path[i], shortest_path[i + 1]
        path_edges.append([u, v])

    return np.array(path_edges)