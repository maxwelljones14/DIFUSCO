import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours
import torch
import imageio
import matplotlib.pyplot as plt
import json
from sklearn.neighbors import KDTree

def dijkstra_with_edge_count(G, source, target):
    """
    Dijkstra's algorithm with edge count tracking.
    """
    # Initialize data structures
    visited = set()
    distance = {node: float('inf') for node in G}
    distance[source] = 0
    previous = {node: None for node in G}
    edges_checked = 0
    
    # Main loop
    while len(visited) < len(G):
        # Find the node with the smallest distance
        min_distance_node = None
        for node in G:
            if node not in visited and (min_distance_node is None or distance[node] < distance[min_distance_node]):
                min_distance_node = node
        
        if min_distance_node == target:
            break
        
        # Update visited set
        visited.add(min_distance_node)
        
        # Relax edges
        for neighbor in G.neighbors(min_distance_node):
            edges_checked += 1
            if neighbor not in visited:
                edge_weight = G[min_distance_node][neighbor]['weight']
                new_distance = distance[min_distance_node] + edge_weight
                if new_distance < distance[neighbor]:
                    distance[neighbor] = new_distance
                    previous[neighbor] = min_distance_node
    
    # Reconstruct path
    path = []
    current = target
    while current is not None:
        path.insert(0, current)
        current = previous[current]
    
    return path, edges_checked

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
    def dfs_modified(node, visited, path, edge_counter):
        visited[node] = True
        path.append(node)

        if node == end_node:
            return True

        neighbors = list(G.neighbors(node))
        neighbors.sort(key=lambda x: G[node][x]['weight'], reverse=True)  # Sort neighbors by edge weight

        for neighbor in neighbors:
            edge_counter[0] += 1  # Increment edge counter
            if not visited[neighbor]:
                if dfs_modified(neighbor, visited, path, edge_counter):
                    return True

        path.pop()  # Remove node from path if it's a dead end
        return False

    # Initialize visited array, empty path, and edge counter
    visited = [False] * num_nodes
    path = []
    edge_counter = [0]  # Use a list to make it mutable within the DFS function

    # Perform modified DFS
    dfs_modified(start_node, visited, path, edge_counter)

    # Reconstruct edges along the path
    path_edges = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        path_edges.append([u, v])

    return np.array(path_edges), edge_counter[0]

def find_dijkstra(heatmap, k, nodes_coord, np_edge_index):
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

    # Find shortest path using modified Dijkstra's algorithm
    shortest_path, edges_checked = dijkstra_with_edge_count(G, source=start_node, target=end_node)

    path_edges = []
    for i in range(len(shortest_path) - 1):
        u, v = shortest_path[i], shortest_path[i + 1]
        path_edges.append([u, v])

    return np.array(path_edges), edges_checked

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
                if np_edge_index[0][i] == 1 or np_edge_index[1][i] == 1:
                    print(np_edge_index[0][i], np_edge_index[1][i])
                G.add_edge(np_edge_index[0][i], np_edge_index[1][i], weight=heatmap[i])

    # Sort edges by weight in descending order
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    # d = {i: [] for i in range(50)}
    # for edge in sorted_edges[0:50]:
    #     d[edge[0]].append(edge[1])
    #     d[edge[1]].append(edge[0])
    # print(d)
    # for edge in [11, 29, 0, 41, 21, 40]:
    #     G.remove_edge(1, edge)
    #     # check if edge exists
    #     if G.has_edge(edge, 1):
    #         G.remove_edge(edge, 1)
    # print(nx.is_connected(G))
    # exit(0)


    cut_edges = []

    # Iterate through the sorted edges
    for edge in sorted_edges:
        G.remove_edge(*edge[:2])
        cut_edges.append((edge[0], edge[1]))

        # Check if the graph is still connected
        if not nx.is_connected(G):

            break

    return np.array(cut_edges)
    

def kruskal(adj_matrix):
    num_cycles_found = 0
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
        else:
            num_cycles_found += 1

    return np.array(tree_edges), num_cycles_found

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

def generate_gif(args):
    if args.val:
        val_or_test = "val"
    else:
        val_or_test = "test"
    path_heatmap = os.path.join(args.path, f"{val_or_test}-heatmap-{args.sample_num}.npy")
    path_locations = os.path.join(args.path, f"{val_or_test}-points-{args.sample_num}.npy")
    # open path as np array:
    with open(path_heatmap, "rb") as f:
        data_heatmap_overall = np.load(f)
    # data_heatmap = data_heatmap[0]
    # print(data_heatmap.shape)
    with open(path_locations, "rb") as f:
        data_locations = np.load(f)
    # print(data_locations.shape)

    # print(data_heatmap_overall.shape)
    overall_images = []
    os.makedirs(args.save_path, exist_ok=True)
    for heatmap_diffusion_num in range(data_heatmap_overall.shape[0]):

        print(data_heatmap_overall.shape)

        data_heatmap = data_heatmap_overall[heatmap_diffusion_num]
        if args.task_name == "tsp":
            tours, merge_iterations = merge_tours(
                data_heatmap, data_locations, None,
                sparse_graph=False,
                parallel_sampling=1,
            )

            # Refine using 2-opt
            solved_tours, ns = batched_two_opt_torch(
                data_locations.astype("float64"), np.array(tours).astype('int64'),
                max_iterations=1000, device=torch.device("cuda:0"))
            # print(solved_tours.shape)

            final_heatmap = np.zeros((data_heatmap.shape[1], data_heatmap.shape[2]))
            for i in range(data_heatmap.shape[1]):
                final_heatmap[solved_tours[0, i], solved_tours[0, i + 1]] = 100
                # print(solved_tours[0, i], solved_tours[0, i + 1])
        elif args.task_name == "mst":
            data_heatmap = data_heatmap[0]
            # row normalize data heatmap
            distances = np.linalg.norm(data_locations[:, None] - data_locations, axis=-1)
            # data_heatmap = data_heatmap / distances
            edges, num_cylces_found =  max_n_minus_1_edges(data_heatmap), 1 # kruskal(data_heatmap) # 
            edge_gt, num_cylces_found_gt = kruskal(np.ones_like(data_heatmap) / distances)
            print(f"Number of cycles found in diffusion num {heatmap_diffusion_num}: {num_cylces_found}, number of cycles in GT: {num_cylces_found_gt}")
            final_heatmap = np.zeros((data_heatmap.shape[0], data_heatmap.shape[1]))
            for i in range(edges.shape[0]):
                final_heatmap[int(edges[i][0])][int(edges[i][1])] = 100
        elif args.task_name == "mincut":
            kdt = KDTree(data_locations, leaf_size=30, metric='euclidean')
            dis_knn, idx_knn = kdt.query(data_locations, k=args.sparse_factor, return_distance=True)
            edge_index_0 = torch.arange(data_locations.shape[0]).reshape((-1, 1)).repeat(1, args.sparse_factor).reshape(-1)
            edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
            edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)
            np_edge_index = edge_index.numpy()

            adj_mat = data_heatmap
            np_points = data_locations
            distances = np.sum((np_points[np_edge_index[0]] - np_points[np_edge_index[1]]) ** 2, axis=1) ** .5
            gt_edges = find_mincut(args.sparse_factor, np_points, distances, np_edge_index)
            gt_cost = np.sum(np.sum((np_points[gt_edges[:, 0]] - np_points[gt_edges[:, 1]]) ** 2, axis=1) ** .5)

            best_solved_edges = find_mincut_approximation(adj_mat / distances, args.sparse_factor, np_points, np_edge_index)
            best_solved_cost = np.sum((np_points[best_solved_edges[:, 0]] - np_points[best_solved_edges[:, 1]]) ** 2) ** .5

            final_heatmap = np.zeros((args.graph_size, args.graph_size))
            for i in range(best_solved_edges.shape[0]):
                final_heatmap[int(best_solved_edges[i][0])][int(best_solved_edges[i][1])] = 100
        elif args.task_name == "djikstra":
            kdt = KDTree(data_locations, leaf_size=30, metric='euclidean')
            dis_knn, idx_knn = kdt.query(data_locations, k=args.sparse_factor, return_distance=True)
            edge_index_0 = torch.arange(data_locations.shape[0]).reshape((-1, 1)).repeat(1, args.sparse_factor).reshape(-1)
            edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
            edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)
            np_edge_index = edge_index.numpy()

            adj_mat = data_heatmap
            np_points = data_locations
            distances = np.sum((np_points[np_edge_index[0]] - np_points[np_edge_index[1]]) ** 2, axis=1) ** .5
            gt_edges, gt_num_searched = find_dijkstra(distances, args.sparse_factor, np_points, np_edge_index)
            
            gt_cost = np.sum(np.sum((np_points[gt_edges[:, 0]] - np_points[gt_edges[:, 1]]) ** 2, axis=1) ** .5)
            #  / distances
            best_solved_edges, num_searched = find_dijkstra_approximation(adj_mat, args.sparse_factor, np_points, np_edge_index)
            print(f"gt_num_searched: {gt_num_searched}, num_searched: {num_searched}")            
            best_solved_cost = np.sum(np.sum((np_points[best_solved_edges[:, 0]] - np_points[best_solved_edges[:, 1]]) ** 2, axis=1) ** .5)
            print(f"gt_cost: {gt_cost}, best_solved_cost: {best_solved_cost}")

            final_heatmap = np.zeros((args.graph_size, args.graph_size))
            for i in range(best_solved_edges.shape[0]):
                final_heatmap[int(best_solved_edges[i][0])][int(best_solved_edges[i][1])] = 100
            for i in range(gt_edges.shape[0]):
                final_heatmap[int(gt_edges[i][0])][int(gt_edges[i][1])] += 101
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_tsp_heatmap(ax, data_locations, final_heatmap, threshold=0.1, title=f"{val_or_test} sample {args.sample_num}")
        fig.savefig(f"{args.save_path}/{val_or_test}-sample-{args.sample_num}_diffusion_num_{heatmap_diffusion_num}.png")
        overall_images.append(imageio.imread(f"{args.save_path}/{val_or_test}-sample-{args.sample_num}_diffusion_num_{heatmap_diffusion_num}.png"))
        # print(f"{args.save_path}/{val_or_test}-sample-{args.sample_num}_diffusion_num_{heatmap_diffusion_num}.png")
    # take all images and save them as single gif
    imageio.mimsave(f"{args.save_path}/{val_or_test}-sample-{args.sample_num}.gif", overall_images)

def get_data(args):
    if args.val:
        val_or_test = "val"
    else:
        val_or_test = "test"
    cycles_unnomralized = np.zeros((args.sample_num, args.num_diffusion_steps))
    cycles = np.zeros((args.sample_num, args.num_diffusion_steps))
    costs_unnormalized = np.zeros((args.sample_num, args.num_diffusion_steps))
    costs = np.zeros((args.sample_num, args.num_diffusion_steps))
    gt_cycles = np.zeros((args.sample_num))
    gt_costs = np.zeros((args.sample_num))
    for sample in range(args.sample_num):
        path_heatmap = os.path.join(args.path, f"{val_or_test}-heatmap-{sample}.npy")
        path_locations = os.path.join(args.path, f"{val_or_test}-points-{sample}.npy")

        with open(path_locations, "rb") as f:
            data_locations = np.load(f)
        # open path as np array:
        with open(path_heatmap, "rb") as f:
            data_heatmap_overall = np.load(f)
        distances = np.linalg.norm(data_locations[:, None] - data_locations, axis=-1)
        if args.task_name == "mst":
            edge_gt, num_cylces_found_gt = kruskal(np.ones_like(data_heatmap_overall[0][0]) / distances)
            cost_gt = sum([distances[int(edge_gt[i][0])][int(edge_gt[i][1])] for i in range(edge_gt.shape[0])])
            gt_cycles[sample] = num_cylces_found_gt
            gt_costs[sample] = cost_gt
        # data_heatmap = data_heatmap[0]
        # print(data_heatmap.shape)

        # print(data_locations.shape)

        # print(data_heatmap_overall.shape)
        os.makedirs(args.save_path, exist_ok=True)
        for heatmap_diffusion_num in range(data_heatmap_overall.shape[0]):
            data_heatmap = data_heatmap_overall[heatmap_diffusion_num]
            if args.task_name == "tsp":
                tours, merge_iterations = merge_tours(
                    data_heatmap, data_locations, None,
                    sparse_graph=False,
                    parallel_sampling=1,
                )

                # Refine using 2-opt
                solved_tours, ns = batched_two_opt_torch(
                    data_locations.astype("float64"), np.array(tours).astype('int64'),
                    max_iterations=1000, device=torch.device("cuda:0"))
                # print(solved_tours.shape)

                final_heatmap = np.zeros((data_heatmap.shape[1], data_heatmap.shape[2]))
                for i in range(data_heatmap.shape[1]):
                    final_heatmap[solved_tours[0, i], solved_tours[0, i + 1]] = 100
                    # print(solved_tours[0, i], solved_tours[0, i + 1])
            elif args.task_name == "mst":
                # print(f"data heatmap shape: {data_heatmap.shape}")
                data_heatmap = data_heatmap[0]
                
                edges_unnormalized, num_cycles_found_unnormalized = kruskal(data_heatmap)
                cost_unnormalized = sum([distances[int(edges_unnormalized[i][0])][int(edges_unnormalized[i][1])] for i in range(edges_unnormalized.shape[0])])
                data_heatmap = data_heatmap / distances
                edges, num_cylces_found = kruskal(data_heatmap) #  # 
                cost = sum([distances[int(edges[i][0])][int(edges[i][1])] for i in range(edges.shape[0])])
                cycles_unnomralized[sample, heatmap_diffusion_num] = num_cycles_found_unnormalized
                cycles[sample, heatmap_diffusion_num] = num_cylces_found
                costs_unnormalized[sample, heatmap_diffusion_num] = cost_unnormalized
                costs[sample, heatmap_diffusion_num] = cost
                # print(f"Number of cycles found in diffusion num {heatmap_diffusion_num}: {num_cylces_found}, number of cycles in GT: {num_cylces_found_gt}")
            elif args.task_name == "mincut":
                kdt = KDTree(data_locations, leaf_size=30, metric='euclidean')
                dis_knn, idx_knn = kdt.query(data_locations, k=args.sparse_factor, return_distance=True)
                edge_index_0 = torch.arange(data_locations.shape[0]).reshape((-1, 1)).repeat(1, args.sparse_factor).reshape(-1)
                edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
                edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)
                np_edge_index = edge_index.numpy()

                data_heatmap = data_heatmap[0]
                np_points = data_locations
                distances = np.sum((np_points[np_edge_index[0]] - np_points[np_edge_index[1]]) ** 2, axis=1) ** .5
                best_solved_edges = find_mincut_approximation(adj_mat / distances, args.sparse_factor, np_points, np_edge_index)
                best_solved_cost = np.sum((np_points[best_solved_edges[:, 0]] - np_points[best_solved_edges[:, 1]]) ** 2) ** .5
                gt_edges = find_mincut(args.sparse_factor, np_points, distances, np_edge_index)
                gt_cost = np.sum(np.sum((np_points[gt_edges[:, 0]] - np_points[gt_edges[:, 1]]) ** 2, axis=1) ** .5)
            elif args.task_name == "djikstra":
                kdt = KDTree(data_locations, leaf_size=30, metric='euclidean')
                dis_knn, idx_knn = kdt.query(data_locations, k=args.sparse_factor, return_distance=True)
                edge_index_0 = torch.arange(data_locations.shape[0]).reshape((-1, 1)).repeat(1, args.sparse_factor).reshape(-1)
                edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
                edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)
                np_edge_index = edge_index.numpy()

                adj_mat = data_heatmap
                np_points = data_locations
                distances = np.sum((np_points[np_edge_index[0]] - np_points[np_edge_index[1]]) ** 2, axis=1) ** .5
                gt_edges, gt_num_searched = find_dijkstra(distances, args.sparse_factor, np_points, np_edge_index)
                
                gt_cost = np.sum(np.sum((np_points[gt_edges[:, 0]] - np_points[gt_edges[:, 1]]) ** 2, axis=1) ** .5)
                #  / distances
                best_solved_edges, num_searched = find_dijkstra_approximation(adj_mat, args.sparse_factor, np_points, np_edge_index)
                # print(f"gt_num_searched: {gt_num_searched}, num_searched: {num_searched}")            
                best_solved_cost = np.sum(np.sum((np_points[best_solved_edges[:, 0]] - np_points[best_solved_edges[:, 1]]) ** 2, axis=1) ** .5)
                best_solved_edges_normalized, num_searched_normalized = find_dijkstra_approximation(adj_mat / distances, args.sparse_factor, np_points, np_edge_index)
                # print(f"gt_num_searched: {gt_num_searched}, num_searched: {num_searched}")            
                best_solved_cost_normalized = np.sum(np.sum((np_points[best_solved_edges[:, 0]] - np_points[best_solved_edges[:, 1]]) ** 2, axis=1) ** .5)
                # print(f"gt_cost: {gt_cost}, best_solved_cost: {best_solved_cost}")

                gt_costs[sample] = gt_cost
                gt_cycles[sample] = gt_num_searched
                cycles_unnomralized[sample, heatmap_diffusion_num] = num_searched_normalized
                cycles[sample, heatmap_diffusion_num] = num_searched
                costs_unnormalized[sample, heatmap_diffusion_num] = best_solved_cost_normalized
                costs[sample, heatmap_diffusion_num] = best_solved_cost

    last_cycle_vals = cycles[:, -1]
    last_gt_cycle_vals = gt_cycles
    last_cycle_val_diff = last_gt_cycle_vals - last_cycle_vals
    gt_cycles, gt_std_cyles, lower_quartile_cylces_gt, upper_quartile_cycles_gt = np.median(gt_cycles), np.std(gt_cycles), np.percentile(gt_cycles, 25), np.percentile(gt_cycles, 75)
    # print(f"GT Cycles: {gt_cycles} +- {std_cyles}")
    gt_costs, gt_std_costs, lower_quartile_gt, upper_quartile_gt = np.median(gt_costs), np.std(gt_costs), np.percentile(gt_costs, 25), np.percentile(gt_costs, 75)
    cycles_unnomralized, std_cycles_unnomralized, lower_quartile_unnormalized, upper_quartile_unnormalized = np.median(cycles_unnomralized, axis=0), np.std(cycles_unnomralized, axis=0), np.percentile(cycles_unnomralized, 25, axis=0), np.percentile(cycles_unnomralized, 75, axis=0)
    cycles, std_cycles, lower_quartile, upper_quartile = np.median(cycles, axis=0), np.std(cycles, axis=0), np.percentile(cycles, 25, axis=0), np.percentile(cycles, 75, axis=0)
    costs_unnormalized, std_costs_unnormalized, lower_quartile_costs_unnormalized, upper_quartile_costs_unnormalized = np.median(costs_unnormalized, axis=0), np.std(costs_unnormalized, axis=0), np.percentile(costs_unnormalized, 25, axis=0), np.percentile(costs_unnormalized, 75, axis=0)
    costs, std_costs, lower_quartile_costs, upper_quartile_costs = np.median(costs, axis=0), np.std(costs, axis=0), np.percentile(costs, 25, axis=0), np.percentile(costs, 75, axis=0)
    info_dict = {
        "last_cycle_vals": list(last_cycle_val_diff),
        "gt_cycles": [gt_cycles.item(), gt_std_cyles.item(), lower_quartile_cylces_gt.item(), upper_quartile_cycles_gt.item()],
        "gt_costs": [gt_costs.item(), gt_std_costs.item(), lower_quartile_gt.item(), upper_quartile_gt.item()],
        "cycles_unnomralized": [list(cycles_unnomralized), list(std_cycles_unnomralized), list(lower_quartile_unnormalized), list(upper_quartile_unnormalized)],
        "cycles": [list(cycles), list(std_cycles), list(lower_quartile), list(upper_quartile)],
        "costs_unnormalized": [list(costs_unnormalized), list(std_costs_unnormalized), list(lower_quartile_costs_unnormalized), list(upper_quartile_costs_unnormalized)],
        "costs": [list(costs), list(std_costs), list(lower_quartile_costs), list(upper_quartile_costs)]
    }
    # save as json file
    with open(f"{args.save_path}/{val_or_test}-sample-{args.sample_num}-info.json", "w") as f:
        json.dump(info_dict, f)
    plt.plot(cycles_unnomralized, label="Cycles unnormalized")
    plt.plot(cycles, label="Cycles")
    # horizontal line with gt cycles:
    plt.axhline(y=gt_cycles, color='r', linestyle='-', label="GT Cycles")
    plt.legend()
    plt.savefig(f"{args.save_path}/{val_or_test}-sample-{args.sample_num}-cycles.png")
    plt.close()
    plt.plot(costs_unnormalized, label="Costs unnormalized")
    plt.plot(costs, label="Costs")
    # horizontal line with gt costs:
    plt.axhline(y=gt_costs, color='r', linestyle='-', label="GT Costs")
    plt.legend()
    plt.savefig(f"{args.save_path}/{val_or_test}-sample-{args.sample_num}-costs.png")
    plt.close()

# https://github.com/chaitjo/learning-tsp/blob/master/vizualizations_heatmaps.ipynb
def plot_tsp_heatmap(p, x_coord, W_pred, threshold=0.1, title="default"):
    """
    Helper function to plot predicted TSP tours with edge strength denoting confidence of prediction.
    
    Args:
        p: Matplotlib figure/subplot
        x_coord: Coordinates of nodes
        W_pred: Edge predictions matrix
        threshold: Threshold above which edge predicion probabilities are plotted
        title: Title of figure/subplot
    
    Returns:
        p: Updated figure/subplot
    """
    W_val = squareform(pdist(x_coord, metric='euclidean'))
    G = nx.from_numpy_matrix(W_val)
    
    pos = dict(zip(range(len(x_coord)), x_coord.tolist()))    

    edge_pairs = []
    edge_color = []
    for r in range(len(W_pred)):
        for c in range(len(W_pred)):
            if W_pred[r][c] >= threshold:
                if W_pred[r][c] == 100:
                    edge_color.append('r')
                elif W_pred[r][c] == 101:
                    edge_color.append('g')
                elif W_pred[r][c] > 101:
                    edge_color.append('purple')
                edge_pairs.append((r, c))
    
    nx.draw_networkx_nodes(G, pos, node_color='b', node_size=30)
    nx.draw_networkx_edges(G, pos, edgelist=edge_pairs, edge_color=edge_color, edge_cmap=plt.cm.Reds, width=1, alpha=1, edge_vmax=1, edge_vmin=0)
    p.set_title(title)
    return p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--val", action="store_true", default=True)
    parser.add_argument("--no-val", dest="val", action="store_false")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="./example")
    parser.add_argument("--task_name", type=str, default="mst")
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--sparse_factor", type=int, default=6)
    parser.add_argument("--graph_size", type=int, default=50)
    args = parser.parse_args()
    # for dijkstra final evaluation: models/tsp_diffusion/y25jtplk/numpy_heatmap
    # new dijkstra final evalution : models/tsp_diffusion/i99a9qei/numpy_heatmap
    # for mst final evaluation : models/tsp_diffusion/wdjbz37m/numpy_heatmap
    assert args.val or args.test and not (args.val and args.test), "Choose either val or test"
    # generate_gif(args)
    get_data(args)