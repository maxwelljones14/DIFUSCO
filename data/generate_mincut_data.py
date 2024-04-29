import argparse
import pprint as pp
import time
import warnings
import networkx as nx
from multiprocessing import Pool
from sklearn.neighbors import NearestNeighbors
import numpy as np
import tqdm
from sklearn.neighbors import KDTree

warnings.filterwarnings("ignore")

def generate_knn_graph(nodes_coord, k):
    num_nodes = len(nodes_coord)
    while True:
        G = nx.Graph()

        # Add nodes to the graph
        for i in range(num_nodes):
            G.add_node(i)

        # Create a nearest neighbors model
        kdt = KDTree(nodes_coord, leaf_size=30, metric='euclidean')
        distances, indices = kdt.query(nodes_coord, k=k, return_distance=True)

        # Add edges based on k closest neighbors
        for i in range(num_nodes):
            for j in indices[i]:
                if i != j:  # Avoid self-loops
                    dist = distances[i][list(indices[i]).index(j)]
                    G.add_edge(i, j, weight=dist)
        
        if nx.is_connected(G):
            return G, nodes_coord
        else:
            print("Generated Graph is not connected, generating new coords and trying again")
            nodes_coord = np.random.random([num_nodes, 2])

def generate_cut(args):
    nodes_coord, k = args
    num_nodes = len(nodes_coord)
    G, nodes_coord = generate_knn_graph(nodes_coord, k)

    # Find the minimum cut using Stoer-Wagner algorithm
    cut_value, partition = nx.stoer_wagner(G)

    # Generate adjacency list from the partition
    partition_0, partition_1 = partition
    adjacency_list = [[] for _ in range(num_nodes)]

    for i in partition_0:
        for j in partition_1:
            if G.has_edge(i, j):
                adjacency_list[i].append(j)

    return nodes_coord, adjacency_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_nodes", type=int, default=50)
    parser.add_argument("--max_nodes", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--k", type=int, default=5)
    opts = parser.parse_args()

    assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"

    np.random.seed(opts.seed)

    if opts.filename is None:
        opts.filename = f"mincut{opts.min_nodes}-{opts.max_nodes}.txt"

    # Pretty print the run args
    pp.pprint(vars(opts))

    with open(opts.filename, "w") as f:
        start_time = time.time()
        for b_idx in tqdm.tqdm(range(opts.num_samples // opts.batch_size)):
            num_nodes = np.random.randint(low=opts.min_nodes, high=opts.max_nodes + 1)
            assert opts.min_nodes <= num_nodes <= opts.max_nodes

            batch_nodes_coord = np.random.random([opts.batch_size, num_nodes, 2])

            with Pool(opts.batch_size) as p:
                nodes_and_cuts = p.map(generate_cut, [(batch_nodes_coord[idx], opts.k) for idx in range(opts.batch_size)])

            for nodes_coord_list, cut in nodes_and_cuts:
                f.write(" ".join(str(coord) for node_coord in nodes_coord_list for coord in node_coord))
                f.write(" output ")
                for i, neighbors in enumerate(cut):
                    for neighbor in neighbors:
                        f.write(f"{i} {neighbor} ")
                f.write("\n")

        end_time = time.time() - start_time

        assert b_idx == opts.num_samples // opts.batch_size - 1

    print(f"Completed generation of {opts.num_samples} samples of Min Cut {opts.min_nodes}-{opts.max_nodes}.")
    print(f"Total time: {end_time / 60:.1f}m")
    print(f"Average time: {end_time / opts.num_samples:.1f}s")