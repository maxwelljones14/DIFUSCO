import argparse
import pprint as pp
import time
import warnings
from multiprocessing import Pool

import numpy as np
import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

warnings.filterwarnings("ignore")


def generate_mst(nodes_coord):
    # Calculate pairwise distances between nodes
    num_nodes = len(nodes_coord)
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist_matrix[i, j] = np.linalg.norm(nodes_coord[i] - nodes_coord[j])
            dist_matrix[j, i] = dist_matrix[i, j]

    # Convert distance matrix to CSR format
    dist_matrix_sparse = csr_matrix(dist_matrix)

    # Compute Minimum Spanning Tree
    mst_sparse = minimum_spanning_tree(dist_matrix_sparse)

    # Convert MST to adjacency list
    adjacency_list = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if mst_sparse[i, j] > 0:
                adjacency_list[i].append(j)

    return adjacency_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_nodes", type=int, default=50)
    parser.add_argument("--max_nodes", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    opts = parser.parse_args()

    assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"

    np.random.seed(opts.seed)

    if opts.filename is None:
        opts.filename = f"mst{opts.min_nodes}-{opts.max_nodes}.txt"

    # Pretty print the run args
    pp.pprint(vars(opts))

    with open(opts.filename, "w") as f:
        start_time = time.time()
        for b_idx in tqdm.tqdm(range(opts.num_samples // opts.batch_size)):
            num_nodes = np.random.randint(low=opts.min_nodes, high=opts.max_nodes + 1)
            assert opts.min_nodes <= num_nodes <= opts.max_nodes

            batch_nodes_coord = np.random.random([opts.batch_size, num_nodes, 2])

            with Pool(opts.batch_size) as p:
                msts = p.map(generate_mst, [batch_nodes_coord[idx] for idx in range(opts.batch_size)])

            for idx, mst in enumerate(msts):
                # Fixing the issue here
                f.write(" ".join(str(coord) for node_coord in batch_nodes_coord[idx] for coord in node_coord))
                f.write(" output ")
                for i, neighbors in enumerate(mst):
                    for neighbor in neighbors:
                        if neighbor > i:
                            f.write(f"{i} {neighbor} ")
                f.write("\n")

        end_time = time.time() - start_time

        assert b_idx == opts.num_samples // opts.batch_size - 1

    print(f"Completed generation of {opts.num_samples} samples of MST{opts.min_nodes}-{opts.max_nodes}.")
    print(f"Total time: {end_time / 60:.1f}m")
    print(f"Average time: {end_time / opts.num_samples:.1f}s")