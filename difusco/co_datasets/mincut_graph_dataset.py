"""TSP (Traveling Salesman Problem) Graph Dataset"""

import numpy as np
import torch

from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData


class MinCutGraphDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, sparse_factor=-1):
    self.data_file = data_file
    self.sparse_factor = sparse_factor
    self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    # Select sample
    line = self.file_lines[idx]
    # Clear leading/trailing characters
    line = line.strip()

    # Extract points
    points = line.split(' output ')[0]
    points = points.split(' ')
    points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
    # Extract tour
    MinCut = line.split(' output ')[1]
    MinCut = MinCut.split(' ')
    MinCut = np.array([[int(MinCut[i]), int(MinCut[i + 1])] for i in range(0, len(MinCut), 2)])

    return points, MinCut

  def __getitem__(self, idx):
    points, MinCut = self.get_example(idx)
    if self.sparse_factor <= 0:
      # Return a densely connected graph
      adj_matrix = np.zeros((points.shape[0], points.shape[0]))
      for i in range(tour.shape[0] - 1):
        adj_matrix[tour[i], tour[i + 1]] = 1
      # return points, adj_matrix, tour

      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          torch.from_numpy(points).float(),
          torch.from_numpy(adj_matrix).float(),
          torch.from_numpy(tour).long(),
      )
    else:
    #   print(f"index: {idx}")
      # Return a sparse graph where each node is connected to its k nearest neighbors
      # k = self.sparse_factor
      sparse_factor = self.sparse_factor
      kdt = KDTree(points, leaf_size=30, metric='euclidean')
      dis_knn, idx_knn = kdt.query(points, k=sparse_factor, return_distance=True)

      edge_index_0 = torch.arange(points.shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))

      edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

      MinCut_edges = torch.from_numpy(MinCut)
      MinCut_edges_empty = torch.zeros(idx_knn.shape)
      MinCut_edges_empty -= 1
      for i in range(MinCut_edges.shape[0]):
        found = False
        for j in range(MinCut_edges_empty.shape[1]):
            if MinCut_edges[i][1] == idx_knn[MinCut_edges[i][0]][j]:
                MinCut_edges_empty[MinCut_edges[i][0]][j] = MinCut_edges[i][1]
                found = True
            elif MinCut_edges[i][0] == idx_knn[MinCut_edges[i][1]][j]:
                MinCut_edges_empty[MinCut_edges[i][1]][j] = MinCut_edges[i][0]
                found = True
        if not found:
            print("Error: Edge not found in KNN graph")
            print(MinCut_edges[i])
            print(idx_knn[MinCut_edges[i][0]])
            print(MinCut_edges_empty[MinCut_edges[i][0]])
            print(idx_knn[MinCut_edges[i][1]])
            exit(1)
    #   print(MinCut_edges_empty)
    #   print(edge_index_1)
      MinCut_edges = MinCut_edges_empty.reshape(-1)
    #   MinCut_edges = MinCut_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      MinCut_edges = torch.eq(edge_index_1, MinCut_edges).reshape(-1, 1)
      graph_data = GraphData(x=torch.from_numpy(points).float(),
                             edge_index=edge_index,
                             edge_attr=MinCut_edges)

      point_indicator = np.array([points.shape[0]], dtype=np.int64)
      edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)
      # pad mincut so it is size (n, 2):
      MinCut = np.pad(MinCut, ((0, points.shape[0] - MinCut.shape[0]), (0, 0)), 'constant', constant_values=-1)
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          graph_data,
          torch.from_numpy(point_indicator).long(),
          torch.from_numpy(edge_indicator).long(),
          torch.from_numpy(MinCut).long(),
      )
