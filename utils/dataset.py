from typing import Union, List, Tuple

import os
import pandas as pd
from tqdm import tqdm
from collections import Counter

from utils.utils import *
import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import dropout_edge
from torch_geometric.data import Data, Dataset

def get_feature_mask(rate, n_nodes, n_features):
    return torch.bernoulli(torch.Tensor([1 - rate/100]).repeat(n_nodes, n_features)).bool()

class GraphDataset(Dataset):
    def __init__(self, root, dataset_name, small=0, num_neighbors=2700,pro_T=20,knn_neigh=5,knn_metric='cosine',alpha=0.01, weakNodes=0, weakEdges=0, binary: bool = False, augmentation: bool = False,
                 val: bool = False, test: bool = False, transform=None, pre_transform=None):
        self.dataset_name = dataset_name
        self.small = small
        self.file_name = f'{dataset_name}{"-" if small>0 else ""}{small if small>0 else ""}-{"val" if val else "train"}{"-binary" if binary else ""}.csv'
        self.weakNodes = weakNodes
        self.weakEdges = weakEdges
        self.num_neighbors = num_neighbors
        self.binary = binary
        self.augmentation = augmentation
        self.df = None
        self.val = val
        self.test = test
        self.labels_encoder = []
        self.values_encoded = []
        self.pro_T = pro_T
        self.knn_neigh=knn_neigh
        self.knn_metric=knn_metric
        self.alpha = alpha
        super(GraphDataset, self).__init__(
            root,
            transform,
            pre_transform
        )

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """ If these files are found in processed_dire, processing is skipped. """
        if self.val:
            # file_path = f'{self.dataset_name}_val_{"binary_" if self.binary else ""}{self.num_neighbors}_{self.weakNodes}_{self.weakEdges}.npz'
            file_path = f'{self.small if self.small>0 else ""}_val_{"binary_" if self.binary else ""}{self.pro_T}_{self.alpha*10}_{self.weakNodes}_{self.weakEdges}.npz'
        else:
            if self.test:
                file_path = f'{self.small if self.small>0 else ""}_test_{"binary_" if self.binary else ""}{self.pro_T}_{self.alpha*10}_{self.weakNodes}_{self.weakEdges}.npz'
            else:
                file_path = f'{self.small if self.small>0 else ""}_{"binary_" if self.binary else ""}{self.pro_T}_{self.alpha*10}_{self.weakNodes}_{self.weakEdges}.npz'
        return [file_path]

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        """ If this file exist in raw_dir, the download is not triggered. """
        return self.file_name


    def compute_graph_diameter(self,graph: nx.Graph) -> int:
        if nx.is_connected(graph):
            return nx.diameter(graph)
        else:
            diameters = []
            for component in nx.connected_components(graph):
                subgraph = graph.subgraph(component) 
                if len(subgraph) > 1:
                    diameters.append(nx.diameter(subgraph))
            if diameters:
                return max(diameters) 
            else:
                return 0


    def process(self):
        #1. Read csv file
        self.df = pd.read_csv(self.raw_paths[0], header=0)
        print(self.df)

        # 2. Define the columns to be extracted from the data frame (attributes for graph nodes)
        extract_col = list(set(self.df.columns) - {'Timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'label'})
        # extract_col = list(set(self.df.columns) - {'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'label'})

        G = nx.Graph()
        iter_df = self.df[extract_col]
        y = self.df['label'].values

        print(Counter(y))

        for index, flow_entry in tqdm(iter_df.iterrows(), total=iter_df.shape[0], desc=f'Creating nodes...'):
            # Create attr for each label
            node_attr = {}
            for label, value in flow_entry.items():
                node_attr[label] = value
            node_attr['y'] = y[index]
            G.add_node(index, **node_attr)

        # Create edges
        if self.num_neighbors > 0:
            if self.dataset_name == 'DoHBrw':
                features_to_link = ['src_ip', 'src_port', 'dst_ip', 'dst_port']
            elif self.dataset_name == 'CICIDS':
                features_to_link = ['src_ip', 'src_port', 'dst_ip', 'dst_port','Protocol']
            else:
                features_to_link = ['src_ip', 'src_port', 'dst_ip', 'dst_port','proto']
            # features_to_link = ['src_ip', 'dst_ip']
            groups = self.df.groupby(features_to_link)
            max_edge = 0
            for group in tqdm(groups, total=len(groups), desc=f'Creating edges for features: {features_to_link}'):
                idx_matches = group[1].index
                if (len(idx_matches) > max_edge):
                    max_edge = len(idx_matches)
                if len(idx_matches) < 1:
                    continue
                for idx in range(len(idx_matches)):
                    a = idx_matches[idx]
                    for i in range(self.num_neighbors):
                        if idx + 1 + i < len(idx_matches):
                            b = idx_matches[idx + 1 + i]
                            # If edge (a, b) not exist create
                            if not G.has_edge(a, b):
                                G.add_edge(a, b)
        print("Max edge:", max_edge)
        try:
            diameter = self.compute_graph_diameter(G)
            print(f"Graph Diameter: {diameter}")
        except ValueError as e:
            print(f"Error calculating diameter: {e}")

        # Count and store the number of stary nodes
        stary_nodes_count = sum(1 for node in G.nodes if G.degree[node] == 0)
        print(f"Number of stary nodes: {stary_nodes_count}")


        # Create PyTorch Geometric data
        data = from_networkx(G, group_node_attrs=extract_col)

        # Weak features %
        if self.weakNodes > 0 and not self.val and not self.test:
            print(torch.sum(data.x))
            feature_mask = get_feature_mask(rate=self.weakNodes, n_nodes=data.num_nodes,
                                        n_features=data.num_features)
            data.x[~feature_mask] = 0.0
            print(torch.sum(data.x))
        # Weak structure %
        if self.weakEdges > 0 and not self.val and not self.test:
            print(f"A total of {data.num_edges} edges.")
            data.edge_index, _ = dropout_edge(data.edge_index, p=self.weakEdges/100, force_undirected=True)
            print(f"A total of {data.num_edges} edges.")



        if self.pro_T>0:
            x = data.x

            adj = edge_index_to_sparse_mx(data.edge_index, data.num_nodes)
            adj = process_adj(adj)
            # propagation
            x_prop = feature_propagation(adj, x, self.pro_T, self.alpha)
            adj_knn = get_knn_graph(x_prop, self.knn_neigh, knn_metric=self.knn_metric,batch_size=5000).tocoo()

            # structure
            adj_knn = process_adj(adj_knn)
            x_prop_aug = feature_propagation(adj_knn, x, self.pro_T, self.alpha)



        data.x_prop = torch.tensor(x_prop, dtype=torch.float)
        data.x_prop_aug = torch.tensor(x_prop_aug, dtype=torch.float)

        # Save data object
        if self.val:
            file_path = f'{self.small if self.small>0 else ""}_val_{"binary_" if self.binary else ""}{self.pro_T}_{self.alpha*10}_{self.weakNodes}_{self.weakEdges}.npz'
            torch.save(data, os.path.join(self.processed_dir, file_path))
        else:
            if self.test:
                file_path = f'{self.small if self.small>0 else ""}_test_{"binary_" if self.binary else ""}{self.pro_T}_{self.alpha*10}_{self.weakNodes}_{self.weakEdges}.npz'
                torch.save(data, os.path.join(self.processed_dir, file_path))
            else:
                file_path = f'{self.small if self.small>0 else ""}_{"binary_" if self.binary else ""}' \
                            f'{self.pro_T}_{self.alpha*10}_{self.weakNodes}_{self.weakEdges}.npz'
                torch.save(data, os.path.join(self.processed_dir, file_path))

    def len(self) -> int:
        """ Return number of graph """
        return 1



    def get(self, idx: int) -> Data:
        """ Return the idx-th graph. """
        if self.val:
            file_path = f'{self.small if self.small>0 else ""}_val_{"binary_" if self.binary else ""}{self.pro_T}_{self.alpha*10}_{self.weakNodes}_{self.weakEdges}.npz'
        else:
            if self.test:
                file_path = f'{self.small if self.small>0 else ""}_test_{"binary_" if self.binary else ""}{self.pro_T}_{self.alpha*10}_{self.weakNodes}_{self.weakEdges}.npz'
            else:
                file_path = f'{self.small if self.small>0 else ""}_{"binary_" if self.binary else ""}' \
                            f'{self.pro_T}_{self.alpha*10}_{self.weakNodes}_{self.weakEdges}.npz'

        data = torch.load(os.path.join(self.processed_dir, file_path), weights_only=False)
        return data
