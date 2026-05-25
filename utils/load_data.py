from typing import Tuple
import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from dgl.data.utils import save_graphs, load_graphs
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import Tuple, Dict

table_datasets = [
    'chameleonf', 'squirrelf', 
    'romanempire', 'amazonratings', 
    'flickr', 
    'photo',  'wikics', 'pubmed', 
] 

def analyze_label_distribution(matrix_dict: Dict[int, np.ndarray], label_dict: Dict[int, torch.Tensor], num_nodes: int):

    neigh_dist_dict = defaultdict(list)

    shared_nodes_mask = torch.zeros(num_nodes, dtype=torch.bool)  
    removed_nodes_mask = torch.zeros(num_nodes, dtype=torch.bool) 

    for label_class, matrix in matrix_dict.items():
        nodes_in_class = label_dict[label_class]
        

        for i, node in enumerate(nodes_in_class):
            neigh_dist_tuple = tuple(matrix[i])  
            neigh_dist_dict[neigh_dist_tuple].append((label_class, node))  


    for neigh_dist, nodes in neigh_dist_dict.items():
        if len(nodes) > 1:  

            class_ids = set([node[0] for node in nodes])  
            if len(class_ids) > 1:  
                #print(f"Neighbor distribution {neigh_dist} is shared by nodes from different classes:")
                for label_class, node in nodes:
                    #print(f"  - Node {node} from class {label_class}")
                    shared_nodes_mask[node] = True  

                class_counts = defaultdict(int)
                for label_class, _ in nodes:
                    class_counts[label_class] += 1


                max_class = max(class_counts, key=class_counts.get)

                for label_class, node in nodes:
                    if label_class != max_class:  
                        removed_nodes_mask[node] = True

    print(f"\nMask for nodes with shared neighbor label distribution: {shared_nodes_mask}")
    print(f"Mask for nodes remaining after removing the largest class: {removed_nodes_mask}")

    return shared_nodes_mask, removed_nodes_mask  


def load_data(
    dataset_name: str,
    normalize: int = -1,
    undirected: bool=True,
    self_loop: bool=True,

) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    dataset_name = dataset_name.lower()
    if dataset_name in table_datasets:
        print("Load Dataset: ", dataset_name)
        file_path = f'data/graphs/{dataset_name}.pt'
        graphs, _ = load_graphs(file_path)
        graph = graphs[0]
        label = graph.ndata['label']
        class_num = (torch.max(label) + 1).long().item()
        if normalize != -1:
            graph.ndata['feat'] = F.normalize(graph.ndata['feat'], dim=1, p=normalize)
        if undirected:
            graph = dgl.to_bidirected(graph, copy_ndata=True)
        if self_loop:
            graph = graph.remove_self_loop().add_self_loop()

        adj = graph.adj().to_dense()
        print(adj.size())
        print(label.size())

        label_neigh_matrix = {}
        label_dict = {}

        for label_class in range(class_num):
            nodes_in_class = (label == label_class).nonzero(as_tuple=True)[0]
            N = nodes_in_class.shape[0]  

            matrix = np.zeros((N, class_num), dtype=float)

            for i, node in enumerate(nodes_in_class):
                if dataset_name in ['romanempire', 'amazonratings','chameleonf', 'squirrelf']:
                    node = node.to(torch.int32)
                else:
                    node = node.to(torch.int64)

                neighbors = graph.successors(node)  
                neighbor_labels = label[neighbors]  

                for neighbor_label in neighbor_labels:
                    matrix[i, neighbor_label.item()] += 1
            """

            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
            """

            label_neigh_matrix[label_class] = matrix
            label_dict[label_class] = nodes_in_class
        """
        print("Normalized Neighbor Label Distribution Matrix for Each Label Class:\n")


        for label_class, matrix in label_neigh_matrix.items():
            print(f"Label {label_class} Neighbor Matrix:\n", matrix)
        """

        shared_nodes_mask, removed_nodes_mask = analyze_label_distribution(label_neigh_matrix, label_dict, num_nodes=graph.num_nodes())
                
    else:
        raise NotImplementedError
    return graph, label, class_num,shared_nodes_mask, removed_nodes_mask


def load_fixed_data_split(dataname, split_idx):
    """load fixed split for benchmark dataset, train/val/test is 48%/32%/20%.
    Parameters
    ----------
    dataname: str
        dataset name.
    split_idx: int
        id of split plan.
    """
    splits_file_path = f'./data/splits/{dataname}_splits.pt'
    splits_file = torch.load(splits_file_path)
    train_mask_list = splits_file['train']
    val_mask_list = splits_file['val']
    test_mask_list = splits_file['test']
    
    train_mask = torch.BoolTensor(train_mask_list[split_idx])
    val_mask = torch.BoolTensor(val_mask_list[split_idx])
    test_mask = torch.BoolTensor(test_mask_list[split_idx])
    return train_mask, val_mask, test_mask

