
import random
import time
from copy import deepcopy
import numpy as np
import scipy.sparse as sp
import argparse
import sys
import torch
import os
import time
import module
import yaml
import torch.nn.functional as F
from utils import load_data, load_fixed_data_split
from utils import set_device, seed_everything
from utils import setup_cfg, tab_printer
from sklearn.metrics import accuracy_score as ACC
from typing import Dict
from collections import defaultdict


def print_res(res_list, info, matrix=False):
    if matrix:
        k = len(res_list)
        res = res_list[0]
        for i in range(k-1):
            res = res + res_list[i+1]
        print(info+':')
        print(res / k)
        return res / k
    else: 
        mean = np.mean(res_list)
        std = np.std(res_list) 
        print(f'{info} mean ± std: {mean*100} ± {std*100}')
        return mean * 100

def filter_confident_predictions(pse, entropy_threshold=0.5, prob_threshold=0.9):
    eps = 1e-12  
    entropy = -torch.sum(pse * torch.log2(pse + eps), dim=1)
    max_prob = torch.max(pse, dim=1).values
    confident_mask = (entropy < entropy_threshold) & (max_prob > prob_threshold)
    return entropy, max_prob, confident_mask

def kl_divergence(p, q, epsilon=1e-8):
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    return torch.sum(p * torch.log(p / q))

def main():
    device = set_device(args.device)
    graph, label, class_num, shared_nodes_mask, removed_nodes_mask = load_data(
        dataset_name=args.dataset,
        normalize=args.normalize,
        undirected=args.undirected,
        self_loop=args.self_loop,
    )

    with open(args.config, 'r', encoding='utf-8') as f:  
        config = yaml.safe_load(f)

    dataset_config = config[args.dataset]
    args.nnodes = graph.num_nodes()


    t_start = time.time()
    seed_everything(args.seed)
    seed_list = [random.randint(0, 99999) for i in range(args.runs * 100)]
    all_mask = torch.BoolTensor(torch.ones(graph.num_nodes(), dtype=bool))

    threshold_map_1 = {
        'chameleonf': (0.4, 0.8), 
        'squirrelf': (0.4, 0.8), 
        'romanempire': (0.0005, 0.99),
        'amazonratings': (0.0005, 0.99),
        'flickr': (0.1, 0.98),
        'photo': (0.0005, 0.99), 
        'wikics': (0.0005, 0.99),
        'pubmed': (0.1, 0.95),
    }
    threshold_map_2 = {
        'chameleonf': (0.4, 0.8), 
        'squirrelf': (0.4, 0.8), 
        'romanempire': (0.0005, 0.99), 
        'amazonratings': (0.0005, 0.99),
        'flickr': (0.1, 0.98),
        'photo': (0.1, 0.95), 
        'wikics': (0.0005, 0.99),
        'pubmed': (0.1, 0.95),
    }
    threshold_map_3 = {
        'chameleonf': (0.4, 0.8), 
        'squirrelf': (0.4, 0.8), 
        'romanempire': (0.0005, 0.99),
        'amazonratings': (0.0005, 0.99),
        'flickr': (0.1, 0.98),
        'photo': (0.1, 0.95), 
        'wikics': (0.0005, 0.99),
        'pubmed': (0.1, 0.95),
    }
    entropy_th_1, prob_th_1 = threshold_map_1.get(args.dataset)
    entropy_th_2, prob_th_2 = threshold_map_2.get(args.dataset)
    entropy_th_3, prob_th_3 = threshold_map_3.get(args.dataset)
    tao_1 = 0.8 #0.85 for squirrelf,0.6 for flickr,0.8 for others
    tao_2 = 0.8 
    tao_3 = 0.8

    res_list_acc = []
    share_list_acc = []
    remove_list_acc = []

    for run in range(args.runs):
        print(f"\nRun: {run}\n") 
        seed_everything(seed_list[run])    
        train_mask, val_mask, test_mask = load_fixed_data_split(args.dataset, run)

        Model = getattr(module, 'CNDGNN')
        model = Model(
            in_features=graph.ndata['feat'].shape[1],
            num_nodes=graph.ndata['feat'].shape[0],
            class_num=class_num,
            device=device,
            args=args,
        )

        model.pretrain(
            graph,
            label,
            train_mask,
            val_mask,
            test_mask,
        )  
        res_1, C_1, Z_1 = model.predict_pretrain(graph)
        acc = ACC(label[test_mask], res_1[test_mask]) 
        print(f"Pretrain Acc: {acc}")    

        C_1 = torch.exp(C_1)  

        mixed_labels = torch.where(train_mask, label, res_1)

        X = F.one_hot(mixed_labels).float()
        N, _ = X.shape
        graph = graph.remove_self_loop()
        adj = graph.adj().to_dense()
        degrees = graph.out_degrees().float()

        pse = torch.where(train_mask.unsqueeze(1), X, C_1)
        entropy, max_prob, confident_mask = filter_confident_predictions(pse, entropy_threshold=entropy_th_1, prob_threshold=prob_th_1)
        confident_mask_binary = confident_mask.int().float()

        neighbor_pattern = torch.matmul(adj, X)
        neighbor_pattern_two = torch.matmul(adj, neighbor_pattern)
        neighbor_entropy = torch.matmul(adj, confident_mask_binary)
        neighbor_entropy = neighbor_entropy / degrees

        row_sums = neighbor_pattern.sum(dim=1, keepdim=True)
        normalized_pattern = neighbor_pattern / (row_sums + 1e-8)
        row_sums_two = neighbor_pattern_two.sum(dim=1, keepdim=True)
        normalized_pattern_two = neighbor_pattern_two / (row_sums_two + 1e-8)

        unique_patterns, inverse_indices, counts = torch.unique(
            normalized_pattern, dim=0, return_inverse=True, return_counts=True)

        shared_mask = counts > 1
        shared_indices = torch.where(shared_mask)[0]  

        A = torch.zeros((N, N), dtype=torch.int8, device=X.device)  
        for idx in shared_indices:
            nodes = (inverse_indices == idx).nonzero(as_tuple=True)[0]
            nodes = nodes[neighbor_entropy[nodes] > tao_1]
            if len(nodes) >= 2:
                pairs = torch.combinations(nodes, 2)
                A[pairs[:, 0], pairs[:, 1]] = 1
                A[pairs[:, 1], pairs[:, 0]] = 1  

        first_fit_params = dataset_config['first_fit'] 

        model.fit(
            graph,
            label,
            mixed_labels,
            train_mask,
            val_mask,
            test_mask,
            normalized_pattern_two, A,
            **first_fit_params
        )

        res, C, Z = model.predict(graph, normalized_pattern_two, A)

        mixed_labels = torch.where(train_mask, label, res)

        X = F.one_hot(mixed_labels).float()
        N, _ = X.shape
        graph = graph.remove_self_loop()
        adj = graph.adj().to_dense()
        degrees = graph.out_degrees().float()

        pse = torch.where(train_mask.unsqueeze(1), X, C_1)
        entropy, max_prob, confident_mask = filter_confident_predictions(pse, entropy_threshold=entropy_th_2, prob_threshold=prob_th_2)
        confident_mask_binary = confident_mask.int().float()

        neighbor_pattern = torch.matmul(adj, X)
        neighbor_pattern_two = torch.matmul(adj, neighbor_pattern)
        neighbor_entropy = torch.matmul(adj, confident_mask_binary)
        neighbor_entropy = neighbor_entropy / degrees

        row_sums = neighbor_pattern.sum(dim=1, keepdim=True)
        normalized_pattern = neighbor_pattern / (row_sums + 1e-8)
        row_sums_two = neighbor_pattern_two.sum(dim=1, keepdim=True)
        normalized_pattern_two = neighbor_pattern_two / (row_sums_two + 1e-8)

        unique_patterns, inverse_indices, counts = torch.unique(
            normalized_pattern, dim=0, return_inverse=True, return_counts=True)

        shared_mask = counts > 1
        shared_indices = torch.where(shared_mask)[0] 

        A = torch.zeros((N, N), dtype=torch.int8, device=X.device)  
        for idx in shared_indices:
            nodes = (inverse_indices == idx).nonzero(as_tuple=True)[0]
            nodes = nodes[neighbor_entropy[nodes] > tao_2]
            if len(nodes) >= 2:
                pairs = torch.combinations(nodes, 2)
                A[pairs[:, 0], pairs[:, 1]] = 1
                A[pairs[:, 1], pairs[:, 0]] = 1  

        second_fit_params = dataset_config['second_fit']

        model.fit(
            graph,
            label,
            mixed_labels,
            train_mask,
            val_mask,
            test_mask,
            normalized_pattern_two, A,
            **second_fit_params
        )

        res, C, Z = model.predict(graph, normalized_pattern_two, A)

        """
        #for flickr and photo dataset
        mixed_labels = torch.where(train_mask, label, res)

        X = F.one_hot(mixed_labels).float()
        N, _ = X.shape
        graph = graph.remove_self_loop()
        adj = graph.adj().to_dense()
        degrees = graph.out_degrees().float()

        pse = torch.where(train_mask.unsqueeze(1), X, C_1)
        entropy, max_prob, confident_mask = filter_confident_predictions(pse, entropy_threshold=entropy_th_3, prob_threshold=prob_th_3)
        confident_mask_binary = confident_mask.int().float()

        neighbor_pattern = torch.matmul(adj, X)
        neighbor_pattern_two = torch.matmul(adj, neighbor_pattern)
        neighbor_entropy = torch.matmul(adj, confident_mask_binary)
        neighbor_entropy = neighbor_entropy / degrees

        row_sums = neighbor_pattern.sum(dim=1, keepdim=True)
        normalized_pattern = neighbor_pattern / (row_sums + 1e-8)
        row_sums_two = neighbor_pattern_two.sum(dim=1, keepdim=True)
        normalized_pattern_two = neighbor_pattern_two / (row_sums_two + 1e-8)

        unique_patterns, inverse_indices, counts = torch.unique(
            normalized_pattern, dim=0, return_inverse=True, return_counts=True)

        shared_mask = counts > 1
        shared_indices = torch.where(shared_mask)[0] 

        A = torch.zeros((N, N), dtype=torch.int8, device=X.device)  
        for idx in shared_indices:
            nodes = (inverse_indices == idx).nonzero(as_tuple=True)[0]
            nodes = nodes[neighbor_entropy[nodes] > tao_3]
            if len(nodes) >= 2:
                pairs = torch.combinations(nodes, 2)
                A[pairs[:, 0], pairs[:, 1]] = 1
                A[pairs[:, 1], pairs[:, 0]] = 1  

        third_fit_params = dataset_config['third_fit']

        model.fit(
            graph,
            label,
            mixed_labels,
            train_mask,
            val_mask,
            test_mask,
            normalized_pattern_two, A,
            **third_fit_params
        )
        res, C, Z = model.predict(graph, normalized_pattern_two, A)
        """

        acc = ACC(label[test_mask], res[test_mask])
        res_list_acc.append(acc)
        print(f"Final test Acc: {acc}")


        shared_nodes_mask_test = shared_nodes_mask[test_mask]
        if shared_nodes_mask_test.sum().item() > 0:
            shared_nodes_pred = res[test_mask][shared_nodes_mask_test]
            shared_nodes_true = label[test_mask][shared_nodes_mask_test]
            shared_nodes_acc = (shared_nodes_pred == shared_nodes_true).sum().item() / shared_nodes_mask_test.sum().item()
        else:
            shared_nodes_acc = 0.0
        print(f"Shared nodes accuracy: {shared_nodes_acc}")
        share_list_acc.append(shared_nodes_acc)


        removed_nodes_mask_test = removed_nodes_mask[test_mask]
        if removed_nodes_mask_test.sum().item() > 0:
            removed_nodes_pred = res[test_mask][removed_nodes_mask_test]
            removed_nodes_true = label[test_mask][removed_nodes_mask_test]
            removed_nodes_acc = (removed_nodes_pred == removed_nodes_true).sum().item() / removed_nodes_mask_test.sum().item()
        else:
            removed_nodes_acc = 0.0
        print(f"Removed nodes accuracy: {removed_nodes_acc}")
        remove_list_acc.append(removed_nodes_acc)

    t_finish = time.time()
    print("\nTrain cost: {:.4f}s".format(t_finish - t_start))
    print("\nResults:")
    acc_avg = print_res(res_list_acc, 'ACC')
    scc = print_res(share_list_acc, 'scc')
    rcc = print_res(remove_list_acc, 'rcc')

def set_args(args, model_type):
    args.model_type = model_type
    args.self_loop = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='All', description='Parameters for Baseline Method')
    # experiment parameter
    parser.add_argument('--seed', type=int, default=42, help='Random seed. Defaults to 4096.') 
    parser.add_argument('--config', type=str, default='./config/xxxx.yaml', help='path of config file')
    parser.add_argument('--device', type=str, default='0', help='GPU id') 
    parser.add_argument('--runs', type=int, default=10, help='The number of runs of task with same parmeter')
    parser.add_argument('--dataset', type=str, default='romanempire', help='Dataset used in the experiment')

    # train parameter 
    parser.add_argument('--epochs', type=int, default=2500, help='num of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate') 
    parser.add_argument('--l2_coef', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=200, help='early stop patience')

    # model parameter
    parser.add_argument('--nhidden', type=int, default=128, help='num of hidden dimension')
    parser.add_argument('--undirected', type=bool, default=True, help='change graph to undirected')
    parser.add_argument('--self_loop', type=bool, default=True, help='add self loop')
    parser.add_argument('--normalize', type=int, default=-1, help='feature norm, -1 for without normalize')
    parser.add_argument('--layers', type=int, default=3, help='layers of model')
    parser.add_argument('--model_type', type=str, default='CNDGNN', help='')

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--variant', type=bool, default=False)
    parser.add_argument('--structure_info', type=bool, default=False)
    parser.add_argument('--lambda_', type=float, default=0.0)

    args = parser.parse_args()
    args_dict = args.__dict__

    model_type = args_dict['model_type']
    args_dict['config'] = f'./config/CNDGNN.yaml'
    args = setup_cfg(args, args_dict)
    set_args(args, model_type)

    tab_printer(args_dict)
    main()


