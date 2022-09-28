#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import torch
import numpy as np
from collections import Counter
from torch_geometric.data import Data

name_dataset = 'Mutagenicity'
dir_data = './datasets/Mutagenicity/'

# filenames
file_edge_list = f'{dir_data}{name_dataset}_A.txt'
file_edge_gt = f'{dir_data}{name_dataset}_edge_gt.txt'
file_graph_indicator = f'{dir_data}{name_dataset}_graph_indicator.txt'
file_graph_labels = f'{dir_data}{name_dataset}_graph_labels.txt'
file_node_labels = f'{dir_data}{name_dataset}_node_labels.txt'

# load data
edge_list = np.loadtxt(file_edge_list, delimiter=',').astype(np.int32)
edge_gt = np.loadtxt(file_edge_gt, delimiter=',').astype(np.int32)
graph_indicator = np.loadtxt(file_graph_indicator,delimiter=',').astype(np.int32)
graph_labels = np.loadtxt(file_graph_labels,delimiter=',').astype(np.int32)
node_labels = np.loadtxt(file_node_labels,delimiter=',').astype(np.int32)
num_features = 14

# create useful data structures
graph_indicator_dict = {i+1 : graph_indicator[i] for i in range(len(graph_indicator))}
num_nodes_dict = Counter(graph_indicator_dict.values())
num_nodes_idx = np.cumsum([*num_nodes_dict.values()])

dataset = []
graph_id = 1
count = 0
start_pointer = 0
end_pointer = 0
start_idx = 0
for edge in edge_list:
    src, dst = edge
    
    if graph_indicator_dict[src] == graph_id:
        count += 1
    else:
        end_pointer += count
        edge_list_graph = edge_list[start_pointer:end_pointer]
        edge_gt_graph = torch.tensor(edge_gt[start_pointer:end_pointer])
        graph_label = graph_labels[graph_id-1]
        node_labels_graph = node_labels[start_idx:num_nodes_idx[graph_id-1]]
        node_attr = np.zeros((num_nodes_dict[graph_id], num_features))
        for i, l in enumerate(node_labels_graph):
            node_attr[i, l] = 1.0
        start_idx = num_nodes_idx[graph_id-1]
        start_pointer += count
        count = 1
        graph_id += 1
        
        edge_index_tmp = torch.tensor(edge_list_graph.T).long()
        src, dst, mask = zip(*sorted(zip(edge_index_tmp[0], edge_index_tmp[1], edge_gt_graph), key=lambda tup: (tup[0],tup[1])))
        edge_index_tmp = torch.tensor([src, dst])
        edge_gt_graph = torch.tensor(mask)
        
        new_g = Data(x=torch.tensor(node_attr).float(),
                     edge_index=edge_index_tmp - min(edge_index_tmp[0]),
                     y=torch.tensor([graph_label]).long(),
                     ground_truth=edge_gt_graph)
    
        dataset.append(new_g)
        
    if graph_id == len(num_nodes_dict):
        edge_list_graph = edge_list[start_pointer:]
        edge_gt_graph = torch.tensor(edge_gt[start_pointer:])
        graph_label = graph_labels[graph_id-1]
        node_labels_graph = node_labels[start_idx:num_nodes_idx[graph_id-1]]
        node_attr = np.zeros((num_nodes_dict[graph_id], num_features))
        for i, l in enumerate(node_labels_graph):
            node_attr[i, l] = 1.0
        
        edge_index_tmp = torch.tensor(edge_list_graph.T).long()
        src, dst, mask = zip(*sorted(zip(edge_index_tmp[0], edge_index_tmp[1], edge_gt_graph), key=lambda tup: (tup[0],tup[1])))
        edge_index_tmp = torch.tensor([src, dst])
        edge_gt_graph = torch.tensor(mask)
        
        new_g = Data(x=torch.tensor(node_attr).float(),
                     edge_index=edge_index_tmp - min(edge_index_tmp[0]),
                     y=torch.tensor([graph_label]).long(),
                     ground_truth=edge_gt_graph)
    
        dataset.append(new_g)
        break

out_filename = dir_data + f'{name_dataset}.pkl'
with open(out_filename, 'wb') as outfile:
    pickle.dump(dataset, outfile)
    outfile.close()   

