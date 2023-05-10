#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import torch
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from custom.utils import parse_boolean

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['ba_2motifs', 'Mutagenicity'],
                    help='Name of the dataset')
parser.add_argument('--model', type=str,
                    choices=['L2XGIN', 'L2XGCN', 'L2XGSG'], default='L2XGIN',
                    help='Name of the model')
parser.add_argument('--connected', type=parse_boolean, default=True, 
                    help='Get connected output or not')
parser.add_argument('--ratio', type=float,
                    help='Ratio of restrained edges')
parser.add_argument('--split', type=int, default=0,
                    help='Fold to evaluate {0-4}')

                    
args = parser.parse_args()
name_dataset = args.dataset
name_model = args.model
connected_flag = args.connected
ratio = args.ratio
split = args.split

ratio_str = str(ratio).replace('.', '')
if connected_flag:
    connected_str = 'connected'
else:
    connected_str = 'disconnected'

dir_results = f'./results/{name_dataset}/{connected_str}_{name_model}/{ratio_str}/s{split}/' 
dir_plot = f'./explanation_plots/{name_dataset}/{connected_str}_{name_model}/{ratio_str}/s{split}/' 
if not os.path.exists(dir_plot):
    os.makedirs(dir_plot)

edge_index_list = pickle.load(open(dir_results + 'edge_index_list.pkl', 'rb'))
test_edge_mask = pickle.load(open(dir_results + 'edge_mask.pkl', 'rb'))
test_sections = pickle.load(open(dir_results + 'test_sections.pkl', 'rb'))
x_attr = pickle.load(open(dir_results + 'x_attributes.pkl', 'rb'))
y_pred = pickle.load(open(dir_results + 'y_pred_test.pkl', 'rb'))
y_true = pickle.load(open(dir_results + 'y_true_test.pkl', 'rb'))


if name_dataset == 'Mutagenicity':
    mutag_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
                      8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
    colors_dict = {0: 'cornflowerblue', 1:'tomato', 2:'gold', 3:'lightgreen',
                   4:'slateblue', 5:'thistle', 6:'teal', 7:'magenta', 8:'black'}
    target = 'Mutagenicity'

if name_dataset == 'ba_2motifs':
    target = 'House motif'
    colors_dict = {0: 'cornflowerblue', 1:'tomato', 2:'gold', 3:'lightgreen',
                   4:'slateblue', 5:'thistle', 6:'teal'}


graph_mask_list = torch.split(test_edge_mask, test_sections.tolist())
    
for idx in range(len(edge_index_list)):
    # we have the ground truth only for the mutagenic class (i.e., 0)
    # for ba_2motifs just remove the last condition (== 0)
    if y_pred[idx] == y_true[idx] == 0:
        label = bool(y_pred[idx] + 1)
        G = nx.Graph()
        src, dst = edge_index_list[idx]
        graph_mask = graph_mask_list[idx]
        edge_list = [(int(src[i]), int(dst[i])) for i in range(len(src))]
        for edge in edge_list:
            G.add_edge(edge[0], edge[1])
        src_mask = src[graph_mask.bool()]
        dst_mask = dst[graph_mask.bool()]    
        sampled_edges = [(int(src_mask[i]), int(dst_mask[i])) for i in range(len(src_mask))]
            
        dict_attr = x_attr[idx]
        colors = [colors_dict[dict_attr[node]] for node in G.nodes()]
        
        nx.set_node_attributes(G, dict_attr, name='attr')
        pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color=colors,
                               edgecolors='black', linewidths=1.0)
        nx.draw_networkx_edges(G, pos, edge_list, width=1.5 , alpha=0.1)
        nx.draw_networkx_edges(G, pos, sampled_edges, width=3, edge_color='black')
        # plt.title(f'{target}: {label}')
        plt.savefig(dir_plot+f'{name_dataset}_{name_model}_graph_{idx}.png', 
                    format='png', dpi=200, transparent=False, bbox_inches='tight')
        plt.savefig(dir_plot+f'{name_dataset}_{name_model}_graph_{idx}.pdf', 
                    format='pdf', dpi=200, transparent=False, bbox_inches='tight')
        plt.close()
        


