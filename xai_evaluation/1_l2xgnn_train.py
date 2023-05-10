#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json
import torch
import argparse
from gin import L2XGIN
from gcn import L2XGCN
from gsg import L2XGSG
from torch_geometric.loader import DataLoader
from custom.utils import load_dataset, create_split_idx, parse_boolean
from custom.train_utils import training_proc, test, save_test_results, save_test_plotutils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['ba_2motifs', 'Mutagenicity'],
                    help='Name of the dataset')
parser.add_argument('--model', type=str,
                    choices=['L2XGIN', 'L2XGCN', 'L2XGSG'], default='L2XGIN',
                    help='Name of the model')
parser.add_argument('--connected', type=parse_boolean, default=False, 
                    help='Get connected output or not')
parser.add_argument('--ratio', type=float,
                    help='Ratio of restrained edges')
parser.add_argument('--split', type=int, default=0,
                    help='Fold to evaluate {0-4}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir_data = './datasets/'

args = parser.parse_args()
name_dataset = args.dataset
name_model = args.model
connected_flag = args.connected
ratio = args.ratio
split = args.split

if connected_flag:
    connected_str = 'connected'
else:
    connected_str = 'disconnected'

ratio_str = str(ratio).replace('.', '')
bs = 64 	

dataset, num_features, num_classes = load_dataset(name_dataset, dir_data)

dir_split = f'{dir_data}/data_splits/'
if not os.path.exists(dir_split):
    os.makedirs(dir_split)
    create_split_idx(dataset, name_dataset)

splits = json.load(open(dir_split + f'{name_dataset}_splits.json', 'r'))
test_index = splits[split]['test']
train_index = splits[split]['model_selection'][0]['train']
val_index = splits[split]['model_selection'][0]['validation']

train_data = [dataset[idx] for idx in train_index]
test_data = [dataset[idx] for idx in test_index]
val_data = [dataset[idx] for idx in val_index]

# hyperparameters should be changed according to the backbone model we want to explain
if name_model == 'L2XGIN':
    num_epochs = 200
    n_layers = 3
    hidden = 64
    model = L2XGIN(num_features, hidden, num_classes, 
                        n_layers, connected_flag).to(device)
    
if name_model == 'L2XGCN':
    num_epochs = 200
    n_layers = 3
    hidden = 128
    model = L2XGCN(num_features, hidden, num_classes, 
                        n_layers, connected_flag).to(device)
                        
if name_model == 'L2XGSG':
    num_epochs = 200
    n_layers = 3
    hidden = 64
    model = L2XGSG(num_features, hidden, num_classes, 
                        n_layers, connected_flag).to(device)
    
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_data, batch_size=bs)
test_loader = DataLoader(test_data, batch_size=bs)

print(f'{name_dataset} - {name_model} - Ratio: {ratio} - Split {split}')
_, best_val_acc = training_proc(model, optimizer, train_loader, val_loader, test_loader,
                                ratio, name_model, name_dataset, split, num_epochs, connected_str)

best_model_cp = torch.load(f'./saved_models/{name_dataset}/s{split}/best_{connected_str}_{name_model}{ratio_str}.pth')
best_model_epoch = best_model_cp['epoch']
print(best_model_epoch)
model.load_state_dict(best_model_cp['model_state_dict'])

test_acc, test_edge_mask, test_sections, \
    y_pred_test, y_true_test = test(model, test_loader, ratio)
    
print(f'Epoch: {best_model_epoch:03d}, Test Acc: {test_acc:.4f}')

dir_results = f'./results/{name_dataset}/{connected_str}_{name_model}/{ratio_str}/s{split}/' 
save_test_results(dir_results, y_true_test, y_pred_test, 
                  test_edge_mask, test_sections)
save_test_plotutils(dir_results, test_loader)
