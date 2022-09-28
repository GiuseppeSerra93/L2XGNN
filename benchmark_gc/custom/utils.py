#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import os, pickle
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import remove_self_loops
from dig.xgraph.dataset import SynGraphDataset, MoleculeDataset
from sklearn.model_selection import StratifiedKFold, train_test_split

def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False

def load_dataset(name_dataset, dir_data):
    if name_dataset in ['MUTAG', 'PROTEINS']:
        torch.manual_seed(7)
        dataset = TUDataset(dir_data, name_dataset).shuffle()
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        
    if name_dataset == 'Mutagenicity':
        dataset = pickle.load(open(dir_data + f'{name_dataset}/{name_dataset}.pkl', 'rb'))
        num_features = dataset[0].x.shape[1]
        num_classes = 2
        
    if name_dataset == 'rnd_2motifs':
        dataset = pickle.load(open(dir_data + f'{name_dataset}/{name_dataset}.pkl', 'rb'))
        num_features = dataset[0].x.shape[1]
        num_classes = 2
        
    if name_dataset == 'ba_2motifs':
        torch.manual_seed(123)
        dataset_orig = SynGraphDataset(dir_data, name_dataset).shuffle()
        num_features = dataset_orig.num_features
        num_classes = dataset_orig.num_classes
        dataset = []
        for idx, g in enumerate(dataset_orig):
            new_g = Data(x=g.x*10, # all 1s, as written in the literature
                         edge_index=remove_self_loops(g.edge_index)[0],
                         y=g.y)
            dataset.append(new_g)

    if name_dataset == 'BBBP':
        torch.manual_seed(7)
        dataset_orig = MoleculeDataset(dir_data, name_dataset).shuffle()
        num_features = dataset_orig.num_features
        num_classes = dataset_orig.num_classes
        dataset = []
        for idx, g in enumerate(dataset_orig):
            new_g = Data(x=g.x.float(),
                          edge_index=remove_self_loops(g.edge_index)[0],
                          y=g.y[0].long())
            dataset.append(new_g)    
            
    return dataset, num_features, num_classes


def create_splits(dataset, name_dataset, num_features, num_classes):
    dir_data = './datasets/'
    dir_split = f'{dir_data}splits/{name_dataset}/'
    if not os.path.exists(dir_split):
        os.makedirs(dir_split)
    
    skf = StratifiedKFold(n_splits=5, random_state=193, shuffle=True)
    y = [data.y[0] for data in dataset]
    
    i = 0
    for train_index, test_index in skf.split(dataset, y):
        train_data = [dataset[train_idx] for train_idx in train_index]
        test_val_data = [dataset[test_idx] for test_idx in test_index]
        y_test_val_data = [y[test_idx] for test_idx in test_index]
        
        test_data, val_data = train_test_split(test_val_data, 
                                               stratify=y_test_val_data,
                                               test_size=0.5,
                                               random_state=123)
        
        print(len(train_data), len(test_data), len(val_data))
            
        out_filename = dir_split + f'train_{i}.pkl'
        with open(out_filename, 'wb') as outfile:
            pickle.dump(train_data, outfile)
            outfile.close() 
            
        out_filename = dir_split + f'test_{i}.pkl'
        with open(out_filename, 'wb') as outfile:
            pickle.dump(test_data, outfile)
            outfile.close()
            
        out_filename = dir_split + f'validation_{i}.pkl'
        with open(out_filename, 'wb') as outfile:
            pickle.dump(val_data, outfile)
            outfile.close() 
        
        i += 1
            
    out_filename = dir_split + 'num_features.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(num_features, outfile)
        outfile.close()
    out_filename = dir_split + 'num_classes.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(num_classes, outfile)
        outfile.close()
        
def save_data_splits(name_dataset):
    dir_data = './datasets/'
    dataset, num_features, num_classes = load_dataset(name_dataset, dir_data)
    create_splits(dataset, name_dataset, num_features, num_classes)
                
                
def load_data_splits(name_dataset, dir_data, i):
    dir_split = f'{dir_data}splits/{name_dataset}/'
    train_data = pickle.load(open(dir_split + f'train_{i}.pkl', 'rb'))
    test_data = pickle.load(open(dir_split + f'test_{i}.pkl', 'rb'))
    val_data = pickle.load(open(dir_split + f'validation_{i}.pkl', 'rb'))
    num_features = pickle.load(open(dir_split + 'num_features.pkl', 'rb'))
    num_classes = pickle.load(open(dir_split + 'num_classes.pkl', 'rb'))
    
    return train_data, test_data, val_data, num_features, num_classes

