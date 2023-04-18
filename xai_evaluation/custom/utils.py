#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import pickle, json
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import remove_self_loops
from dig.xgraph.dataset import SynGraphDataset, MoleculeDataset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

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


def create_split_idx(dataset, name_dataset):
    dir_data = './datasets/'
    dir_split = f'{dir_data}/data_splits/'
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=193)
    y = [data.y[0] for data in dataset]
    
    splits = []
    for train_index, test_val_index in skf.split(dataset, y):
        dict_split = {}
        y_test_val_data = [y[idx] for idx in test_val_index]
        
        sss = StratifiedShuffleSplit(n_splits=1, 
                                     test_size=0.5,
                                     random_state=123)
        
        for test_idx, val_idx in sss.split(test_val_index, y_test_val_data):
            test_index = [int(test_val_index[idx]) for idx in test_idx]
            val_index = [int(test_val_index[idx]) for idx in val_idx]
            
            print(len(train_index), len(test_index), len(val_index))
            dict_split['test'] = test_index
            dict_split['model_selection'] = [{'train':train_index.tolist(), 
                                              'validation':val_index}]
            
        splits.append(dict_split)
        
    out_filename = dir_split + f'{name_dataset}_splits.json'
    with open(out_filename, 'w') as outfile:
        json.dump(splits, outfile)
        outfile.close()   

