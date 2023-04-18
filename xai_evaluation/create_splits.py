#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, pickle, json
from custom.utils import load_dataset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

dir_data = './datasets/'
name_dataset = 'ba_2motifs' # 'Mutagenicity', 'ba_2motifs',
dataset, num_features, num_classes = load_dataset(name_dataset, dir_data)

dir_split = f'{dir_data}/data_splits/'
if not os.path.exists(dir_split):
    os.makedirs(dir_split)

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