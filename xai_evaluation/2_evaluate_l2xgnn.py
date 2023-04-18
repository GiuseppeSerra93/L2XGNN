#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pickle, json
import torch
import numpy as np
from custom.utils import load_dataset, parse_boolean
from sklearn.metrics import classification_report, accuracy_score

def explanation_acc_report(gt_mask, pred_mask):
    report = classification_report(gt_mask, pred_mask, output_dict=True)
    accuracy = accuracy_score(gt_mask, pred_mask)
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']
    return accuracy, precision, recall, f1
    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['ba_2motifs', 'Mutagenicity'],
                    help='Name of the dataset')
parser.add_argument('--model', type=str,
                    choices=['L2XGIN', 'L2XGCN'], default='L2XGIN',
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

dir_data = './datasets/'
dir_split = f'{dir_data}/data_splits/'

dataset, num_features, num_classes = load_dataset(name_dataset, dir_data)

splits = json.load(open(dir_split + f'{name_dataset}_splits.json', 'r'))

ratio_str = str(ratio).replace('.', '')
if connected_flag:
    connected_str = 'connected'
else:
    connected_str = 'disconnected'

accs = []
pr = []
rec = []
f1s = []
for split in range(5):
    test_index = splits[split]['test']
    test_data = [dataset[idx] for idx in test_index]
    dir_results = f'./results/{name_dataset}/{connected_str}_{name_model}/{ratio_str}/s{split}/' 
    test_edge_mask = pickle.load(open(dir_results + 'edge_mask.pkl', 'rb'))
    test_sections = pickle.load(open(dir_results + 'test_sections.pkl', 'rb'))
    y_pred = pickle.load(open(dir_results + 'y_pred_test.pkl', 'rb'))
    graph_mask_list = torch.split(test_edge_mask, test_sections.tolist())
    
    
    accs_tmp = []
    pr_tmp = []
    rec_tmp = []
    f1s_tmp = []
    if name_dataset == 'ba_2motifs':
        for i, data in enumerate(test_data):
            if y_pred[i] == test_data[i].y:
                gt_mask = torch.logical_and(data.edge_index[0] >= 20, data.edge_index[1] >= 20).int()
                pred_mask = graph_mask_list[i]
                accuracy, precision, recall, f1 = explanation_acc_report(gt_mask, pred_mask)
                accs_tmp.append(accuracy)
                pr_tmp.append(precision)
                rec_tmp.append(recall)
                f1s_tmp.append(f1)
        
    else:
        for i, data in enumerate(test_data):
        # we have the ground truth only for the mutagenic class (i.e., 0)
            if y_pred[i] == test_data[i].y == 0:
                gt_mask = test_data[i].ground_truth
                pred_mask = graph_mask_list[i]
                accuracy, precision, recall, f1 = explanation_acc_report(gt_mask, pred_mask)
                accs_tmp.append(accuracy)
                pr_tmp.append(precision)
                rec_tmp.append(recall)
                f1s_tmp.append(f1)
                
    accs.append(np.mean(accs_tmp))   
    pr.append(np.mean(pr_tmp)) 
    rec.append(np.mean(rec_tmp)) 
    f1s.append(np.mean(f1s_tmp))

print(f'Accuracy:\t {np.mean(accs):.3f} ± {np.std(accs):.3f}')
print(f'Precision:\t {np.mean(pr):.3f} ± {np.std(pr):.3f}')
print(f'Recall:  \t {np.mean(rec):.3f} ± {np.std(rec):.3f}')
print(f'F1-score:\t {np.mean(f1s):.3f} ± {np.std(f1s):.3f}')