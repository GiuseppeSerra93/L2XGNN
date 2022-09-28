#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pickle
import torch
from custom.utils import load_data_splits, parse_boolean
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
                    
args = parser.parse_args()
name_dataset = args.dataset
name_model = args.model
connected_flag = args.connected
ratio = args.ratio

dir_data = './datasets/'
split = 2
train_data, test_data, val_data, num_features, num_classes = load_data_splits(name_dataset, dir_data, split)

ratio_str = str(ratio).replace('.', '')
if connected_flag:
    connected_str = 'connected'
else:
    connected_str = 'disconnected'

dir_results = f'./results/{name_dataset}/{connected_str}_{name_model}/{ratio_str}/' 
test_edge_mask = pickle.load(open(dir_results + 'edge_mask.pkl', 'rb'))
test_sections = pickle.load(open(dir_results + 'test_sections.pkl', 'rb'))
y_pred = pickle.load(open(dir_results + 'y_pred_test.pkl', 'rb'))
graph_mask_list = torch.split(test_edge_mask, test_sections.tolist())

accs = []
pr = []
rec = []
f1s = []
if name_dataset == 'ba_2motifs':
    for i, data in enumerate(test_data):
        if y_pred[i] == test_data[i].y:
            gt_mask = torch.logical_and(data.edge_index[0] >= 20, data.edge_index[1] >= 20).int()
            pred_mask = graph_mask_list[i]
            accuracy, precision, recall, f1 = explanation_acc_report(gt_mask, pred_mask)
            accs.append(accuracy)
            pr.append(precision)
            rec.append(recall)
            f1s.append(f1)
    
else:
    for i, data in enumerate(test_data):
    # we have the ground truth only for the mutagenic class (i.e., 0)
        if y_pred[i] == test_data[i].y == 0:
            gt_mask = test_data[i].ground_truth
            pred_mask = graph_mask_list[i]
            accuracy, precision, recall, f1 = explanation_acc_report(gt_mask, pred_mask)
            accs.append(accuracy)
            pr.append(precision)
            rec.append(recall)
            f1s.append(f1)

print(f'Accuracy:\t {sum(accs)/len(accs)}')
print(f'Precision:\t {sum(pr)/len(pr)}')
print(f'Recall:  \t {sum(rec)/len(rec)}')
print(f'F1-score:\t {sum(f1s)/len(f1s)}')
