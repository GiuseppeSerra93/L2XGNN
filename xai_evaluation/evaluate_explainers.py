#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json
import pickle
import torch
import numpy as np
from custom.utils import load_dataset
from dig.xgraph.models import GIN_3l

dir_data = './datasets/'
dir_split = f'{dir_data}/data_splits/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
name_dataset = 'ba_2motifs' # 'Mutagenicity', 'ba_2motifs',
name_model = 'GIN_3l'
split = 1                   # {0-4}

accs = []
pr = []
rec = []
f1s = []
for split in range(5):
    dir_output = f'./pred_mask/{name_dataset}/s{split}/' 
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    
    dataset, num_features, num_classes = load_dataset(name_dataset, dir_data)
    splits = json.load(open(dir_split + f'{name_dataset}_splits.json', 'r'))
    test_index = splits[split]['test']
    train_index = splits[split]['model_selection'][0]['train']
    val_index = splits[split]['model_selection'][0]['validation']
    
    train_data = [dataset[idx] for idx in train_index]
    test_data = [dataset[idx] for idx in test_index]
    val_data = [dataset[idx] for idx in val_index]
    
    best_model_cp = torch.load(f'./saved_models/{name_dataset}/s{split}/best_{name_model}.pth')
    model = GIN_3l('graph', num_features, 64, num_classes).to(device)
    model.load_state_dict(best_model_cp['model_state_dict'])

    # out_filename = dir_output + 'edge_mask_SubgraphX.pkl'
    # out_filename = dir_output + 'edge_mask_GNNExplainer.pkl'
    # out_filename = dir_output + 'edge_mask_GradCAM.pkl'
    # out_filename = dir_output + 'edge_mask_PGEExplainer.pkl'
    out_filename = dir_output + 'edge_mask_GNNLRP.pkl'
    
    graph_mask_list = pickle.load(open(out_filename, 'rb'))
    
    if out_filename == dir_output + 'edge_mask_GNNLRP.pkl':
        graph_mask_list = [mask > 0.2 for mask in graph_mask_list]

    accs_tmp = []
    pr_tmp = []
    rec_tmp = []
    f1s_tmp = []
    
    from sklearn.metrics import classification_report, accuracy_score
    if name_dataset == 'ba_2motifs':
        for i, data in enumerate(test_data):
            data = data.to(device)
            logits = model(data.x, data.edge_index)
            y_pred = logits.argmax(-1).item()
            if y_pred == test_data[i].y:
                target = str(bool(y_pred))
                gt_mask = torch.logical_and(data.edge_index[0] >= 20, data.edge_index[1] >= 20).int().cpu()
                pred_mask = graph_mask_list[i].cpu()
                
                report = classification_report(gt_mask, pred_mask, output_dict=True)
                accuracy = accuracy_score(gt_mask, pred_mask)
                precision = report['1']['precision']
                recall = report['1']['recall']
                f1 = report['1']['f1-score']
                accs_tmp.append(accuracy)
                pr_tmp.append(precision)
                rec_tmp.append(recall)
                f1s_tmp.append(f1)
            
    else:
        for i, data in enumerate(test_data):
            data = data.to(device)
            logits = model(data.x, data.edge_index)
            y_pred = logits.argmax(-1).item()
            # we have the ground truth only for the mutagenic class (i.e., 0)
            if y_pred == test_data[i].y == 0:
                gt_mask = test_data[i].ground_truth.cpu()
                pred_mask = graph_mask_list[i].cpu()
                report = classification_report(gt_mask, pred_mask, output_dict=True)
                accuracy = accuracy_score(gt_mask, pred_mask)
                precision = report['1']['precision']
                recall = report['1']['recall']
                f1 = report['1']['f1-score']
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
    

