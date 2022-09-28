#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import torch
from custom.utils import load_data_splits
from dig.xgraph.models import GIN_3l
from torch_geometric.loader import DataLoader

dir_data = './datasets/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir_data = './datasets/'
name_dataset = 'ba_2motifs' # 'Mutagenicity', 'ba_2motifs',
name_model = 'GIN_3l'
split = 2

dir_output = f'./pred_mask/{name_dataset}/' 
if not os.path.exists(dir_output):
    os.makedirs(dir_output)

train_data, test_data, val_data, num_features, num_classes = load_data_splits(name_dataset, dir_data, split)

best_model_cp = torch.load(f'./saved_models/{name_dataset}/best_{name_model}.pth')
model = GIN_3l('graph', num_features, 64, num_classes).to(device)
model.load_state_dict(best_model_cp['model_state_dict'])

# =============================================================================
# GNN-Explainer
# =============================================================================
from dig.xgraph.method import GNNExplainer
from torch_geometric.utils import to_undirected
from torch_geometric.utils.convert import to_networkx
explainer = GNNExplainer(model, epochs=500, lr=0.01, explain_graph=True)
sparsity = 0.6
edge_mask_out = []
y_pred = []
for i, data in enumerate(test_data):
    print(f'Explain graph {i}')
    data.to(device)
    logits = model(data.x, data.edge_index)
    target = logits.argmax(-1).item()
    edge_masks, hard_edge_masks, related_preds = \
                explainer(data.x, data.edge_index,
                          sparsity=sparsity,
                          num_classes=num_classes)

    # remove mask over self-loop edges (i.e., [: -data.x.shape[0]])
    hard_edge_mask = hard_edge_masks[target][: -data.x.shape[0]].bool()
    src_mask, dst_mask = to_undirected(data.edge_index[:, hard_edge_mask])
    edge_list_mask = [tuple(x) for x in zip(src_mask.cpu().numpy(), dst_mask.cpu().numpy())]

    G = to_networkx(data, remove_self_loops=True)
    edge_mask = torch.zeros(data.edge_index.shape[1])
    for i, e in enumerate(G.edges()):
        if e in edge_list_mask:
            edge_mask[i] = 1.0
    edge_mask_out.append(edge_mask)
    y_pred.append(target)
    
out_filename = dir_output + 'edge_mask_GNNExplainer.pkl'
with open(out_filename, 'wb') as outfile:
    pickle.dump(edge_mask_out, outfile)
    outfile.close()
    
# =============================================================================
# GradCAM
# =============================================================================
# from dig.xgraph.method import GradCAM
# explainer = GradCAM(model, explain_graph=True)
# sparsity = 0.7
# edge_mask_out = []
# y_pred = []
# for i, data in enumerate(test_data):
#     print(f'Explain graph {i}')
#     data.to(device)
#     logits = model(data.x, data.edge_index)
#     target = logits.argmax(-1).item()
#     walks, masks, related_preds = \
#             explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes)
#     masks[target][: -data.x.shape[0]].bool()
    
#     # remove mask over self-loop edges (i.e., [: -data.x.shape[0]])
#     edge_mask = masks[target][: -data.x.shape[0]].bool()
#     edge_mask_out.append(edge_mask)
#     y_pred.append(target)
    
# out_filename = dir_output + 'edge_mask_GradCAM.pkl'
# with open(out_filename, 'wb') as outfile:
#     pickle.dump(edge_mask_out, outfile)
#     outfile.close()

# =============================================================================
# PGE-Explainer
# =============================================================================
# from dig.xgraph.method import PGExplainer
# from dig.xgraph.method.pgexplainer import PlotUtils
# from torch_geometric.utils.convert import to_networkx

# explainer = PGExplainer(model, in_channels=128, device=device, epochs=100, t1=2.0)
# explainer.train_explanation_network(train_data)

# edge_mask_out = []
# plotutils = PlotUtils(dataset_name='ba_2motifs', is_show=True)
# for i, data in enumerate(test_data):
#     print(f'Explain graph {i}')
#     with torch.no_grad():
#         walks, masks, related_preds = \
#             explainer(data.x, data.edge_index, top_k=20, y=data.y, num_classes=num_classes)

#         edge_mask = torch.zeros(data.edge_index.shape[1])
#         G = to_networkx(data, remove_self_loops=True)
#         edgelist = plotutils.get_topk_edges_subgraph(data.edge_index.cpu(), masks[0].cpu(), 
#                                           top_k=10, un_directed=True)[1]
#         for i, (src, dst) in enumerate(G.edges()):
#             if (src, dst) in edgelist:
#                 edge_mask[i] = 1.0
                              
#         edge_mask_out.append(edge_mask)        
    
# out_filename = dir_output + 'edge_mask_PGEExplainer.pkl'
# with open(out_filename, 'wb') as outfile:
#     pickle.dump(edge_mask_out, outfile)
#     outfile.close()


# plotutils = PlotUtils(dataset_name='ba_2motifs', is_show=True)
# plotutils.get_topk_edges_subgraph(data.edge_index.cpu(), masks[0].cpu(), top_k=20)
# explainer.visualization(data, edge_mask=masks[0], top_k=20, plot_utils=plotutils)

# torch.logical_and(masks[0].cpu(), masks[0].cpu() == 1.0)
# edgelist = plotutils.get_topk_edges_subgraph(data.edge_index.cpu(), masks[0].cpu(), 
#                                   top_k=10, un_directed=True)[1]

# =============================================================================
# SubgraphX
# =============================================================================
# from dig.xgraph.method import SubgraphX
# from dig.xgraph.method.subgraphx import find_closest_node_result
# from torch_geometric.utils.convert import to_networkx

# explainer = SubgraphX(model, num_classes, device)
# edge_mask_out = []
# y_pred = []
# for i, data in enumerate(test_data):
#     print(f'Explain graph {i}')
#     data = data.to(device)
#     logits = model(data.x, data.edge_index)
#     target = logits.argmax(-1).item()
    
#     _, explanation_results, related_preds = explainer(data.x, data.edge_index,
#                                                       max_nodes=10)
#     explanation_results = explainer.read_from_MCTSInfo_list(explanation_results[target])
#     result = find_closest_node_result(explanation_results, max_nodes=10)
#     node_list = result.coalition
    
#     edge_mask = torch.zeros(data.edge_index.shape[1])
#     G = to_networkx(data, remove_self_loops=True)
    
#     for i, (src, dst) in enumerate(G.edges()):
#         if src in node_list and dst in node_list:
#             edge_mask[i] = 1.0
            
#     edge_mask_out.append(edge_mask)
#     y_pred.append(target)
    
# out_filename = dir_output + 'edge_mask_SubgraphX.pkl'
# with open(out_filename, 'wb') as outfile:
#     pickle.dump(edge_mask_out, outfile)
#     outfile.close()


dir_output = f'./pred_mask/{name_dataset}/' 
# out_filename = dir_output + 'edge_mask_SubgraphX.pkl'
out_filename = dir_output + 'edge_mask_GNNExplainer.pkl'
# out_filename = dir_output + 'edge_mask_GradCAM.pkl'
# out_filename = dir_output + 'edge_mask_PGEExplainer.pkl'
graph_mask_list = pickle.load(open(out_filename, 'rb'))
graph_mask_list = edge_mask_out
accs = []
pr = []
rec = []
f1s = []

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
            accs.append(accuracy)
            pr.append(precision)
            rec.append(recall)
            f1s.append(f1)
        
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
            accs.append(accuracy)
            pr.append(precision)
            rec.append(recall)
            f1s.append(f1)
    
print(f'Accuracy:\t {sum(accs)/len(accs)}')
print(f'Precision:\t {sum(pr)/len(pr)}')
print(f'Recall:  \t {sum(rec)/len(rec)}')
print(f'F1-score:\t {sum(f1s)/len(f1s)}')
