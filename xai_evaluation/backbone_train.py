#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dig.xgraph.models import GIN_3l
from custom.utils import load_data_splits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)
    
def train(model, optimizer, loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.batch, mode='graph')
        loss = F.cross_entropy(pred, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * num_graphs(data)
    
    out_loss = total_loss / len(loader.dataset)
    return out_loss

@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    correct = 0
    for data in loader: 
        data.to(device)
        pred = model(data.x, data.edge_index, data.batch, mode='graph')
        y_pred = pred.max(1)[1]
        correct += y_pred.eq(data.y.view(-1)).sum().item()
        
    accuracy = correct / len(loader.dataset)
    return accuracy

def save_best_model(model, optimizer, epoch, loss, model_name, dataset_name):
    dir_output = f'./saved_models/{dataset_name}'
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
	
    else:
	    torch.save({
	    'epoch': epoch,
	    'model_state_dict': model.state_dict(),
	    'optimizer_state_dict': optimizer.state_dict(),
	    'loss': loss},
        f'{dir_output}/best_{model_name}.pth')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['ba_2motifs', 'Mutagenicity'],
                    help='Name of the dataset')
name_dataset = args.dataset

dir_data = './datasets/'
name_model = 'GIN_3l'
split = 2
num_epochs = 200
bs = 64
train_data, test_data, val_data, num_features, num_classes = load_data_splits(name_dataset, dir_data, split)
model = GIN_3l('graph', num_features, 64, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_data, batch_size=bs)
val_loader = DataLoader(val_data, batch_size=bs)
best_val_acc = 0.0
for epoch in range(1, num_epochs + 1):

    loss = train(model, optimizer, train_loader)
    train_acc = eval_acc(model, train_loader)
    val_acc = eval_acc(model, val_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f} '
      f'Val Acc: {val_acc:.4f}')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = eval_acc(model, test_loader)
        best_loss = loss
        print(f'Epoch: {epoch:03d} - Val Acc: {best_val_acc:.4f} - Test Acc: {test_acc:.4f}')
        save_best_model(model, optimizer, epoch, loss, name_model, name_dataset)

    elif val_acc == best_val_acc:
        if loss < best_loss:
            best_val_acc = val_acc
            test_acc = eval_acc(model, test_loader)
            best_loss = loss
            print(f'Epoch: {epoch:03d} - Val Acc: {best_val_acc:.4f} - Test Acc: {test_acc:.4f}')
            save_best_model(model, optimizer, epoch, loss, name_model, name_dataset)
