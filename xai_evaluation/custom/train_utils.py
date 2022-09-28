#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import torch
import numpy as np
from torch_geometric.utils import degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_best_model(model, optimizer, epoch, loss, model_name, dataset_name,
                    ratio_str=None, connected_str=None):
    dir_output = f'./saved_models/{dataset_name}'
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    if ratio_str is not None:
	    torch.save({
		    'epoch': epoch,
		    'model_state_dict': model.state_dict(),
		    'optimizer_state_dict': optimizer.state_dict(),
		    'loss': loss},
        f'./{dir_output}/best_{connected_str}_{model_name}{ratio_str}.pth')
	
    else:
	    torch.save({
	    'epoch': epoch,
	    'model_state_dict': model.state_dict(),
	    'optimizer_state_dict': optimizer.state_dict(),
	    'loss': loss},
        f'{dir_output}/best_{model_name}.pth')
		
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(model, optimizer, loader, ratio):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred, edge_mask, sections = model(data, ratio, train_phase=True)
        loss = model.loss(pred, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    
    out_loss = total_loss / len(loader.dataset)
    return out_loss


@torch.no_grad()
def validation(model, loader, ratio):
    model.eval()
    total_correct = 0
    
    for data in loader: 
        data.to(device)
        out, edge_mask, sections = model(data, ratio, train_phase=False)
        y_pred = out.argmax(-1)
        y_true = data.y
        total_correct += int((y_pred == y_true).sum())
        
    accuracy = total_correct / len(loader.dataset)
    return accuracy

def training_proc(model, optimizer, train_loader, val_loader, test_loader, ratio, 
                  model_name, dataset_name, num_epochs, connected_str):
    ratio_str = str(ratio).replace('.', '')
    best_val_acc = 0.0
    best_loss = 100.0
    patience = num_epochs
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        loss = train(model, optimizer, train_loader, ratio)
        train_acc = validation(model, train_loader, ratio)
        val_acc = validation(model, val_loader, ratio)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_loss = loss
            patience_counter = 0
            test_acc = validation(model, test_loader, ratio)
            print(f'Epoch: {epoch:03d} - Val Acc: {best_val_acc:.4f} - Test Acc: {test_acc:.4f}')
            save_best_model(model, optimizer, epoch, loss, model_name, 
                            dataset_name, ratio_str, connected_str)

        elif val_acc == best_val_acc:
            if loss < best_loss:
                best_val_acc = val_acc
                best_loss = loss
                patience_counter = 0
                test_acc = validation(model, test_loader, ratio)
                print(f'Epoch: {epoch:03d} - Val Acc: {best_val_acc:.4f} - Test Acc: {test_acc:.4f}')
                save_best_model(model, optimizer, epoch, loss, model_name, 
                                dataset_name, ratio_str, connected_str)

            else:
                patience_counter += 1
        else:
            patience_counter += 1
                
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f} '
          f'Val Acc: {val_acc:.4f}')
        if patience_counter == patience:
            break
        
    return model, best_val_acc


@torch.no_grad()
def test(model, loader, ratio):
    model.eval()
    total_correct = 0
    edge_mask_out = np.array([])
    sections_out = np.array([], dtype=int)
    y_pred_out = np.array([])
    y_out = np.array([])
    
    for data in loader: 
        data.to(device)
        out, edge_mask, sections = model(data, ratio, train_phase=False)
        y_pred = out.argmax(-1)
        y_true = data.y
        total_correct += int((y_pred == y_true).sum())
        edge_mask_out = np.append(edge_mask_out, edge_mask.cpu())
        sections_out = np.append(sections_out, sections)
        y_pred_out = np.append(y_pred_out, y_pred.cpu())
        y_out = np.append(y_out, y_true.cpu())
        
    accuracy = total_correct / len(loader.dataset)
    return accuracy, torch.tensor(edge_mask_out), sections_out, y_pred_out, y_out
    
def save_test_results(dir_results, y_true, y_pred, edge_mask, sections):
	if not os.path.exists(dir_results):
		os.makedirs(dir_results)
		
	out_filename = dir_results + 'y_true_test.pkl'
	with open(out_filename, 'wb') as outfile:
		pickle.dump(y_true, outfile)
		outfile.close()
		
	out_filename = dir_results + 'y_pred_test.pkl'
	with open(out_filename, 'wb') as outfile:
		pickle.dump(y_pred, outfile)
		outfile.close()
		
	out_filename = dir_results + 'edge_mask.pkl'
	with open(out_filename, 'wb') as outfile:
		pickle.dump(edge_mask, outfile)
		outfile.close()
		
	out_filename = dir_results + 'test_sections.pkl'
	with open(out_filename, 'wb') as outfile:
		pickle.dump(sections, outfile)
		outfile.close()


def save_test_plotutils(dir_results, test_loader):
	if not os.path.exists(dir_results):
		os.makedirs(dir_results)
	edge_index_list = []
	node_attr_out = []
	for data in test_loader:
		src, dst = data.edge_index
		sections = degree(data.batch[src], dtype=torch.long).tolist()
		edge_index_idx = torch.split(data.edge_index, sections, dim=1)
		edge_index_list.extend(edge_index_idx)
		
		x_attr_batch = np.argmax(data.x, axis=1).tolist()
		for i, edge_index in enumerate(edge_index_idx):
		    src, dst = edge_index
		    dict_node_attr = {int(node): x_attr_batch[node] for node in src}
		    
		    node_attr_out.append(dict_node_attr)

	out_filename = dir_results + 'x_attributes.pkl'
	with open(out_filename, 'wb') as outfile:
		pickle.dump(node_attr_out, outfile)
		outfile.close()

	out_filename = dir_results + 'edge_index_list.pkl'
	with open(out_filename, 'wb') as outfile:
		pickle.dump(edge_index_list, outfile)
		outfile.close()

