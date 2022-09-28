#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import math
import itertools
import numpy as np
import networkx as nx
from torch import Tensor
from typing import Optional, Tuple
from torch.distributions.gamma import Gamma
from torch_geometric.utils import to_undirected
    
    
class GraphIMLETopKConnected(torch.autograd.Function):
    tau: float = 1.0
    lambda_: float = 10.0    # choose from [10.0, 100.0, 1000.0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def sample_gumbel_k(shape: torch.Size, 
                        k: int, 
                        tau: float) -> Tensor:
        
        device = GraphIMLETopKConnected.device
        sog = list(map(lambda t: Gamma(1.0/k, t/k).sample(shape),
                      torch.arange(1, 10, dtype=torch.float32)))
        sog = torch.sum(torch.stack(sog), dim=0)
        sog = sog - math.log(10.0)
        sog = tau * (sog / k)
        
        return sog.to(device)
    
    @staticmethod
    def sample_k(logits: Tensor, 
                  k: int, 
                  tau: float, 
                  edge_index: Tensor,
                  train_phase: bool,
                  sample: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        
        device = GraphIMLETopKConnected.device
            
        if train_phase:
            if sample is None:
                sample = GraphIMLETopKConnected.sample_gumbel_k(logits.shape, k, tau)
            gumbel_softmax_sample = logits + sample
        else:
        	gumbel_softmax_sample = logits
        	
        z = torch.zeros(logits.shape, dtype=torch.float32).to(device)
        not_sampled = torch.ones(logits.shape, dtype=torch.bool)
        sampled_edge_list = []

        G = nx.Graph()
        edge_map = {}
        value = 0
        src, dst = edge_index
        edge_list = [(int(src[i]), int(dst[i])) for i in range(len(src))]

        for edge in edge_list:
            src, dst = edge
            G.add_edge(src, dst)
            edge_map[edge] = value
            edge_map[(dst,src)] = value
            value += 1

        # edge sampling initialization
        # get index of the first sampled edge
        sampled_edge_idx = torch.topk(gumbel_softmax_sample, 1, sorted=True)[1][-1]
        # set discrete sampling weight to 1
        z[sampled_edge_idx] = 1
        # change not_sampled flag to False
        not_sampled[sampled_edge_idx] = 0
        # append sampled edge to the sampled_edge_list
        sampled_edge_list.append(edge_list[sampled_edge_idx])
        # get the list of sampled nodes from the sampled_edge_list
        sampled_nodes = list(itertools.chain(*sampled_edge_list))
        # use the sampled_node list to find the adjacent edges
        connected_edges = G.edges(sampled_nodes)
        # get indices of the adjacent edges
        idx_list = torch.tensor([edge_map[edge] for edge in connected_edges])
        # get the not_sampled flag of the adjacent edges
        not_sampled_i = not_sampled[idx_list]
        # select logits of the adjacent edges that are not already sampled
        gumbel_softmax_sample_sub = gumbel_softmax_sample[idx_list][not_sampled_i]
        
        # single edge sampling
        for i in range(1, k):
            # check if the list is not empty
            if len(gumbel_softmax_sample_sub):
                # here, we are using a subset of the gumbel_softmax_sample list
                # we first get the index from the sublist
                idx = torch.topk(gumbel_softmax_sample_sub, 1, sorted=True)[1][-1]
                # then, we extract the index of the original list (using idx_list)
                sampled_edge_idx = idx_list[not_sampled_i][idx]
                z[sampled_edge_idx] = 1
                not_sampled[sampled_edge_idx] = 0
                sampled_edge_list.append(edge_list[sampled_edge_idx])
                sampled_nodes = list(itertools.chain(*sampled_edge_list))
                connected_edges = G.edges(sampled_nodes)
                idx_list = torch.tensor([edge_map[edge] for edge in connected_edges])
                not_sampled_i = not_sampled[idx_list]
                gumbel_softmax_sample_sub = gumbel_softmax_sample[idx_list][not_sampled_i]
            else:
                return z, sample
        
        return z, sample

    @staticmethod
    def forward(ctx, logits: Tensor, edge_index: Tensor,
                mask: Tensor, sections: tuple, ratio: float, train_phase: bool):
        
        device = GraphIMLETopKConnected.device
        mask_list = torch.split(mask, split_size_or_sections=sections)
        logits_list = torch.split(logits, split_size_or_sections=sections)
        edge_index_list = torch.split(edge_index, split_size_or_sections=sections, dim=1)
        
        z = np.array([])
        sample = np.array([])
        # for-loop to sample from each graph[i] in the batch separately
        for i in range(len(mask_list)):
            # get the mask to select one weight per edge (i.e., directed version of the graph)
            mask_i = mask_list[i].to(device)
            # logits of the 'directed' graph only
            logits_i = logits_list[i][mask_i].to(device)
            # edge_index list of the 'directed' graph
            directed_edge_index = edge_index_list[i][:, mask_i]
            # select k depending on the number of edges
            num_edges = directed_edge_index.shape[1]
            k_i = int(np.ceil(ratio * num_edges))
            if k_i == 0:
            	k_i = num_edges
            # sample from the logits of the 'directed' graph
            z_tmp, sample_tmp = GraphIMLETopKConnected.sample_k(logits_i, k_i, 
                                                           GraphIMLETopKConnected.tau,
                                                           directed_edge_index,
                                                           train_phase)
            # assign the (discrete) sampling weights to the original graph
            _, z_out = to_undirected(directed_edge_index, z_tmp)
            # append the (discrete) sampling weights to the output tensor
            z = np.append(z, z_out.cpu())
            # assign the SoG values to the original graph
            _, sample_out = to_undirected(directed_edge_index, sample_tmp)
            # append the SoG values to the output tensor
            sample = np.append(sample, sample_out.cpu())

        # convert np.arrays to tensors
        z = torch.tensor(z, dtype=torch.float32).to(device)
        sample = torch.tensor(sample, dtype=torch.float32).to(device)
        # save tensors for backward function
        ctx.ratio = ratio
        ctx.train_phase = train_phase
        ctx.sections = sections
        ctx.mask_list = mask_list
        ctx.logits_list = logits_list
        ctx.edge_index_list = edge_index_list
        ctx.save_for_backward(logits, edge_index, mask, z, sample)
        return z
    
    @staticmethod
    def backward(ctx, dy: Tensor):
        
        logits, edge_index, mask, z, sample = ctx.saved_tensors
        ratio = ctx.ratio
        train_phase = ctx.train_phase
        sections = ctx.sections
        mask_list = ctx.mask_list
        logits_list = ctx.logits_list
        edge_index_list = ctx.edge_index_list
        device = GraphIMLETopKConnected.device
        
        dy_list = torch.split(dy, split_size_or_sections=sections)
        z_list = torch.split(z, split_size_or_sections=sections)
        sample_list = torch.split(sample, split_size_or_sections=sections)
        
        gradient = np.array([])
        for i in range(len(mask_list)):
            # get the mask to select one weight per edge (i.e., directed version of the graph)
            mask_i = mask_list[i].to(device)
            # logits of the 'directed' graph only
            logits_i = logits_list[i][mask_i].to(device)
            # get the gradient of the 'directed' graph
            dy_i = dy_list[i][mask_i].to(device)
            # get SoG values of the 'directed' graph
            sample_i = sample_list[i][mask_i].to(device)
            # get the (discrete) sampling weights of the ORIGINAL graph (NO MASK NEEDED)
            z = z_list[i].to(device)
            # edge_index list of the 'directed' graph
            directed_edge_index = edge_index_list[i][:, mask_i]
            # select k depending on the number of edges
            num_edges = directed_edge_index.shape[1]
            k_i = int(np.ceil(ratio * num_edges))
            if k_i == 0:
            	k_i = num_edges
            # compute target logits of the 'directed' graph
            target_logits = logits_i - (GraphIMLETopKConnected.lambda_ * dy_i)
            # sample from the target_logits of the 'directed' graph
            map_dy_tmp, _ = GraphIMLETopKConnected.sample_k(target_logits, k_i, 
                                                       GraphIMLETopKConnected.tau,
                                                       directed_edge_index, train_phase,
                                                       sample_i)

            # assign the (discrete) sampling weights to the original graph
            _, map_dy_out = to_undirected(directed_edge_index, map_dy_tmp)
            
            # gradient of the undirected graph
            gradient_out = z - map_dy_out
            # append the graph gradient to output tensor
            gradient = np.append(gradient, gradient_out.cpu())
            
        # convert np.array to tensor
        gradient = torch.tensor(gradient, dtype=torch.float32).to(device)
        return gradient, None, None, None, None, None
    
class GraphIMLETopK(torch.autograd.Function):
    tau: float = 1.0
    lambda_: float = 10.0    # choose from [10.0, 100.0, 1000.0] 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def sample_gumbel_k(shape: torch.Size, 
                        k: int, 
                        tau: float) -> Tensor:
        
        device = GraphIMLETopK.device
        sog = list(map(lambda t: Gamma(1.0/k, t/k).sample(shape),
                      torch.arange(1, 10, dtype=torch.float32)))
        sog = torch.sum(torch.stack(sog), dim=0)
        sog = sog - math.log(10.0)
        sog = tau * (sog / k)
        
        return sog.to(device)
    
    @staticmethod
    def sample_k(logits: Tensor, 
                 k: int, 
                 tau: float,
                 train_phase: bool,
                 sample: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        
        if train_phase:
            if sample is None:
                sample = GraphIMLETopK.sample_gumbel_k(logits.shape, k, tau)
            gumbel_softmax_sample = logits + sample
        else:
        	gumbel_softmax_sample = logits
        
        z = torch.zeros(logits.shape[0]).to(GraphIMLETopK.device)
        top_k = torch.topk(gumbel_softmax_sample, k)[1]
        z[top_k] = 1.

        return z, sample

    @staticmethod
    def forward(ctx, logits: Tensor, edge_index: Tensor,
                mask: Tensor, sections: tuple, ratio: float, train_phase: bool):
        
        device = GraphIMLETopK.device
        mask_list = torch.split(mask, split_size_or_sections=sections)
        logits_list = torch.split(logits, split_size_or_sections=sections)
        edge_index_list = torch.split(edge_index, split_size_or_sections=sections, dim=1)
        z = np.array([])
        sample = np.array([])

        # for-loop to sample from each graph[i] in the batch separately
        for i in range(len(mask_list)):
            # get the mask to select one weight per edge
            mask_i = mask_list[i].to(device)
            # logits of the 'directed' graph only
            logits_i = logits_list[i][mask_i].to(device)
            # edge_index list of the 'directed' graph
            directed_edge_index = edge_index_list[i][:, mask_i]
            # select k depending on the number of edges
            num_edges = directed_edge_index.shape[1]
            k_i = int(np.ceil(ratio * num_edges))
            if k_i == 0:
            	k_i = num_edges
            # sample from the logits of the 'directed' graph
            z_tmp, sample_tmp = GraphIMLETopK.sample_k(logits_i, k_i, GraphIMLETopK.tau, train_phase)
            # assign the (discrete) sampling weights to the original graph
            _, z_out = to_undirected(directed_edge_index, z_tmp)
            # append the (discrete) sampling weights to the output tensor
            z = np.append(z, z_out.cpu())
            # assign the SoG values to the original graph
            _, sample_out = to_undirected(directed_edge_index, sample_tmp)
            # append the SoG values to the output tensor
            sample = np.append(sample, sample_out.cpu())

        # convert np.arrays to tensors
        z = torch.tensor(z, dtype=torch.float32).to(device)
        sample = torch.tensor(sample, dtype=torch.float32).to(device)
        # save tensors for backward function
        ctx.ratio = ratio
        ctx.train_phase = train_phase
        ctx.sections = sections
        ctx.mask_list = mask_list
        ctx.logits_list = logits_list
        ctx.edge_index_list = edge_index_list
        ctx.save_for_backward(logits, edge_index, mask, z, sample)
        return z
    
    @staticmethod
    def backward(ctx, dy: Tensor):
        
        logits, edge_index, mask, z, sample = ctx.saved_tensors
        ratio = ctx.ratio
        train_phase = ctx.train_phase
        sections = ctx.sections
        mask_list = ctx.mask_list
        logits_list = ctx.logits_list
        edge_index_list = ctx.edge_index_list
        device = GraphIMLETopK.device
        
        dy_list = torch.split(dy, split_size_or_sections=sections)
        z_list = torch.split(z, split_size_or_sections=sections)
        sample_list = torch.split(sample, split_size_or_sections=sections)
        
        gradient = np.array([])
        for i in range(len(mask_list)):
            # get the mask to select one weight per edge (i.e., directed version of the graph)
            mask_i = mask_list[i].to(device)
            # logits of the 'directed' graph only
            logits_i = logits_list[i][mask_i].to(device)
            # get the gradient of the 'directed' graph
            dy_i = dy_list[i][mask_i].to(device)
            # get SoG values of the 'directed' graph
            sample_i = sample_list[i][mask_i].to(device)
            # get the (discrete) sampling weights of the ORIGINAL graph (NO MASK NEEDED)
            z = z_list[i].to(device)
            # edge_index list of the 'directed' graph
            directed_edge_index = edge_index_list[i][:, mask_i]
            # select k depending on the number of edges
            num_edges = directed_edge_index.shape[1]
            k_i = int(np.ceil(ratio * num_edges))
            if k_i == 0:
            	k_i = num_edges
            # compute target logits of the 'directed' graph
            target_logits = logits_i - (GraphIMLETopK.lambda_ * dy_i)
            # sample from the target_logits of the 'directed' graph
            map_dy_tmp, _ = GraphIMLETopK.sample_k(target_logits, k_i, GraphIMLETopK.tau,
                                              train_phase, sample_i)

            # assign the (discrete) sampling weights to the original graph
            _, map_dy_out = to_undirected(directed_edge_index, map_dy_tmp)
            
            # gradient of the undirected graph
            gradient_out = z - map_dy_out
            # append the graph gradient to output tensor
            gradient = np.append(gradient, gradient_out.cpu())
            
        # convert np.array to tensor
        gradient = torch.tensor(gradient, dtype=torch.float32).to(device)
        return gradient, None, None, None, None, None


class TopKConnectedLayer(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, logits, edge_index, mask, sections, ratio, train_phase):
        return GraphIMLETopKConnected.apply(logits, edge_index, mask, sections, ratio, train_phase)
    
class TopKLayer(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, logits, edge_index, mask, sections, ratio, train_phase):
        return GraphIMLETopK.apply(logits, edge_index, mask, sections, ratio, train_phase)
    
