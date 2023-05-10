import torch
import torch.nn.functional as F
from torch.nn import Linear
from l2xgnn.graph_imle import GraphIMLETopK, GraphIMLETopKConnected, TopKConnectedLayer, TopKLayer
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import degree
from custom.layers import WeightedSAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        input_dim = dataset.num_features
        output_dim = dataset.num_classes
        
        self.conv1 = SAGEConv(input_dim, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, output_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class L2XGSG(torch.nn.Module):
    # def __init__(self, input_dim, hidden, output_dim, num_layers, connected_flag: bool):
    def __init__(self, dataset, num_layers, hidden, connected_flag: bool):
        super().__init__()
        input_dim = dataset.num_features
        output_dim = dataset.num_classes

        self.conv1 = SAGEConv(input_dim, hidden)
        
        if connected_flag:
            self.l2xgnn = TopKConnectedLayer(GraphIMLETopKConnected.apply)
            print(f'Lambda: {GraphIMLETopKConnected.lambda_}')
        else:
            self.l2xgnn = TopKLayer(GraphIMLETopK.apply)
            print(f'Lambda: {GraphIMLETopK.lambda_}')
            
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(WeightedSAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, output_dim)
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
            
    def forward(self, data, ratio: float, train_phase: bool):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        ################## L2XGNN EDGE SAMPLING ##################
        src, dst = edge_index
        mask = src < dst
        edge_scores = (x[src] * x[dst]).sum(dim=-1)
        sections = degree(batch[src], dtype=torch.long).tolist()
        sampled_edges = self.l2xgnn(edge_scores, edge_index, mask, sections, ratio, train_phase)
        edge_weights = edge_scores * sampled_edges
        #################                     ##################
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weights))
 
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), sampled_edges, sections
    
    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
    def __repr__(self):
        return self.__class__.__name__
    

