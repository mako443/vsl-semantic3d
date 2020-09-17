import time
import numpy as np

import torch
import torch.nn
import torch.nn.functional as F
import torchvision.models

import torch_geometric.data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader

'''
TODO: 
-layers different?
-other and/or separate embedding dims?
'''

'''
Network to extract a simple embedding from a graph (normalized to 1), can be used to score the similarity of multiple graphs
'''
class GraphEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(GraphEmbedding, self).__init__()

        self.embedding_dim=embedding_dim

        #Graph layers
        self.conv1 = GCNConv(self.embedding_dim, self.embedding_dim)
        self.conv2 = GCNConv(self.embedding_dim, self.embedding_dim)
        self.conv3 = GCNConv(self.embedding_dim, self.embedding_dim)

        self.node_embedding=torch.nn.Embedding(30, self.embedding_dim) #30 should be enough
        self.node_embedding.requires_grad_(False) #TODO: train embedding?

        #TODO: other layer dimensions, linear layer?

    def forward(self, graphs):
        #x, edges, edge_attr, batch = graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch
        
        x = self.node_embedding(graphs.x) #CARE: is this ok? X seems to be simply stacked
        edges=graphs.edge_index
        edge_attr=graphs.edge_attr
        batch=graphs.batch

        x = self.conv1(x, edges, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edges, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edges, edge_attr)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x= x/torch.norm(x, dim=1,keepdim=True) #Norm output
        
        return x