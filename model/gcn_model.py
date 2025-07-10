import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import AvgPooling
import dgl
import logging

class MutualGNN(nn.Module):
    def __init__(self,in_channels,hid_channels,out_channels):
        super(MutualGNN,self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels

        self.WC = nn.Linear(in_channels,hid_channels,bias=False)
        self.WD = nn.Linear(in_channels,hid_channels,bias=False)
        self.proj = nn.Linear(2*hid_channels,out_channels)

    def forward(self,graph,feat):
        # feat: N x 256
        # 对于X_I->P
        # 共性信息聚合
        # Apply linear transformations
        z_hat_com = self.WC(feat)
        z_hat_dis = self.WD(feat)
        # Common feature aggregation
        graph.ndata['z'] = z_hat_com
        graph.apply_edges(lambda edges: {'com_score': (edges.src['z'] * edges.dst['z']).sum(-1)})
        graph.edata['alpha_com'] = F.softmax(graph.edata['com_score'], dim=0)
        graph.update_all(dgl.function.u_mul_e('z', 'alpha_com', 'm'), dgl.function.sum('m', 'z_com'))

        # Distinct feature aggregation
        graph.ndata['z'] = z_hat_dis
        graph.apply_edges(lambda edges: {'dis_score': (edges.src['z'] - edges.dst['z']).pow(2).sum(-1, keepdim=True)})
        graph.edata['alpha_dis'] = F.softmax(graph.edata['dis_score'], dim=0)
        graph.update_all(dgl.function.u_mul_e('z', 'alpha_dis', 'm'), dgl.function.sum('m', 'z_dis'))

        # Combine and project the features
        z_com = graph.ndata['z_com']
        z_dis = graph.ndata['z_dis']
        output = self.proj(torch.cat([z_com, z_dis], dim=1))

        return output

class GCN(nn.Module):
    def __init__(self,in_channels,hid_channels,out_channels,poi_adj=None,num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.proj = nn.Linear(in_channels,hid_channels)
        for layer in range(self.num_layers):
            if layer == 0:
                self.layers.append(GraphConv(in_channels,hid_channels))
                self.batch_norms.append(nn.BatchNorm1d(hid_channels))
            else:
                self.layers.append(GraphConv(hid_channels,hid_channels))
                self.batch_norms.append(nn.BatchNorm1d(hid_channels))

        self.fc = nn.Linear(hid_channels,out_channels)
        self.drop = nn.Dropout(p=0.5)
        self.pooling = (AvgPooling())

        self.poi_adj = poi_adj
        self.weight_dict = {
            0: 0.7,
            1: 0.9,
            2: 0.7,
            3: 0.9,
            4: 0.3,
            5: 0.7,
            6: 0.3,
            7: 0.9,
            8: 0.3,
            9: 0.3,
            10: 0.9,
            11: 0.7,
            12: 0.1,
            13: 0.9,
            14: 0.5,
            15: 0.9,
            16: 0.5,
            17: 0.5,
            18: 0.1,
            19: 0.3,
            20: 0.7,
            21: 0.9,
            22: 0.9,
            23: 0.9,
            24: 0.9,
            25: 0.9,
        }

        
    def forward(self,g,h):
        sampled_graph = g
        # 初始节点特征加权
        node_cats = torch.argmax(h,dim=1)
        node_weights = torch.tensor([self.weight_dict[cat.item()] for cat in node_cats],device=h.device)
        for i, layer in enumerate(self.layers):
            h = layer(g,h)
            if len(h) != 1:
                h = self.batch_norms[i](h)
            h = F.relu(h)

        h = self.fc(h)
        sampled_graph.ndata['w'] = node_weights.unsqueeze(1)
      
        return h, sampled_graph

class VanillaGCN(nn.Module):
    def __init__(self,in_channels,hid_channels,out_channels,poi_adj=None,num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.proj = nn.Linear(in_channels,hid_channels)
        for layer in range(self.num_layers):
            if layer == 0:
                self.layers.append(GraphConv(in_channels,hid_channels)) # 度采样
                self.batch_norms.append(nn.BatchNorm1d(hid_channels))
            else:
                self.layers.append(GraphConv(hid_channels,hid_channels))
                self.batch_norms.append(nn.BatchNorm1d(hid_channels))

        self.fc = nn.Linear(hid_channels,out_channels)
        self.drop = nn.Dropout(p=0.5)
        self.pooling = (AvgPooling())

        self.poi_adj = poi_adj


    def forward(self,g,h):
        for i, layer in enumerate(self.layers):
            h = layer(g,h)
            h = self.batch_norms[i](h)
            h = F.relu(h)

        h = self.fc(h)

        return h

