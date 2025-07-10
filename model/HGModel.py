import dgl
import dgl.nn.pytorch as dglnn
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.gcn_model import GCN, MutualGNN, VanillaGCN
from model.resnet import ResNet, Bottleneck
import math


class Baseline(nn.Module):
    def __init__(self,in_channels,hid_channels,out_channels):
        super().__init__()
        self.poi_encoder = GCN(in_channels,hid_channels,hid_channels)
        self.hrs_encoder = ResNet(Bottleneck,[3,4,6,3])
        self.hrs_encoder.load_state_dict(
            torch.load(r'./ckpt/resnet50-19c8e357.pth')
        )
        self.hrs_encoder.fc = nn.Linear(2048,128)
        self.fc = nn.Linear(2 * hid_channels, out_channels)
    
    def forward(self, g_poi, g_poi_features, img):
      hrs_features = self.hrs_encoder(img)  # 4 x 128
      poi_features = self.poi_encoder(g_poi, g_poi_features)
      with g_poi.local_scope():
          g_poi.ndata['h'] = poi_features
          poi_features = dgl.mean_nodes(g_poi,'h')
      output = torch.concat([hrs_features, poi_features],dim=1)
      output = self.fc(output)
      
      return output


class NGLU_ablation(nn.Module):
    def __init__(self,in_channels,hid_channels,out_channels,args):
        super().__init__()
        self.args = args
        self.poi_encoder = GCN(in_channels,hid_channels,hid_channels)
        self.hrs_encoder = ResNet(Bottleneck, [3, 4, 6, 3])
        self.hrs_encoder.load_state_dict(
            torch.load(r'./ckpt/resnet50-19c8e357.pth')
        )
        self.hrs_encoder.fc = nn.Linear(2048,hid_channels)
        self.hetConv = dglnn.HeteroGraphConv({
            'in': dglnn.GATConv(hid_channels, hid_channels, num_heads=3),
            'with': dglnn.GATConv(hid_channels, hid_channels, num_heads=3),
        }, aggregate='sum')

        self.poi_agg = MutualGNN(hid_channels,hid_channels,hid_channels)
        self.img_agg = MutualGNN(hid_channels,hid_channels,hid_channels)

        self.proj_hrs = nn.Linear(hid_channels, hid_channels)
        self.proj_poi = nn.Linear(hid_channels, hid_channels)

        if self.args.ab_status == 'w_PRC':
            self.fc = nn.Linear(2 * hid_channels,out_channels)
        else:
            self.fc = nn.Linear(4 * hid_channels, out_channels)

        self.dropout = nn.Dropout(p=0.5)

        self.hrs_GNN = VanillaGCN(hid_channels,hid_channels,hid_channels,num_layers=1)
        self.poi_GNN = VanillaGCN(hid_channels,hid_channels,hid_channels,num_layers=1)
        self.beta = nn.Parameter(torch.tensor(0.5)).to('cuda')

        self.sim_list = []

    def getHetGraph(self, g_poi, ori_hrs_features, ori_poi_features):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ori_hrs_features, ori_poi_features = self.proj_hrs(ori_hrs_features), self.proj_poi(ori_poi_features)
        # construct parcel-poi graph
        g_poi_num = g_poi.batch_num_nodes(None)
        offsets = torch.cumsum(torch.cat([torch.zeros(1, dtype=g_poi_num.dtype).to('cuda'), g_poi_num], dim=0), 0)
        offsets = np.array(offsets.cpu().data)
        parcel_poi_src, parcel_poi_dst = [], []
        batch_num = len(ori_hrs_features)
        for i in range(batch_num):
            parcel_poi_src += [i] * g_poi_num[i]
            parcel_poi_dst += [offsets[i] + j for j in range(g_poi_num[i])]
        parcel_poi_src, parcel_poi_dst = np.array(parcel_poi_src), np.array(parcel_poi_dst)
        HGraph = dgl.heterograph({
            ('poi', 'in', 'parcel'): (parcel_poi_dst, parcel_poi_src),
            ('parcel', 'with', 'poi'): (parcel_poi_src, parcel_poi_dst),
        },
            {'parcel': len(ori_hrs_features), 'poi': len(ori_poi_features)}
        ).to(device)

        with HGraph.local_scope():
            HGraph.nodes['parcel'].data['feature'] = ori_hrs_features
            HGraph.nodes['poi'].data['feature'] = ori_poi_features
            node_features = {'poi': ori_poi_features,
                             'parcel': ori_hrs_features}
            output = self.hetConv(HGraph, node_features)
            parcel_output = output['parcel']
            poi_output = output['poi']

            parcel_output = parcel_output.mean(1)
            poi_output = poi_output.mean(1)

        return parcel_output, poi_output
    def UniSimGraph(self, med_output, hrs_features, poi_features,T0):
        # ���������������Թ�ͼ
        batch_parcel_adj = torch.zeros_like(med_output, dtype=torch.int8).to('cuda')
        batch_parcel_adj.fill_diagonal_(1)
        batch_parcel_adj[med_output >= T0] = 1
        batch_parcel_graph = dgl.graph((np.nonzero(batch_parcel_adj)[:, 0], np.nonzero(batch_parcel_adj)[:, 1])).to(
            'cuda')
        batch_parcel_graph.ndata['hrs_feat'] = hrs_features
        batch_parcel_graph.ndata['poi_feat'] = poi_features

        hrs_output = self.hrs_GNN(batch_parcel_graph, batch_parcel_graph.ndata['hrs_feat'])
        poi_output = self.poi_GNN(batch_parcel_graph, batch_parcel_graph.ndata['poi_feat'])

        return hrs_output, poi_output, batch_parcel_graph

    def MutualAgg(self,g_poi,hrs_agg,poi_agg,hrs_features,poi_features):
        poi2img, img2poi = self.getHetGraph(g_poi,hrs_features,poi_features)
        if poi_agg is None:
            g_poi.ndata['h'] = poi_features
            poi_agg = dgl.mean_nodes(g_poi,'h')
        poi2poi = poi_agg

        with g_poi.local_scope():
            g_poi.ndata['h'] = img2poi
            img2poi = dgl.mean_nodes(g_poi,'h')
        ##img2poi = self.leakyRelu(img2poi)
        ##poi2poi = self.leakyRelu(poi2poi)
        agg_poi = torch.concat([img2poi,poi2poi],dim=1)
        # ��hrs�������оۺ�
        img2img = hrs_agg
        ## if g_parcel is None:
        ##     img2img = hrs_features
        ## else:
        ##     img2img = self.img_agg(g_parcel,hrs_features)
        ##poi2img = self.leakyRelu(poi2img)
        ##img2img = self.leakyRelu(img2img)
        agg_img = torch.concat([poi2img,img2img],dim=1)

        agg_output = torch.concat([agg_poi,agg_img],dim=1)
        ## agg_output = self.beta * agg_poi + (1 - self.beta) * agg_img
        
        # ɾ���칹ͼ�ۺ�
        #agg_output = torch.concat([img2img,poi2poi],dim=1)
        # ɾ���ؿ�ͼ�ۺ�
        #agg_output = torch.concat([poi2img,img2poi],dim=1)
        
        return agg_output

    def forward(self, g_poi, g_poi_features, img, T0):
        hrs_features = self.hrs_encoder(img)  # 4 x 128
        poi_features, g_poi = self.poi_encoder(g_poi, g_poi_features)
        ori_hrs_features, ori_poi_features = hrs_features.clone(), poi_features.clone()
        with g_poi.local_scope():
            g_poi.ndata['h'] = poi_features
            poi_features = dgl.mean_nodes(g_poi, 'h')

        med_fused = torch.concat([hrs_features, poi_features], dim=1)
        # ori med_output
        med_fused = med_fused / torch.linalg.norm(med_fused, axis=1, keepdims=True)
        med_output = (torch.matmul(med_fused, med_fused.T) + 1) / 2

        if self.args.ab_status == 'w_PRC':
            output = self.fc(torch.concat([hrs_features,poi_features],dim=1))
            return output, med_output

        else:
            if len(hrs_features) == 1:
                agg_output = self.MutualAgg(g_poi, hrs_features, None, hrs_features, ori_poi_features)
            else:
                hrs_agg, poi_agg, g_parcel = self.UniSimGraph(med_output, hrs_features, poi_features, T0)
                agg_output = self.MutualAgg(g_poi, hrs_agg, poi_agg, hrs_features, ori_poi_features)

            output = self.fc(agg_output)

            return output, med_output



class NGLU(nn.Module):
    def __init__(self,in_channels,hid_channels,out_channels):
        super().__init__()
        self.poi_encoder = GCN(in_channels,hid_channels,hid_channels)
        self.hrs_encoder = ResNet(Bottleneck, [3, 4, 6, 3])
        self.hrs_encoder.load_state_dict(
            torch.load(r'./ckpt/resnet50-19c8e357.pth')
        )
        self.hrs_encoder.fc = nn.Linear(2048,hid_channels)

        self.hetConv = dglnn.HeteroGraphConv({
            'in': dglnn.GATConv(hid_channels, hid_channels, num_heads=3),
            'with': dglnn.GATConv(hid_channels, hid_channels, num_heads=3),
        }, aggregate='sum')

        self.fc = nn.Linear(4 * hid_channels, out_channels)
        self.hrs_GNN = VanillaGCN(hid_channels,hid_channels,hid_channels,num_layers=1)
        self.poi_GNN = VanillaGCN(hid_channels,hid_channels,hid_channels,num_layers=1)

        self.sim_list = []

    def getHetGraph(self, g_poi, ori_hrs_features, ori_poi_features):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ori_hrs_features, ori_poi_features = self.proj_hrs(ori_hrs_features), self.proj_poi(ori_poi_features)
        # construct parcel-poi graph
        g_poi_num = g_poi.batch_num_nodes(None)
        offsets = torch.cumsum(torch.cat([torch.zeros(1, dtype=g_poi_num.dtype).to('cuda'), g_poi_num], dim=0), 0)
        offsets = np.array(offsets.cpu().data)
        parcel_poi_src, parcel_poi_dst = [], []
        batch_num = len(ori_hrs_features)
        for i in range(batch_num):
            parcel_poi_src += [i] * g_poi_num[i]
            parcel_poi_dst += [offsets[i] + j for j in range(g_poi_num[i])]
        parcel_poi_src, parcel_poi_dst = np.array(parcel_poi_src), np.array(parcel_poi_dst)
        HGraph = dgl.heterograph({
            ('poi', 'in', 'parcel'): (parcel_poi_dst, parcel_poi_src),
            ('parcel', 'with', 'poi'): (parcel_poi_src, parcel_poi_dst),
        },
            {'parcel': len(ori_hrs_features), 'poi': len(ori_poi_features)}
        ).to(device)

        with HGraph.local_scope():
            HGraph.nodes['parcel'].data['feature'] = ori_hrs_features
            HGraph.nodes['poi'].data['feature'] = ori_poi_features
            node_features = {'poi': ori_poi_features,
                             'parcel': ori_hrs_features}
            output = self.hetConv(HGraph, node_features)
            parcel_output = output['parcel']
            poi_output = output['poi']

            parcel_output = parcel_output.mean(1)
            poi_output = poi_output.mean(1)

        return parcel_output, poi_output
    def UniSimGraph(self, med_output, hrs_features, poi_features,T0):
        batch_parcel_adj = torch.zeros_like(med_output, dtype=torch.int8).to('cuda')
        batch_parcel_adj.fill_diagonal_(1)
        batch_parcel_adj[med_output >= T0] = 1
        batch_parcel_graph = dgl.graph((np.nonzero(batch_parcel_adj)[:, 0], np.nonzero(batch_parcel_adj)[:, 1])).to(
            'cuda')
        batch_parcel_graph.ndata['hrs_feat'] = hrs_features
        batch_parcel_graph.ndata['poi_feat'] = poi_features

        hrs_output = self.hrs_GNN(batch_parcel_graph, batch_parcel_graph.ndata['hrs_feat'])
        poi_output = self.poi_GNN(batch_parcel_graph, batch_parcel_graph.ndata['poi_feat'])

        return hrs_output, poi_output, batch_parcel_graph

    def MutualAgg(self,g_poi,hrs_agg,poi_agg,hrs_features,poi_features):
        poi2img, img2poi = self.getHetGraph(g_poi,hrs_features,poi_features)
        if poi_agg is None:
            g_poi.ndata['h'] = poi_features
            poi_agg = dgl.mean_nodes(g_poi,'h')
        poi2poi = poi_agg

        with g_poi.local_scope():
            g_poi.ndata['h'] = img2poi
            img2poi = dgl.mean_nodes(g_poi,'h')
        agg_poi = torch.concat([img2poi,poi2poi],dim=1)
        img2img = hrs_agg
        agg_img = torch.concat([poi2img,img2img],dim=1)

        agg_output = torch.concat([agg_poi,agg_img],dim=1)

        return agg_output

    def forward(self, g_poi, g_poi_features, img, T0):
        hrs_features = self.hrs_encoder(img)
        poi_features, g_poi = self.poi_encoder(g_poi, g_poi_features)
        ori_hrs_features, ori_poi_features = hrs_features.clone(), poi_features.clone()
        with g_poi.local_scope():
            g_poi.ndata['h'] = poi_features
            poi_features = dgl.mean_nodes(g_poi, 'h')

        med_fused = torch.concat([hrs_features, poi_features], dim=1)
        med_fused = med_fused / torch.linalg.norm(med_fused, axis=1, keepdims=True)
        med_output = (torch.matmul(med_fused, med_fused.T) + 1) / 2

        
        if len(hrs_features) == 1:
            agg_output = self.MutualAgg(g_poi,hrs_features,None,hrs_features,ori_poi_features)
        else:
            hrs_agg, poi_agg, g_parcel = self.UniSimGraph(med_output,hrs_features,poi_features,T0)
            agg_output = self.MutualAgg(g_poi,hrs_agg,poi_agg,hrs_features,ori_poi_features)

        output = self.fc(agg_output)

        return output, med_output


