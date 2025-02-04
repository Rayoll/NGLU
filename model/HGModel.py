import dgl
import dgl.nn.pytorch as dglnn
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.gcn_model import WA_GCN, GCN, GATV2, WA_GAT, WA_GIN, GAT, APPNP, SAGE, GCN_NoisyView, MutualGNN, VanillaGCN,\
VanillaEncoder
from dgl.nn.pytorch import GATConv, GraphConv, GATv2Conv
from model.resnet import ResNet, Bottleneck
import math

# ��ע����ģ��
class selfAttnLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads,):
        super().__init__()

        self.Wk = nn.Linear(in_channels, out_channels*num_heads)
        self.Wq = nn.Linear(in_channels, out_channels*num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads

    def attention_conv(self,q,k,v):
        # normalization
        q = q / torch.norm(q, p=2)
        k = k / torch.norm(k, p=2)
        N = q.shape[0]
        # numerator
        kv = torch.einsum('lhm,lhd->hmd',k,v)
        attention_num = torch.einsum('nhm,hmd->nhd',q,kv)
        attention_num += N * v
        # den
        all_ones = torch.ones([k.shape[0]]).to(k.device)
        k_sum = torch.einsum('lhm,l->hm',k,all_ones)
        attention_normalizer = torch.einsum('nhm,hm->nh',q,k_sum)
        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer,len(attention_normalizer.shape)
        )
        attention_normalizer += torch.ones_like(attention_normalizer)
        attn_output = attention_num / attention_normalizer

        return attn_output


    def forward(self,query,source):
        query = self.Wq(query).reshape(-1,self.num_heads,self.out_channels)
        key = self.Wk(source).reshape(-1,self.num_heads,self.out_channels)
        value = source.reshape(-1,1,self.out_channels)
        # compute full attentive aggregation
        attn_output = self.attention_conv(query,key,value)
        attn_output = attn_output.mean(dim=1)

        return attn_output

class MSA(nn.Module):
    def __init__(self,num_heads,hidden_size):
        super().__init__()
        self.num_attention_heads = num_heads # 8
        self.attention_head_size = int(hidden_size/num_heads) # 32
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 8 x 32

        self.query = nn.Linear(hidden_size,self.all_head_size)
        self.key = nn.Linear(hidden_size,self.all_head_size)
        self.value = nn.Linear(hidden_size,self.all_head_size)

        self.dense = nn.Linear(hidden_size,hidden_size)

    def transpose_for_scores(self,x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self,x):
        mixed_query_layer = self.query(x)  # bs x hidden_size
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # bs x num_heads x hidden_size
        key_layer = self.transpose_for_scores(mixed_key_layer)  # bs x num_heads x hidden_size
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # bs x num_heads x hidden_size

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
                                                                         -2))  # bs x num_heads x hidden_size
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)  # bs x num_heads x hidden_size
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # bs x num_heads x hidden_size
        context_layer = torch.matmul(attention_probs,
                                     value_layer)  # bs x num_heads x hidden_size
        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
        self.all_head_size,)  # bs x num_heads x hidden_size
        context_layer = context_layer.view(*new_context_layer_shape)  # bs x num_heads x hidden_size

        output = self.dense(context_layer)

        return output
class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)

class GraphBased_selfAttnLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels,
                 ):
        super().__init__()

        self.Wk = nn.Linear(in_channels, hid_channels)
        self.Wq = nn.Linear(in_channels, hid_channels)
        self.Wv = nn.Linear(in_channels, hid_channels)

        # self.graphConv = dglnn.GraphConv(hid_channels,hid_channels)
        self.graphConv = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.mlp = MLP(hid_channels,hid_channels//2,hid_channels)

        self.relu = nn.ReLU(inplace=True)

        self.ln1 = nn.LayerNorm(hid_channels)
        self.ln2 = nn.LayerNorm(hid_channels)


    def attention_graph_conv(self,q,k,v):
        # normalization
        q = q / torch.norm(q,p=2)
        k = k / torch.norm(k,p=2)
        N = q.shape[0]

        A = torch.matmul(q,k.T)
        # ����ȫ��ͨͼ���൱��ֱ����ͼ�Ͻ��о�����㣩
        A = A.unsqueeze(0)
        agg_A = self.graphConv(A)
        agg_A = agg_A.squeeze()
        # �����ۺ�
        attn_output = torch.matmul(agg_A,v)
        return attn_output

    def forward(self,x):
        query = self.Wq(x)
        key = self.Wk(x)
        value = self.Wv(x)
        attn_output = self.attention_graph_conv(query,key,value)
        x = x + self.ln1(attn_output)
        x = x + self.mlp(self.ln2(x))

        return x

class SimpleGlobalAttention(nn.Module):
    def __init__(self,in_channels,hidden_channels,num_layers=2):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels,hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))

        self.relu = nn.ReLU(inplace=True)

        for _ in range(num_layers):
            self.convs.append(
                GraphBased_selfAttnLayer(hidden_channels,hidden_channels)
            )
            self.bns.append(nn.LayerNorm(hidden_channels))

    def forward(self,x):
        x = self.fcs[0](x)
        x = self.bns[0](x)
        x = self.relu(x)
        x = F.dropout(x,p=0.5)

        layer_ = []
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = 0.5 * x + (1-0.5)*layer_[i]
            x = self.bns[i+1](x)
            x = self.relu(x)
            x = F.dropout(x,p=0.5)
            layer_.append(x)

        return x

class BiLSTM(nn.Module):
    def __init__(self,lstm_hidden_dim,lstm_num_layers,embed_num,output_dim):
        super().__init__()

        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers

        # self.embed_num = embed_num
        # self.embed = nn.Embedding(embed_num,self.hidden_dim)
        self.bilstm = nn.LSTM(1,self.hidden_dim//2,num_layers=1,batch_first=True,bidirectional=True,bias=False)
        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, output_dim)

    def forward(self,x):
        # x = x.view(len(x),x.size(1),-1)
        # embed = self.embed(x)
        # x = embed.view(len(x),embed.size(1),-1)
        x = x.unsqueeze(2)
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = bilstm_out[:,-1,:] # ֻ�������һ�������Ľ��
        #print('bilstm_out:',bilstm_out.shape)
        y = self.hidden2label1(bilstm_out)
        y = self.hidden2label2(y)
        logit = y

        return logit


class HetGeoAware(nn.Module):
    def __init__(self,in_channels,hid_channels,out_channels,parcel_adj,poi_adj):
        super().__init__()
        self.poi_encoder = WA_GCN(in_channels,64,hid_channels,init_weight_fn='/data1/yry23/codes_dir/codes_v2/data/poi_lu_init_weight.npy')
        #self.poi_encoder = GCN(in_channels,hid_channels,hid_channels)
        self.hrs_encoder = ResNet(Bottleneck,[3,4,6,3])
        self.hrs_encoder.load_state_dict(
            torch.load(r'/data1/yry23/checkpoint/resnet50-19c8e357.pth')
        )
        self.hrs_encoder.fc = nn.Linear(2048,128)
        self.hrs_fc = nn.Linear(hid_channels,out_channels)
        self.poi_fc = nn.Linear(hid_channels,out_channels)

        self.hetConv = dglnn.HeteroGraphConv({
            'in':dglnn.GATConv(hid_channels,hid_channels,num_heads=3),
            'with': dglnn.GATConv(hid_channels, hid_channels, num_heads=3),
            #'in': dglnn.SAGEConv(hid_channels,hid_channels,'lstm'),
            #'with': dglnn.SAGEConv(hid_channels,hid_channels,'lstm'),
            #'connect':dglnn.GATConv(hid_channels,hid_channels,num_heads=3),
            #'adjacent':dglnn.GATConv(hid_channels,hid_channels,num_heads=3),
            #'connect':dglnn.GraphConv(hid_channels,hid_channels),
            #'adjacent': dglnn.GraphConv(hid_channels, hid_channels),
            #'in':dglnn.GraphConv(hid_channels,hid_channels),
            #'with': dglnn.GraphConv(hid_channels, hid_channels),
        },aggregate='sum')
        self.homoConv = dglnn.HeteroGraphConv({
            'connect':dglnn.GraphConv(hid_channels,hid_channels),
            'adjacent':dglnn.GraphConv(hid_channels,hid_channels),
        },aggregate='mean')
        
        self.proj_hrs = nn.Linear(hid_channels,hid_channels)
        self.proj_poi = nn.Linear(hid_channels,hid_channels)
        self.gHRSConv = dglnn.GraphConv(hid_channels,hid_channels)
        self.gPOIConv = dglnn.GraphConv(hid_channels,hid_channels)

        self.fc = nn.Linear(2*hid_channels,out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bilstm = BiLSTM(2*hid_channels,1,3,out_channels)

        self.parcel_adj = parcel_adj
        self.poi_adj = poi_adj
        
        #self.hrs_attn_conv = selfAttnLayer(hid_channels,hid_channels,num_heads=3)
        #self.poi_attn_conv = selfAttnLayer(hid_channels,hid_channels,num_heads=3)
        
        self.poiRelation = GraphBased_selfAttnLayer(hid_channels,hid_channels)
        self.hrsRelation = GraphBased_selfAttnLayer(hid_channels,hid_channels)
        
        self.fusedRelation = GraphBased_selfAttnLayer(2*hid_channels,2*hid_channels)

    def geo_aware_aggregation(self,hrs_features,poi_features,graphID,poiID):
        # parcel adjacency
        graphID -= 1
        batch_parcel_adj = self.parcel_adj[graphID]
        batch_parcel_adj = batch_parcel_adj[:,graphID]
        batch_parcel_adj += torch.eye(batch_parcel_adj.size(0)).to('cuda')
        batch_parcel_adj = batch_parcel_adj.to('cuda')

        ori_hrs_features, ori_poi_features = hrs_features.clone(), poi_features.clone()

        # ����ͼ�ṹ
        # batch_parcel_graph = dgl.graph((torch.nonzero(batch_parcel_adj)[:,0],np.nonzero(batch_parcel_adj)[:,1])).to('cuda')
        # batch_parcel_graph.ndata['feat'] = hrs_features
        # �����Ӿ���ʾ�����Թ���HRSͼ
        hrs_features = hrs_features / torch.linalg.norm(hrs_features,axis=1,keepdims=True)
        cos_hrs_mtx = torch.matmul(hrs_features,hrs_features.T)
        batch_parcel_adj = torch.zeros_like(cos_hrs_mtx,dtype=torch.int8).to('cuda')
        batch_parcel_adj[cos_hrs_mtx > 0.8] = 1
        batch_parcel_graph = dgl.graph((np.nonzero(batch_parcel_adj)[:,0],np.nonzero(batch_parcel_adj)[:,1])).to('cuda')
        batch_parcel_graph.ndata['feat'] = ori_hrs_features
        
        # ���ݽڵ��ʾ�����Թ���POIͼ
        # batch_poi_graph = dgl.graph((torch.nonzero(batch_parcel_adj)[:,0],np.nonzero(batch_parcel_adj)[:,1])).to('cuda')
        # batch_poi_graph.ndata['feat'] = proj_poi
        poi_features = poi_features / torch.linalg.norm(poi_features,axis=1,keepdims=True)
        cos_mtx = torch.matmul(poi_features,poi_features.T)
        batch_poi_adj = torch.zeros_like(cos_mtx,dtype=torch.int8).to('cuda')
        batch_poi_adj[cos_mtx > 0.8] = 1
        batch_poi_graph = dgl.graph((np.nonzero(batch_poi_adj)[:,0],np.nonzero(batch_poi_adj)[:,1])).to('cuda')
        batch_poi_graph.ndata['feat'] = ori_poi_features

        # �����ۺ�
        agg_hrs_features = self.relu(self.gHRSConv(batch_parcel_graph,batch_parcel_graph.ndata['feat']))
        agg_poi_features = self.relu(self.gPOIConv(batch_poi_graph,batch_poi_graph.ndata['feat']))


        return agg_hrs_features, agg_poi_features
        
    def getHetGraph(self,g_poi,ori_hrs_features,ori_poi_features):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ori_hrs_features, ori_poi_features = self.proj_hrs(ori_hrs_features), self.proj_poi(ori_poi_features) # projection of heterogeneous
        # construct parcel-poi graph
        g_poi_num = g_poi.batch_num_nodes(None)
        offsets = torch.cumsum(torch.cat([torch.zeros(1,dtype=g_poi_num.dtype).to('cuda'),g_poi_num],dim=0),0)
        offsets = np.array(offsets.cpu().data)
        parcel_poi_src, parcel_poi_dst = [], []
        batch_num = len(ori_hrs_features)
        for i in range(batch_num):
            parcel_poi_src += [i] * g_poi_num[i]
            parcel_poi_dst += [offsets[i]+j for j in range(g_poi_num[i])]
        parcel_poi_src, parcel_poi_dst = np.array(parcel_poi_src), np.array(parcel_poi_dst)
        HGraph = dgl.heterograph({
            ('poi','in','parcel'):(parcel_poi_dst,parcel_poi_src),
            ('parcel','with','poi'):(parcel_poi_src,parcel_poi_dst),
        },
            {'parcel':len(ori_hrs_features),'poi':len(ori_poi_features)}
        ).to(device)

        with HGraph.local_scope():
            HGraph.nodes['parcel'].data['feature'] = ori_hrs_features
            HGraph.nodes['poi'].data['feature'] = ori_poi_features
            node_features = {'poi':ori_poi_features,
                             'parcel':ori_hrs_features}
            output = self.hetConv(HGraph,node_features)
            # output = self.hetSANN(HGraph,node_features,HGraph.ntypes,HGraph.etypes)
            parcel_output = output['parcel']
            poi_output = output['poi']
            
            parcel_output = parcel_output.mean(1)
            poi_output = poi_output.mean(1)

        return parcel_output, poi_output

    def attention_aware_aggregation(self,hrs_features,poi_features):
        # update_hrs_features = self.hrs_attn_conv(hrs_features,hrs_features)
        # update_poi_features = self.poi_attn_conv(poi_features,poi_features)

        update_hrs_features = self.hrsEA(hrs_features)
        update_poi_features = self.poiEA(poi_features)

        return update_hrs_features, update_poi_features
        
    def intraCorrelation_aggregation(self,hrs_features,poi_features):
        update_hrs_features = self.hrsRelation(hrs_features)
        update_poi_features = self.poiRelation(poi_features)

        return update_hrs_features, update_poi_features

    def getHetGraphV2(self,g_poi,ori_hrs_features,ori_poi_features):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ori_hrs_features, ori_poi_features = self.proj_hrs(ori_hrs_features), self.proj_poi(ori_poi_features)
        # construct parcel-poi graph
        g_poi_num = g_poi.batch_num_nodes(None)
        offsets = torch.cumsum(torch.cat([torch.zeros(1,dtype=g_poi_num.dtype).to('cuda'),g_poi_num],dim=0),0)
        offsets = np.array(offsets.cpu().data)
        parcel_poi_src, parcel_poi_dst = [], []
        batch_num = len(ori_hrs_features)
        for i in range(batch_num):
            parcel_poi_src += [i] * g_poi_num[i]
            parcel_poi_dst += [offsets[i]+j for j in range(g_poi_num[i])]
        parcel_poi_src, parcel_poi_dst = np.array(parcel_poi_src), np.array(parcel_poi_dst)
        # �����Ӿ���ʾ�����Թ���HRSͼ
        hrs_features = ori_hrs_features / torch.linalg.norm(ori_hrs_features,axis=1,keepdims=True)
        cos_hrs_mtx = torch.matmul(hrs_features,hrs_features.T)
        batch_parcel_adj = torch.zeros_like(cos_hrs_mtx,dtype=torch.int8).to('cuda')
        batch_parcel_adj[cos_hrs_mtx > 0.9] = 1
        batch_parcel_graph = dgl.graph((np.nonzero(batch_parcel_adj)[:,0],np.nonzero(batch_parcel_adj)[:,1])).to('cuda')
        batch_parcel_graph.ndata['feat'] = ori_hrs_features
        parcel_src, parcel_dst = torch.where(cos_hrs_mtx == 1)
        parcel_src = np.array(parcel_src.cpu().data)
        parcel_dst = np.array(parcel_dst.cpu().data)
        poi_features = ori_poi_features / torch.linalg.norm(ori_poi_features,axis=1,keepdims=True)
        cos_mtx = torch.matmul(poi_features,poi_features.T)
        batch_poi_adj = torch.zeros_like(cos_mtx,dtype=torch.int8).to('cuda')
        batch_poi_adj[cos_mtx > 0.9] = 1
        batch_poi_graph = dgl.graph((np.nonzero(batch_poi_adj)[:,0],np.nonzero(batch_poi_adj)[:,1])).to('cuda')
        batch_poi_graph.ndata['feat'] = ori_poi_features
        poi_src, poi_dst = torch.where(cos_mtx == 1)
        poi_src = np.array(poi_src.cpu().data)
        poi_dst = np.array(poi_dst.cpu().data)

        HGraph = dgl.heterograph({
            ('poi','in','parcel'):(parcel_poi_dst,parcel_poi_src),
            ('parcel','with','poi'):(parcel_poi_src,parcel_poi_dst),
        },{'parcel':len(ori_hrs_features),'poi':len(ori_poi_features)}
        ).to(device)
        
        homoGraph = dgl.heterograph({
            ('poi','connect','poi'):(poi_src,poi_dst),
            ('parcel','adjacent','parcel'):(parcel_src,parcel_dst),
        }, {'parcel':len(ori_hrs_features),'poi':len(ori_poi_features)}
        ).to(device)
        
        with HGraph.local_scope():
            HGraph.nodes['parcel'].data['feature'] = ori_hrs_features
            HGraph.nodes['poi'].data['feature'] = ori_poi_features
            node_features = {'poi':ori_poi_features,
                             'parcel':ori_hrs_features}
            output = self.hetConv(HGraph,node_features)
        with homoGraph.local_scope():
            updated_features = {'poi':output['poi'].mean(1),
                                'parcel':output['parcel'].mean(1)}
            output = self.homoConv(homoGraph,updated_features)
            # output = self.hetSANN(HGraph,node_features,HGraph.ntypes,HGraph.etypes)
            parcel_output = output['parcel']
            poi_output = output['poi']

        return parcel_output, poi_output
        
    def forward(self,g_poi,g_poi_features,img,graphID,poiID):
        hrs_features = self.hrs_encoder(img) # 4 x 128
        hrs_output = self.hrs_fc(hrs_features)
        poi_features = self.poi_encoder(g_poi,g_poi_features)
        ori_hrs_features, ori_poi_features = hrs_features.clone(), poi_features.clone()
        #agg_het_parcel_features = self.getHetGraph(g_poi,hrs_features,poi_features)
        with g_poi.local_scope():
            g_poi.ndata['h'] = poi_features
            poi_features = dgl.mean_nodes(g_poi,'h')
        #output = torch.concat([hrs_features,poi_features],dim=1)
        #output = self.fc(output)
        poi_output = self.poi_fc(poi_features)
        
        ## tmp
        #with g_poi.local_scope():
        #    g_poi.ndata['h'] = poi_features
        #    poi_features = dgl.max_nodes(g_poi,'h')
        
        # geographic-aware aggregation
        ##agg_hrs_features, agg_poi_features = self.geo_aware_aggregation(hrs_features,ori_poi_features,graphID,poiID)
        #agg_hrs_features, agg_poi_features = self.intraCorrelation_aggregation(hrs_features,ori_poi_features)
        ##agg_hrs_features = self.proj_hrs(agg_hrs_features)
        ##agg_poi_features = self.proj_poi(agg_poi_features)
        #agg_het_parcel_features, agg_het_poi_features = self.getHetGraphV2(g_poi,hrs_features,ori_poi_features)
        agg_het_parcel_features, agg_het_poi_features = self.getHetGraph(g_poi,hrs_features,ori_poi_features)
        #agg_het_parcel_features, agg_het_poi_features = self.getHetGraphV2(g_poi,ori_hrs_features,ori_poi_features)
        ##agg_hrs_features, agg_poi_features = self.intraCorrelation_aggregation(agg_het_parcel_features,agg_het_poi_features)
        ## heterogeneous graph construction
        ##agg_het_parcel_features = self.getHetGraph(g_poi,agg_hrs_features,agg_poi_features)
        #with g_poi.local_scope():
        #    g_poi.ndata['h'] = agg_poi_features
        #    agg_poi_features = dgl.mean_nodes(g_poi,'h')
        with g_poi.local_scope():
            g_poi.ndata['h'] = agg_het_poi_features
            agg_het_poi_features = dgl.mean_nodes(g_poi,'h')
        ##output = self.fc(agg_het_parcel_features)
        #output = torch.concat([agg_hrs_features,agg_poi_features,agg_het_parcel_features,agg_het_poi_features],dim=1)
        #final_parcel_features = agg_hrs_features + agg_het_parcel_features
        #final_poi_features = agg_poi_features + agg_het_poi_features
        #with g_poi.local_scope():
        #    g_poi.ndata['h'] = final_poi_features
        #    final_poi_features = dgl.mean_nodes(g_poi,'h')
        #output = torch.concat([final_parcel_features,final_poi_features],dim=1)
        #print(agg_het_parcel_features.shape)
        #print(agg_het_poi_features.shape)
        output = torch.concat([agg_het_parcel_features,agg_het_poi_features],dim=1)
        #output = self.bilstm(output)
        ##output = agg_het_parcel_features + agg_hrs_features

        #output = self.fusedRelation(output)
        output = self.fc(output)
        
        return output, hrs_output, poi_output


class Baseline(nn.Module):
    def __init__(self,in_channels,hid_channels,out_channels,parcel_adj,poi_adj):
        super().__init__()
        self.poi_encoder = GCN(in_channels,hid_channels,hid_channels)
        #self.poi_encoder = GCN_NoisyView(in_channels,hid_channels,hid_channels)
        self.hrs_encoder = ResNet(Bottleneck,[3,4,6,3])
        self.hrs_encoder.load_state_dict(
            torch.load(r'/data1/yry23/checkpoint/resnet50-19c8e357.pth')
        )
        self.hrs_encoder.fc = nn.Linear(2048,128)
        self.fc = nn.Linear(2 * hid_channels, out_channels)
    
    def forward(self, g_poi, g_poi_features, img, epoch,labels=None, state='train'):
      hrs_features = self.hrs_encoder(img)  # 4 x 128
      #poi_features, poi_logits = self.poi_encoder(g_poi, g_poi_features)
      poi_features = self.poi_encoder(g_poi, g_poi_features)
      with g_poi.local_scope():
          g_poi.ndata['h'] = poi_features
          poi_features = dgl.mean_nodes(g_poi,'h')
      output = torch.concat([hrs_features, poi_features],dim=1)
      output = self.fc(output)
      
      
      return output
      #return output, poi_logits
        
        

class CLHI(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, parcel_adj, poi_adj):
        super().__init__()
        # self.poi_encoder = WA_GCN(in_channels, hid_channels, hid_channels,
        #                           init_weight_fn='/data1/yry23/codes_dir/codes_v2/data/poi_lu_init_weight.npy')
        self.poi_encoder = GCN(in_channels,hid_channels,hid_channels)
        self.hrs_encoder = ResNet(Bottleneck, [3, 4, 6, 3])
        self.hrs_encoder.load_state_dict(
            torch.load(r'/data1/yry23/checkpoint/resnet50-19c8e357.pth')
        )
        self.hrs_encoder.fc = nn.Linear(2048, 128)
        self.hetConv = dglnn.HeteroGraphConv({
            'in': dglnn.GATConv(hid_channels, hid_channels, num_heads=3),
            'with': dglnn.GATConv(hid_channels, hid_channels, num_heads=3),
        }, aggregate='sum')

        self.proj_hrs = nn.Linear(hid_channels, hid_channels)
        self.proj_poi = nn.Linear(hid_channels, hid_channels)

        self.fc = nn.Linear(2 * hid_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.hrs_GNN = GCN(hid_channels,hid_channels,hid_channels,num_layers=1)
        self.poi_GNN = GCN(hid_channels,hid_channels,hid_channels,num_layers=1)
        #self.pos_GNN = GCN(2*hid_channels,2*hid_channels,2*hid_channels,num_layers=1)
        self.pos_GNN = GAT(2*hid_channels,2*hid_channels,2*hid_channels,num_layers=1)
        self.sim_list = []
        #self.hrs_GNN = GAT(hid_channels, hid_channels, hid_channels, num_layers=1)
        #self.poi_GNN = GAT(hid_channels, hid_channels, hid_channels, num_layers=1)


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
            # output = self.hetSANN(HGraph,node_features,HGraph.ntypes,HGraph.etypes)
            parcel_output = output['parcel']
            poi_output = output['poi']

            parcel_output = parcel_output.mean(1)
            poi_output = poi_output.mean(1)

        return parcel_output, poi_output


    def UniSimGraph(self, med_output, hrs_features, poi_features,T0):
        # ���������������Թ�ͼ
        # med_output = med_output / torch.linalg.norm(med_output,axis=1,keepdims=True)
        # med_output = (torch.matmul(med_output,med_output.T) + 1) / 2
        batch_parcel_adj = torch.zeros_like(med_output, dtype=torch.int8).to('cuda')
        batch_parcel_adj[med_output >= T0] = 1
        batch_parcel_graph = dgl.graph((np.nonzero(batch_parcel_adj)[:, 0], np.nonzero(batch_parcel_adj)[:, 1])).to(
            'cuda')
        batch_parcel_graph.ndata['hrs_feat'] = hrs_features
        batch_parcel_graph.ndata['poi_feat'] = poi_features

        hrs_output = self.hrs_GNN(batch_parcel_graph, batch_parcel_graph.ndata['hrs_feat'])
        poi_output = self.poi_GNN(batch_parcel_graph, batch_parcel_graph.ndata['poi_feat'])

        return hrs_output, poi_output
    def posSimGraph(self,pos,parcel_features):
        pos = pos / torch.linalg.norm(pos,axis=1,keepdims=True)
        #pos = (torch.matmul(pos,pos.T) + 1) / 2
        pos = torch.matmul(pos,pos.T)
        neighbor_adj = torch.zeros_like(pos,dtype=torch.int8).to('cuda')
        neighbor_adj[pos >= 0.9999] = 1
        #a = neighbor_adj[neighbor_adj==1]
        #print('pos_num:',len(a))
        neighbor_graph = dgl.graph((np.nonzero(neighbor_adj)[:,0],np.nonzero(neighbor_adj)[:,1])).to('cuda')
        neighbor_graph.ndata['features'] = parcel_features
        parcel_updated_features = self.pos_GNN(neighbor_graph,neighbor_graph.ndata['features'])

        return parcel_updated_features

    def forward(self, g_poi, g_poi_features, img, epoch, T0, pos=None,labels=None, state='train'):
        hrs_features = self.hrs_encoder(img)  # 4 x 128
        poi_features = self.poi_encoder(g_poi, g_poi_features)
        ori_hrs_features, ori_poi_features = hrs_features.clone(), poi_features.clone()
        with g_poi.local_scope():
            g_poi.ndata['h'] = poi_features
            poi_features = dgl.mean_nodes(g_poi, 'h')
        med_fused = torch.concat([hrs_features, poi_features], dim=1)
        med_fused = med_fused / torch.linalg.norm(med_fused, axis=1, keepdims=True)
        med_output = (torch.matmul(med_fused, med_fused.T) + 1) / 2  # ���������Լ���

        ## ����norm features��ͼ
        #if len(hrs_features) == 1:
        #   agg_hrs_features, agg_poi_features = hrs_features, poi_features
        #else:
        #   agg_hrs_features, agg_poi_features = self.UniSimGraph(med_output,hrs_features,poi_features,T0)

        #agg_het_parcel_features, agg_het_poi_features = self.getHetGraph(g_poi, hrs_features, ori_poi_features)
        #with g_poi.local_scope():
        #   g_poi.ndata['h'] = agg_het_poi_features
        #   agg_het_poi_features = dgl.mean_nodes(g_poi, 'h')
        output = torch.concat([hrs_features,poi_features],dim=1)
        #output = torch.concat([agg_hrs_features, agg_poi_features],dim=1)
        output = self.posSimGraph(pos,output)
        output = self.fc(output)

        return output, med_output
        
class MemoryReadout(nn.Module):
    def __init__(self, in_channels, attn_channels):
        super(MemoryReadout, self).__init__()
        self.attn_fc = nn.Linear(in_channels, attn_channels, bias=False)
        # self.memory_bank = nn.Linear(in_channels,attn_channels) # 128 x 17
        self.memory_bank = nn.Parameter(torch.ones((attn_channels,1)))
        # memory_bank��ʼ��
        # self.memory_bank = nn.Linear(attn_channels, 17, bias=False)
        # self.memory_bank_reverse = nn.Linear(17, in_channels, bias=False)

    def forward(self, graph, feat):
        with graph.local_scope():
            attn_base = self.attn_fc(feat)
            # attn_score = self.memory_bank(attn_base) # N x 17
            # attn_weights, _ = torch.max(F.softmax(attn_score,dim=1),dim=1,keepdim=True) # N x 1
            attn_score = torch.matmul(attn_base,self.memory_bank)
            attn_weights = F.sigmoid(attn_score)
            graph.ndata['h'] = feat * attn_weights
            output = dgl.sum_nodes(graph, 'h')

            return output

class NGLU(nn.Module):
    def __init__(self,in_channels,hid_channels,out_channels):
        super().__init__()
        self.poi_encoder = GCN(in_channels,hid_channels,hid_channels)
        self.hrs_encoder = ResNet(Bottleneck, [3, 4, 6, 3])
        self.hrs_encoder.load_state_dict(
            torch.load(r'/data1/yry23/checkpoint/resnet50-19c8e357.pth')
        )
        self.hrs_encoder.fc = nn.Linear(2048,hid_channels)
        self.WP = nn.Linear(2*hid_channels,2*hid_channels,bias=False)
        
        self.wI2P, self.wI2I, self.wP2I, self.wP2P = nn.Linear(hid_channels,hid_channels,bias=False),\
        nn.Linear(hid_channels,hid_channels,bias=False),nn.Linear(hid_channels,hid_channels,bias=False), \
                                                     nn.Linear(hid_channels, hid_channels, bias=False)

        self.hetConv = dglnn.HeteroGraphConv({
            'in': dglnn.GATConv(hid_channels, hid_channels, num_heads=3),
            'with': dglnn.GATConv(hid_channels, hid_channels, num_heads=3),
        }, aggregate='sum')

        self.poi_agg = MutualGNN(hid_channels,hid_channels,hid_channels)
        self.img_agg = MutualGNN(hid_channels,hid_channels,hid_channels)

        self.proj_hrs = nn.Linear(hid_channels, hid_channels)
        self.proj_poi = nn.Linear(hid_channels, hid_channels)

        self.fc = nn.Linear(4 * hid_channels, out_channels)
        self.fc2 = nn.Linear(2 * hid_channels, out_channels)
        #self.fc = nn.Linear(768, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.leakyRelu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.poi_readout = MemoryReadout(hid_channels,attn_channels=64)

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
            # output = self.hetSANN(HGraph,node_features,HGraph.ntypes,HGraph.etypes)
            parcel_output = output['parcel']
            poi_output = output['poi']

            parcel_output = parcel_output.mean(1)
            poi_output = poi_output.mean(1)

        return parcel_output, poi_output
    def UniSimGraph(self, med_output, hrs_features, poi_features,T0):
        # ���������������Թ�ͼ
        #med_output = med_output / torch.linalg.norm(med_output,axis=1,keepdims=True)
        #med_output = (torch.matmul(med_output,med_output.T) + 1) / 2
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
        ## ��parcel�ڵ�POI�������оۺ�
        ## poi2poi = self.poi_agg(g_poi,poi_features)
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

    def forward(self, g_poi, g_poi_features, img, epoch, T0, labels=None, state='train'):
        hrs_features = self.hrs_encoder(img)  # 4 x 128
        #poi_features, g_poi = self.poi_encoder(g_poi, g_poi_features)
        #poi_features = self.poi_encoder(g_poi, g_poi_features)
        poi_features, g_poi = self.poi_encoder(g_poi, g_poi_features) # �Ƿ����g_poi�����ս��û��Ӱ��
        ori_hrs_features, ori_poi_features = hrs_features.clone(), poi_features.clone()
        with g_poi.local_scope():
            g_poi.ndata['h'] = poi_features
            poi_features = dgl.mean_nodes(g_poi, 'h')
        
        #poi_features = self.poi_readout(g_poi,poi_features)
        med_fused = torch.concat([hrs_features, poi_features], dim=1)
        # ori med_output
        med_fused = med_fused / torch.linalg.norm(med_fused, axis=1, keepdims=True)
        med_output = (torch.matmul(med_fused, med_fused.T) + 1) / 2  # ���������Լ���
        # modified med_output
        #med_output = F.sigmoid(torch.matmul(self.WP(med_fused),med_fused.T)) 
        #hrs_agg, poi_agg, g_parcel = self.UniSimGraph(med_output, hrs_features, poi_features, T0)
        #agg_output = self.MutualAgg(g_poi,hrs_agg,poi_agg,hrs_features,ori_poi_features)
        
        ##poi2img, img2poi = self.getHetGraph(g_poi,hrs_features,ori_poi_features)
        ##with g_poi.local_scope():
        ##    g_poi.ndata['h'] = img2poi
        ##    img2poi = dgl.mean_nodes(g_poi, 'h')
        
        if len(hrs_features) == 1:
            agg_output = self.MutualAgg(g_poi,hrs_features,None,hrs_features,ori_poi_features)
        else:
            hrs_agg, poi_agg, g_parcel = self.UniSimGraph(med_output,hrs_features,poi_features,T0)
            agg_output = self.MutualAgg(g_poi,hrs_agg,poi_agg,hrs_features,ori_poi_features)
        
        #output = torch.concat([hrs_features,poi_features],dim=1)
        #output = torch.concat([poi2img,img2poi],dim=1)
        #output = torch.concat([hrs_agg,poi_agg],dim=1)
        #output = self.fc(self.dropout(output))
        #a = output.detach()
        #output = self.fc2(output)
        #output = self.fc(self.dropout(agg_output))
        output = self.fc(agg_output)

        return output, med_output
        #return a, med_output

class NGLU_China(nn.Module):
    def __init__(self,in_channels,hid_channels,out_channels):
        super().__init__()
        self.poi_encoder = GCN(in_channels,hid_channels,hid_channels)
        self.hrs_encoder = ResNet(Bottleneck, [3, 4, 6, 3])
        self.hrs_encoder.load_state_dict(
            torch.load(r'/data1/yry23/checkpoint/resnet50-19c8e357.pth')
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
            # output = self.hetSANN(HGraph,node_features,HGraph.ntypes,HGraph.etypes)
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






