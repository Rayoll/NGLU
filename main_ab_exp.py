import pandas as pd
import numpy as np
import random
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import logging
import torch
import torch.nn as nn
from torchvision import transforms
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils.dataset import China_AOI_POI_Sample_Dataset, China_AOI_POI_Dataset
from model.HGModel import NGLU_ablation, Baseline
from utils.auxiliary import dataSplit,Unievaluate
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='heterogeneous node classification')
    parser.add_argument('--city',type=str,default='China',help='experimental city')
    parser.add_argument('--epoch',type=int,default=15,help='epochs for training')
    parser.add_argument('--bs',type=int,default=4,help='batch size')
    parser.add_argument('--total_num',type=int,help='number of samples')
    parser.add_argument('--ntrain',type=int,help='number of training samples')
    parser.add_argument('--nval',type=int,help='number of validation samples')
    parser.add_argument('--lam',type=float,default=0.7,help='hyperparameter to balance the loss')
    parser.add_argument('--gam',type=int,default=85,help='sampling ratio')
    parser.add_argument('--lr',type=float,default=1e-4,help='initial learning rate')
    parser.add_argument('--T0',type=int,default=95,help='percentile for selecting threshold ')
    parser.add_argument('--ckptDir', type=str, default='./ckpt', help='path to store the checkpoint')
    parser.add_argument('--ab_status', type=str, choices=['baseline','w_CDP','w_GMS_HGA','w_PRC','w_o_PRC'])

    args = parser.parse_args()
    return args

def calSimList(med_output,label_matrix):
    med_output = torch.mul(med_output, label_matrix)
    med_output.fill_diagonal_(0)
    sim_elements = med_output[med_output != 0]
    sim_elements = np.array(sim_elements.cpu().data)
    sim_elements = list(sim_elements)
    return sim_elements
def Unitrain(model,train_loader,val_loader,args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_acc, best_epoch, best_kappa = 0., 0, 0.
    criterion_CE = nn.CrossEntropyLoss()
    criterion_SML1 = nn.SmoothL1Loss()
    model.train()
    accs, kappas = [], []
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    for epoch in range(args.epoch):
        total_loss, count = 0., 0
        before_sim_list = np.array(model.sim_list)
        if epoch != 0 and len(before_sim_list) != 0:
            T0 = np.percentile(before_sim_list, args.T0)
        else:
            T0 = args.T0 / 100.0

        logging.info(f'epoch-{epoch},T0-{T0}')
        model.sim_list = []
        with tqdm(total=args.ntrain,desc=f'training {epoch+1}') as pbar:
            for img, graph, aoi_labels, graphid in train_loader:
                img = img.to(device)
                graph = graph.to(device)
                aoi_labels = aoi_labels.to(device)
                graphid = list(graphid) # 得出当前graphid
                graphid = [int(x) for x in graphid]
                graphid = np.array(graphid)
                if args.ab_status == 'baseline' or args.ab_status == 'w_CDP':
                    logits_parcel = model(graph,graph.ndata.pop('attr'),img)
                    loss = criterion_CE(logits_parcel,aoi_labels.long())
                else:
                    # 每个graph对应的POI id
                    logits_parcel, med_logits = model(graph,graph.ndata.pop('attr'),img,T0)
                    label_matrix = aoi_labels.cpu().data
                    label_matrix = torch.zeros(len(label_matrix), 17).scatter_(1, label_matrix.view(-1, 1), 1).to(
                        'cuda')
                    label_matrix = torch.matmul(label_matrix, label_matrix.T)

                    sim_elements = calSimList(med_logits, label_matrix)
                    model.sim_list = model.sim_list + sim_elements
                    if args.ab_status == 'w_GMS_HGA' or args.ab_status == 'w_o_PRC':
                        loss = criterion_CE(logits_parcel, aoi_labels.long())
                    else:
                        loss = (1-args.lam)*criterion_CE(logits_parcel,aoi_labels.long()) + args.lam*criterion_SML1(med_logits,label_matrix)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1
                pbar.update(len(img))

        train_acc, train_kappa, _ = Unievaluate(train_loader,device,model,criterion_CE,args,epoch,T0)
        val_acc, val_kappa, val_loss = Unievaluate(val_loader,device,model,criterion_CE,args,epoch,T0)

        print(
            "Epoch {:05d} | Loss {:.3f} | ValLoss {:.3f} | Train Acc. {:.4f} | Train Kappa. {:.4f} | Val Acc. {:.4f}| Val Kappa. {:.4f}  ".format(
                epoch, total_loss/count, val_loss, train_acc, train_kappa, val_acc, val_kappa
            )
        )
        logging.info(
            "Epoch {:05d} | Loss {:.3f} | ValLoss {:.3f} | Train Acc. {:.4f} | Train Kappa. {:.4f} | Val Acc. {:.4f}| Val Kappa. {:.4f}  ".format(
                epoch, total_loss/count, val_loss, train_acc, train_kappa, val_acc, val_kappa
            )
        )

        accs.append(val_acc)
        kappas.append(val_kappa)

        if val_acc > best_acc:
            best_acc = val_acc
            best_kappa = val_kappa
            best_epoch = epoch
            if not os.path.exists(args.ckptDir):
                os.makedirs(args.ckptDir)
            torch.save(model.state_dict(),os.path.join(args.ckptDir,f'{args.city}_bs{args.bs}_gam{args.gam}_lam{args.lam}_epoch{epoch}_acc{best_acc}.pth'))

    accs, kappas = np.array(accs), np.array(kappas)
    accs.sort()
    kappas.sort()
    accs = accs[::-1]
    kappas = kappas[::-1]
    accs, kappas = accs[:3], kappas[:3]

    return accs, kappas

def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True,warn_only=True)

def main(args):
    set_random_seed(seed=3407)
    data = pd.read_csv(rf'./data/gt_data/{args.city}_aoi17_label.csv', dtype={'id': str})
    data['id'] = data['id'].apply(lambda x: x[2:])
    data = data.drop_duplicates(subset='id')
    data = data.sort_values(by='id')
    valid_aoi_id = np.loadtxt(f'./data/graph_data/cat17V2_{args.city}_DT_GraphIndexData.txt',dtype=str)
    valid_aoi_id = valid_aoi_id[1:]
    data = data[data['id'].apply(lambda x: valid_aoi_id.__contains__(x))]
    valid_AOI_withPOIs_id, AOI_withPOI_label = data['id'].values, data['class'].values
    train_idx, val_idx = dataSplit(aoi_labels=AOI_withPOI_label,train_ratio=0.7,seed=3407)

    if args.ab_status == 'w_CDP' or args.ab_status == 'w_o_PRC':
        aoi_poi_dataset = China_AOI_POI_Sample_Dataset(
            filepath=r'./data/datasets/ChinaSelected',
            imgs_dir=valid_AOI_withPOIs_id,
            labels=AOI_withPOI_label,
            gamma=args.gam,
            transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ToTensor()]),
            raw_dir=f'./data/graph_data/cat17V2_China_DT_GraphData.txt',
            save_dir=f'./cache/',
            cityName=f'{args.city}_parcel_sample_gam{args.gam}',
        )
    else:
        aoi_poi_dataset = China_AOI_POI_Dataset(
            filepath=r'./data/datasets/ChinaSelected',
            imgs_dir=valid_AOI_withPOIs_id,
            labels=AOI_withPOI_label,
            transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ToTensor()]),
            raw_dir=f'./data/graph_data/cat17V2_China_DT_GraphData.txt',
            save_dir=f'./cache/',
            cityName=f'{args.city}_parcel',
        )

    args.total_num = len(aoi_poi_dataset)
    args.ntrain, args.nval = len(train_idx), len(val_idx)

    aoi_poi_train_loader = GraphDataLoader(
        aoi_poi_dataset,
        sampler=SubsetRandomSampler(np.array(train_idx)),
        batch_size=args.bs,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    aoi_poi_val_loader = GraphDataLoader(
        aoi_poi_dataset,
        sampler=SubsetRandomSampler(np.array(val_idx)),
        batch_size=args.bs,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.ab_status == 'baseline' or args.ab_status == 'w_CDP':
        model = Baseline(in_channels=23,hid_channels=128,out_channels=17).to(device)
    else:
        model = NGLU_ablation(in_channels=23,hid_channels=128,out_channels=17,args=args).to(device)

    logging.basicConfig(level=logging.INFO,format='%(levelname)s:%(message)s',filename=f'./log/ablation_{args.ab_status}_{args.city}_bs{args.bs}_gam{args.gam}_lam{args.lam}.txt')
    best_acc, best_kappa = Unitrain(model,aoi_poi_train_loader,aoi_poi_val_loader,args)

    logging.info('************************************')
    logging.info(
       'oa={:.4f}, kappa={:.4f}'.format(np.mean(best_acc),np.mean(best_kappa)))
    logging.info('************************************')
    
    print('************************************')
    print(
       'oa={:.4f}, kappa={:.4f}'.format(np.mean(best_acc),np.mean(best_kappa)))
    print('************************************')

if __name__ == '__main__':
    args = parse_args()
    main(args)












