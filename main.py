import os
import gc
import os.path as osp
import warnings

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, GRU
import torch.optim as optim

from torch_geometric.data import Data, DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import NNConv, Set2Set, AGNNConv
from torch_geometric.nn import GCNConv, SGConv, ChebConv, DynamicEdgeConv, XConv
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
# from torch_geometric.datasets import QM9
# from torch_geometric.nn import MessagePassing

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

from dataset import ChampsDataset
from model import Net


warnings.filterwarnings('ignore')

def eval_df(gb, scaler):
    """
    Eval the results on the validation set which is in form of dataframe. 
    Args:
        gb      : groupby dataframe which contains the results.
        scaler  : the scikit-learn transformer that scaled the training data.
    
    Return:
        log mean MAE score 
    """
    if scaler:
        gb[['pred']] = scaler.inverse_transform(gb[['pred']])
        gb[['true']] = scaler.inverse_transform(gb[['true']]) 
    gb['abs_dif'] = (gb['pred']- gb['true']).abs()
    ss = gb.groupby('type').abs_dif.mean()
    lb_score = np.log(ss).mean()
    return lb_score

def train_eval_epoch(model_, optimizer_,scheduler_, train_loader_, val_loader_, number_epoch, PRINT_EACH_=1000, scaler_=None):
""" A whole pipeline: train - val - repeat for this problem
    Args:
        the name of each args should tell about themself if reader is familiar with the concept of pytorch
        
    Returns: 
        model_ : trained model
        train_losses : loss for training
        val_losses : loss for evaluation
        lb_scores : score for leaderboard of CHAMPS competition, aka mean of log mae for each type. 
"""
    train_losses = []
    val_losses = []
    lb_scores = []
    for ep in range(number_epoch):
        model_.train()
        loss_all = 0
        i = 0
        for data in train_loader_:
            data = data.to(device)
            optimizer_.zero_grad()
            cls_, precs_ = model_(data)
            loss = sum([0.8 * criterion(cls_, data.y_cls.view(-1).long()), 0.2 * F.mse_loss(precs_, data.y_precs)])
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer_.step()
            if scheduler_:
                scheduler_.step()
            if i % PRINT_EACH_==0:
                print(f" Loss item : {loss.item()}")
            i+=1
        train_l = loss_all / len(train_loader_.dataset)
        print(f"avg train loss at epoch {ep} : {train_l}")
        train_losses.append(train_l)

        model_.eval()
        error = []
        lb_error = 0
        nb_edges = 0
        mega_type = []
        mega_pred = []
        mega_true = []
        print_ = True
        with torch.no_grad():
            i = 0
            for data in val_loader_:
                data = data.to(device)
                o_cls, o_precs = model_(data)
                loss = sum([0.8 * F.cross_entropy(o_cls, data.y_cls.view(-1).long()), 0.2 * F.mse_loss(o_precs, data.y_precs)])
                _, predicted = torch.max(o_cls, 1)
                pred = predicted.float() + o_precs.view(predicted.size())
                pred = np.reshape(pred.detach().cpu().numpy(), (-1, 1)) # torch.sum(out, dim=1)
                tp = data.edge_atr.cpu().numpy()
                gt = np.reshape((data.y_cls.float() + data.y_precs).cpu().view(-1).numpy(), (-1,1)) 
                mega_type.append(tp)
                mega_pred.append(pred)
                mega_true.append(gt)
                error.append(loss.item()*data.num_graphs)
                if i % PRINT_EACH_ == 0 :
                    print(f'            cur_loss_avg: {loss}')
                i+=1
        types = np.concatenate(mega_type, axis=0)
        pred = np.concatenate(mega_pred, axis= 0)
        trues = np.concatenate(mega_true, axis= 0)
        dataset = pd.DataFrame({'type' : types[:,0], 'pred' : pred[:,0], 'true' : trues[:,0]})
        val_l = np.sum([x/len(val_loader_.dataset) for x in error])
        print(f"avg val loss at epoch {ep} : {val_l}")
        val_losses.append(val_l)
        l = eval_df(dataset,None) # 
        lb_scores.append(l)
        print(f"Epoch : {ep} val lb loss = {l}")
        del dataset
        gc.collect()
    return model_, train_losses, val_losses, lb_scores

def main():
    ## train_test.parquet is the combination of train.csv and test.csv 
    ## with some rduction on memory the original csv consumes a lot of 
    ## memory and time to load so I saved them on parquet for faster loading.
    train_val_test_df = pd.read_parquet('train_test.parquet', engine='fastparquet', 
                                columns = ['atom_index_0', 'atom_index_1', 'molecule_name', 
                                           'scalar_coupling_constant','type','dist','inv_dist',
                                           'inv_dist_p_2','inv_dist_p_3', 'coulomb_0_1'])
    structures_df = pd.read_csv('structures.csv')
    train_set = ChampsDataset(train_df, structures_df, './processed_node', './', debug=2560, add_ele=False,
                      saved_name='train_data.pt', save_id = False, train=True)
    val_set = ChampsDataset(val_df, structures_df, './processed_node', './', debug=1280, add_ele=False,
                      saved_name='val_data.pt', save_id = False, train=True)
    
    ## if you notice that there is no test set, because the test set is separated here. 
    ## This is the experiment to improve the Local Validation score. 
    
    ## parameter to control the networks
    dim = 256
    egde_attr_size = 8
    input_size = 10 if train_set.add_ele else 8

    OUT_SIZE = 1
    D_LR = 0.001
    NB_EPOCH = 300
    BATCH_SZ = 64
    PRINT_EACH = 1000
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device) # in_size = 1, out_size=OUT_SIZE, num_cls = 242
    
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    criterion = nn.CrossEntropyLoss().to(device)

    pass
  
  
if __name__=="__main__":
    main()

