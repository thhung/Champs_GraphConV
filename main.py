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


warnings.filterwarnings('ignore')


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
    
    
    pass
  
  
if __name__=="__main__":
    main()

