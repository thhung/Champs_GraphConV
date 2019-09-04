import os
import gc
import os.path as osp
import warnings

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
import torch.optim as optim

from torch_geometric.data import Data, DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import NNConv, Set2Set, AGNNConv
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
# from torch_geometric.datasets import QM9
# from torch_geometric.nn import MessagePassing

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm


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
    pass
  
  
if __name__=="__main__":
    main()

