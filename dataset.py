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
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm


class ChampsDataset(InMemoryDataset):
    def __init__(self, train_df, struc_df, processed_dir,root,debug = -1, add_ele = False,
                 slice=None, transform=None, pre_transform=None, saved_name = 'champs_data.pt',
                 save_id = False, train=False):
        self.df = train_df
        self.save_id = save_id
        if self.df is not None:
            if slice is None:
                self.processed = list(self.df.molecule_name.unique())
            else:
                self.processed = list(self.df.molecule_name.unique())[slice]
            self.fs = [x + ".pt" for x in self.processed]
        else:
            self.processed = None
            self.fs = None
        
        self.struct_df = struc_df
        self.add_ele  = add_ele
        self.debug = debug
        self.saved_name = saved_name
        self.train = train
        super(ChampsDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.saved_name

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        i = 0
        train_gb = self.df.groupby('molecule_name')
        struct_gb = self.struct_df.groupby('molecule_name')
        c = 0 
        for key in tqdm(train_gb.groups.keys()): # 
            cur_mol = train_gb.get_group(key)
            ## 'C_0', 'H_0','N_0','O_0','F_0',
            ## 'C_1', 'H_1','N_1','O_1','F_1'
            np_edges_1 = cur_mol[['atom_index_0','atom_index_1']].values.transpose()
            np_edges_2 = cur_mol[['atom_index_1','atom_index_0']].values.transpose()
            edge_fts_1 = cur_mol[['dist','inv_dist','inv_dist_p_2','inv_dist_p_3', 'coulomb_0_1']].values
            edge_fts = np.concatenate((edge_fts_1, edge_fts_1), axis=0)
            np_edges = np.concatenate((np_edges_1, np_edges_2), axis=1)
            y_cls_1 = cur_mol[['scl_c2_int']].values
            y_cls = np.concatenate((y_cls_1, y_cls_1), axis=0)
            
            y_precs_1 = cur_mol[['scl_c2_precs']].values
            y_precs = np.concatenate((y_precs_1, y_precs_1), axis=0)
            
            ## if we use contribution instead of the direct scalar value, uncomment these lines
            # contribute_1 = cur_mol[['fc','sd','pso', 'dso']].values if self.train else None #.transpose()
            # contribute = np.concatenate((contribute_1, contribute_1), axis=0) if self.train else None
            
            cur_mol_atoms = struct_gb.get_group(key)
            node_attr = cur_mol_atoms[['encoded_atom']].values
            node_pos = cur_mol_atoms[['x','y','z']].values
            # if self.add_ele:
            #     node_fts = ['so_el','do_am_dien','x','y','z'] ## 'C', 'H','N','O','F'
            # else:
            #     node_fts = ['x','y','z'] # 'C', 'H','N','O','F' 
            # np_atoms = cur_mol_atoms[node_fts].values #.transpose()

            type_code_1 = cur_mol[['encoded_type']].values #.transpose()
            type_code = np.concatenate((type_code_1, type_code_1), axis=0)

            # edge_type_1 = cur_mol[['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']].values #.transpose()
            # edge_type = np.concatenate((edge_type_1, edge_type_1), axis=0)
            id_col_1 = cur_mol[['id']].values if self.save_id else None
            id_col = np.concatenate((id_col_1, id_col_1), axis=0) if self.save_id else None
            edge_index = torch.tensor(np_edges, dtype=torch.long)
            torch_pos = torch.tensor(node_pos, dtype=torch.float)
            torch_node = torch.tensor(node_attr, dtype=torch.float)
            torch_edge_fts = torch.tensor(edge_fts, dtype=torch.float)
            type_code_ts = torch.tensor(type_code, dtype=torch.long)
            # edge_atr = torch.tensor(edge_type, dtype=torch.float)
            # contribute_tensor = torch.tensor(contribute, dtype=torch.float) if self.train else None
            y_cls_torch = torch.tensor(y_cls, dtype=torch.float)
            y_precs_torch = torch.tensor(y_precs, dtype=torch.float)
            id_torch = torch.tensor(id_col, dtype=torch.long) if self.save_id else None
            if self.save_id:
                data = Data(x=torch_node, edge_index=edge_index, id__=id_torch, #atom_code = atom_code,
                            y_cls = y_cls_torch, y_precs=y_precs_torch, edge_atr = torch_edge_fts,edge_type = type_code_ts, pos = torch_pos) #type_ts = | contribute = contribute_tensor,
            else:
                data = Data(x=torch_node, edge_index=edge_index, #atom_code = atom_code,
                            y_cls = y_cls_torch, y_precs=y_precs_torch, edge_atr = torch_edge_fts,edge_type = type_code_ts, pos = torch_pos) #type_ts = | id__=id_torch,
            data_list.append(data)
            if self.debug> 0 and c == self.debug:
                break
            if self.debug > 0: 
                c+=1

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
