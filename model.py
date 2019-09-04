import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops


class MdlConvBn(torch.nn.Module):
    """ Convience module to concat later in the pipeline of neural networks
    """
    def __init__(self, in_size, out_size, mid_size):
        super(MdlConvBn, self).__init__()
        self.conv = XConv(in_size, out_size, dim=3, kernel_size=3, 
                        hidden_channels = mid_size)

    def forward(self, x, pos, batch=None):
        return self.conv(x,pos,batch)
        
class Net(torch.nn.Module):
    """ Main architecture of graph neural network
    """
    def __init__(self):
        super(Net, self).__init__()
        internal_dim = 256
        self.lin0 = torch.nn.Linear(8, internal_dim)

        m_nn = Sequential(Linear(5, 128), ReLU(), Linear(128, internal_dim * internal_dim))
        self.conv = NNConv(internal_dim, internal_dim, m_nn, aggr='mean', root_weight=False)
        self.gru = GRU(internal_dim, internal_dim)

        self.set2set = Set2Set(internal_dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * internal_dim, internal_dim)
        self.lin2 = torch.nn.Linear(internal_dim, 242)
        
        self.lin_edge = nn.Embedding(8, 128)
        self.node_embb = nn.Embedding(5, 5)
        
        self.lin6 = nn.Sequential(nn.Linear(2* internal_dim, 128), # egde_attr_size,
                                  nn.BatchNorm1d(128),
                                  nn.ReLU(),
                                  nn.Linear(128, 1) # egde_attr_size,
                                  )

    def forward(self, data):
        node_emb = self.node_embb(data.x.long())
        in_p =torch.cat([node_emb.squeeze(1), data.pos], dim=1)
        out = F.relu(self.lin0(in_p))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_atr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            
        row, col = data.edge_index
        
        out_s = out[row]
        out_t = out[col]
        out_concat = torch.cat([out_s, out_t], dim=1)
        
        out = F.relu(self.lin1(out_concat))
        out = self.lin2(out)
        
        precs = self.lin6(out_concat)
        return out, precs
