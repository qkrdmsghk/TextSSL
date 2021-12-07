from torch_geometric.data import Data
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset

class PairData(Data):
    def __init__(self, x_n=None, edge_index_n=None, y_n=None, batch_n=None, pos_n=None,
                        x_p=None, edge_index_p=None, y_p=None, edge_attr_p=None):
        super(PairData, self).__init__()
        self.edge_index_n = edge_index_n
        self.x_n = x_n
        self.y_n = y_n
        self.batch_n = batch_n
        self.pos_n = pos_n
        self.edge_index_p = edge_index_p
        self.x_p = x_p
        self.y_p = y_p
        self.edge_attr_p = edge_attr_p

    def __inc__(self, key, value):

        if key == 'edge_index_n':
            return self.x_n.size(0)
        elif key == 'edge_index_p':
            return self.x_p.size(0)
        elif 'edge_index_pr' in key:
            return self.x_pr.size(0)
        elif 'edge_index_gr' in key:
            return self.x_gr.size(0)
        else:
            return super(PairData, self).__inc__(key, value)

