import torch
from torch_geometric.nn.inits import glorot, zeros, reset
from torch.nn import Parameter
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import sys
sys.path.append('/data/project/yinhuapark/scripts/models/ssl/ssl_graphmodels')
from utils.ATGCConv_GCN import ATGCConv_GCN, GINConv, GATConv
from utils.enhwa_utils import transform_note_batch, split_and_pad_to_seq
from torch.nn import Sequential as Seq, Linear as Lin
from torch_geometric.nn import BatchNorm as BN

class GRAPHLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0, act=torch.nn.ReLU(), bias=True, num_layer=2, gnn_type='gcn'):
        super(GRAPHLayer, self).__init__()
        self.dropout = dropout
        self.act = act
        self.num_layer = num_layer
        self.gnn_type = gnn_type

        self.weight = Parameter(torch.Tensor(input_dim, output_dim))
        if bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.convs = torch.nn.ModuleList()
        for _ in range(self.num_layer):
            if 'gin' in self.gnn_type:
                self.convs.append(GINConv(Seq(Lin(output_dim, output_dim), BN(output_dim), torch.nn.ReLU())))
            elif 'gat' in self.gnn_type:
                self.convs.append(GATConv(output_dim, output_dim, heads=1))
            else:
                self.convs.append(ATGCConv_GCN(output_dim, output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        # glorot(self.lin.weight)
        # zeros(self.lin.bias)

    def forward(self, input, **kwargs):
        x = input
        edge_index = kwargs['edge_index']

        # dropout
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        # encode --> dense layer
        x = self.act(torch.matmul(x, self.weight) + self.bias)
        # gnn
        for i in range(self.num_layer):
            x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
            x = self.act(self.convs[i](x, edge_index))
        return x


class DENSELayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0, act=torch.nn.ReLU(), bias=False):
        super(DENSELayer, self).__init__()
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.Tensor(input_dim, output_dim))
        if bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        # glorot(self.lin.weight)
        # zeros(self.lin.bias)

    def forward(self, input, **kwargs):
        x = input
        # dropout
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        # dense encode
        x = self.act(torch.matmul(x, self.weight) + self.bias)

        return x

class READOUTLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, func, dropout=0, act=torch.nn.ReLU(), aggr='sum'):
        super(READOUTLayer, self).__init__()

        self.func = func
        self.dropout = dropout
        self.act = act
        self.aggr = aggr
        self.emb_weight = Parameter(torch.Tensor(input_dim, input_dim))
        self.emb_bias = Parameter(torch.Tensor(input_dim))

        if 'cat' in self.func:
            output_dim = 2*output_dim

        self.mlp_weight = Parameter(torch.Tensor(input_dim, output_dim))
        self.mlp_bias = Parameter(torch.Tensor(output_dim))
        # self.lin_emb = torch.nn.Linear(input_dim, input_dim)
        # self.lin_mlp = torch.nn.Linear(input_dim, output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.emb_weight)
        glorot(self.mlp_weight)
        zeros(self.emb_bias)
        zeros(self.mlp_bias)
        # glorot(self.lin_emb.weight)
        # glorot(self.lin_mlp.weight)
        # zeros(self.lin_emb.bias)
        # zeros(self.lin_mlp.bias)

    def forward(self, input, **kwargs):
        x = input
        batch = kwargs['batch']
        # embedding
        x = self.act(torch.matmul(x, self.emb_weight) + self.emb_bias)

        # paragraph node only!
        if 'docpr' in self.func:
            if  'only' in self.func:
                batch = batch[kwargs['x_pr_mask']!=-1]
                x = x[kwargs['x_pr_mask']!=-1]
                assert batch.size(0) == x.size(0)
            elif 'cat' in self.func:
                batch_pr = batch[kwargs['x_pr_mask']!=-1]
                x_pr = x[kwargs['x_pr_mask']!=-1]
                batch_w = batch[kwargs['x_pr_mask']==-1]
                x_w = x[kwargs['x_pr_mask']==-1]
                assert batch_w.size(0) == x_w.size(0)
                assert batch_pr.size(0) == x_pr.size(0)
        if 'docgr' in self.func:
            if  'only' in self.func:
                batch = batch[kwargs['x_gr_mask']!=-1]
                x = x[kwargs['x_gr_mask']!=-1]
                assert batch.size(0) == x.size(0)

        if 'note' in self.func:
            batch_n = kwargs['batch_n']
            x_n_batch_ = transform_note_batch(batch_n, batch)
            x = global_add_pool(x, x_n_batch_)
            x, seq_lens = split_and_pad_to_seq(x, batch_n, batch)
            x = torch.sum(x, dim=1)
        else:
            # global pooling
            if self.aggr == 'sum':
                x = global_add_pool(x, batch)
            elif self.aggr == 'mean':
                x = global_mean_pool(x, batch)
            elif self.aggr == 'max':
                x = global_max_pool(x, batch)
            elif 'cat' in self.func and self.aggr=='sum' and 'pr' in self.func:
                x_w = global_add_pool(x_w, batch_w)
                x_pr = global_add_pool(x_pr, batch_pr)
                x = torch.cat([x_w, x_pr])
        # dropout
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        # linear layer
        x = torch.matmul(x, self.mlp_weight) + self.mlp_bias
        return x

class READOUT_cat_Layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, func, dropout=0, act=torch.nn.ReLU(), aggr='sum', aggr_='cat'):
        super(READOUT_cat_Layer, self).__init__()

        self.func = func
        self.dropout = dropout
        self.act = act
        self.aggr = aggr
        self.aggr_ = aggr_
        self.emb_weight = Parameter(torch.Tensor(input_dim, input_dim))
        self.emb_bias = Parameter(torch.Tensor(input_dim))

        if self.aggr_ == 'cat':
            input_dim = input_dim * 2

        self.mlp_weight = Parameter(torch.Tensor(input_dim, output_dim))
        self.mlp_bias = Parameter(torch.Tensor(output_dim))


        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.emb_weight)
        glorot(self.mlp_weight)
        zeros(self.emb_bias)
        zeros(self.mlp_bias)


    def forward(self, **kwargs):
        xs = []
        for x, batch in zip([kwargs['x_p'], kwargs['x_n']], [kwargs['x_p_batch'],kwargs['x_n_batch']]):
            # embedding
            x = self.act(torch.matmul(x, self.emb_weight) + self.emb_bias)
            # global pooling
            if self.aggr == 'sum':
                x = global_add_pool(x, batch)
            elif self.aggr == 'mean':
                x = global_mean_pool(x, batch)
            elif self.aggr == 'max':
                x = global_max_pool(x, batch)
            xs.append(x)

        if self.aggr_ == 'cat':
            x = torch.cat(xs,dim=1)
        else:
            x = xs[0] + xs[1]
        # dropout
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        # linear layer
        x = torch.matmul(x, self.mlp_weight) + self.mlp_bias
        return x
