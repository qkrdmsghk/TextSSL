import torch
from torch_geometric.nn.inits import glorot, zeros, reset
from torch.nn import Parameter
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import sys
sys.path.append('../ssl_graphmodels')
from utils.SSL_GCN import SSL_GCN
from torch_geometric.utils import softmax, add_remaining_self_loops

class StructureLearinng(torch.nn.Module):
    def __init__(self, input_dim, sparse, emb_type='1_hop', threshold=0.5, temperature=0.5):
        super(StructureLearinng, self).__init__()
        self.att = Parameter(torch.Tensor(1, input_dim * 2))
        glorot(self.att.data)
        self.emb_layer = int(emb_type.split('_')[0]) - 1
        if self.emb_layer > 0:
            self.gnns = torch.nn.ModuleList()
            for i in range(self.emb_layer):
                self.gnns.append(SSL_GCN(input_dim, input_dim))

        self.threshold = threshold
        self.temperature = temperature
        self.sparse = sparse

    def forward(self, x, edge_index, edge_weight, edge_mask, layer):

        raw_edges = edge_weight[edge_weight==1].shape[0]
        _, edge_mask = add_remaining_self_loops(edge_index, edge_mask, -1, x.size(0))
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, x.size(0))
        '''
            embedding node using emb_type
        '''
        if self.emb_layer > 0:
            for i in range(self.emb_layer):
                x = torch.nn.functional.relu(self.gnns[i](x, edge_index[:, edge_weight!=0], edge_weight[edge_weight!=0]))

        weights = (torch.cat([x[edge_index[0]], x[edge_index[1]]], 1) * self.att).sum(-1)
        weights = torch.nn.functional.leaky_relu(weights) + edge_weight

        row, col = edge_index

        col, col_id = col.sort()
        weights = weights[col_id]
        row = row[col_id]
        edge_index = torch.stack([row, col])
        edge_weight = edge_weight[col_id]

        edge_mask = edge_mask[col_id]

        if self.sparse == 'soft':
            edge_weight = softmax(weights, col)
        elif self.sparse == 'hard':
            weights = softmax(weights, col)
            sample_prob = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(self.temperature, probs=weights)
            y = sample_prob.rsample()
            y_soft = y
            y_hard = (y>self.threshold).to(y.dtype)
            y = (y_hard - y).detach() + y
            intra_edges = y[edge_weight == 1]
            inter_edges = y[edge_weight == 0]
            edge_weight[edge_weight==0] = inter_edges
            edge_mask[(edge_mask==0) & (y==1)] = layer+1
            intra_soft_edge = y_soft[edge_mask==-1]
        else:
            print('sparse operation is not found...')

        assert edge_index.size(1) == edge_weight.size(0)
        torch.cuda.empty_cache()

        return edge_index, edge_weight, y_soft, edge_mask, intra_soft_edge


class GRAPHLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, func, dropout=0, act=torch.nn.ReLU(), bias=True, num_layer=2,  threshold=0.5, temperature=0.5):
        super(GRAPHLayer, self).__init__()
        self.dropout = dropout
        self.act = act
        self.num_layer = num_layer
        self.func = func
        self.threshold = threshold
        self.temperature = temperature
        self.weight = Parameter(torch.Tensor(input_dim, output_dim))
        if bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.convs = torch.nn.ModuleList()
        self.sls = torch.nn.ModuleList()
        for _ in range(self.num_layer):
            if 'lin' in self.func:
                self.Lin = True
            else:
                self.Lin = False
            self.convs.append(SSL_GCN(output_dim, output_dim, Lin=self.Lin))
            if 'attn' in self.func:
                sparse = ''
                if 'soft' in self.func:
                    sparse = 'soft'
                elif 'gumbel' in self.func:
                    sparse = 'hard'
                self.sls.append(StructureLearinng(output_dim, sparse, threshold=self.threshold, temperature=self.temperature))
                torch.cuda.empty_cache()
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, input, **kwargs):
        x = input
        batch = kwargs['batch']
        edge_index = kwargs['edge_index']

        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.act(torch.matmul(x, self.weight) + self.bias)
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)

        if 'attn' in self.func:
            edge_mask = kwargs['edge_mask']
            # regard self as intra nodes! --> -1
            edge_weight = torch.ones((edge_mask.size(0), ), dtype=torch.float, device=edge_mask.device)
            edge_weight[edge_mask!=-1] = 0

            _, edge_mask = add_remaining_self_loops(edge_index, edge_mask, -1, x.size(0))
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, x.size(0))

            raw_edge_intra_weight = torch.ones((edge_weight[edge_mask==-1].size(0), ), dtype=edge_weight.dtype, device=edge_mask.device)
            raw_edge_intra_weight = softmax(raw_edge_intra_weight, edge_index[0][edge_mask==-1])

            raw_size = edge_weight[edge_weight!=0].size(0)
            self.kl_terms = []
            soft_weights = []
            inter_edge_indexs = []
            edge_masks = []
            for i in range(self.num_layer):
                x = self.act(self.convs[i](x, edge_index[:, edge_weight!=0], edge_weight=edge_weight[edge_weight!=0]))
                if i != self.num_layer-1:
                    edge_index, edge_weight, soft_weight, edge_mask, intra_soft_edge = self.sls[i](x, edge_index, edge_weight, edge_mask, layer=i)
                    soft_weights.append(soft_weight[edge_mask==i+1])
                    inter_edge_indexs.append(edge_index[:, edge_mask==i+1])
                    edge_masks.append(edge_mask[edge_mask==i+1])

                if 'reg' in self.func:
                    assert intra_soft_edge.size() == raw_edge_intra_weight.size()
                    log_p = torch.log(raw_edge_intra_weight + 1e-12)
                    log_q = torch.log(intra_soft_edge+ 1e-12)
                    self.kl_terms.append(torch.mean(raw_edge_intra_weight * (log_p - log_q)))

            if 'explain' in kwargs:
                soft_weights = torch.cat(soft_weights)
                inter_edge_indexs = torch.cat(inter_edge_indexs, 1)
                edge_masks = torch.cat(edge_masks)
                assert inter_edge_indexs.shape[1] == soft_weights.shape[0] == edge_masks.shape[0]
                exp_dict = {'edge_index': inter_edge_indexs, 'edge_weight': soft_weights, 'edge_mask': edge_masks}
                return x, exp_dict
            else:
                return x

        else:
            # gnn
            for i in range(self.num_layer):
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


    def forward(self, input, **kwargs):
        x = input
        # dropout
        x = torch.nn.functional.dropout(x, 0.7, training=self.training)
        # dense encode
        x = self.act(torch.matmul(x, self.weight) + self.bias)

        return x

class READOUTLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, func, aggr, dropout=0, act=torch.nn.ReLU()):
        super(READOUTLayer, self).__init__()

        self.func = func
        self.dropout = dropout
        self.act = act
        self.aggr = aggr
        self.emb_weight = Parameter(torch.Tensor(input_dim, input_dim))
        self.emb_bias = Parameter(torch.Tensor(input_dim))


        self.mlp_weight = Parameter(torch.Tensor(input_dim, output_dim))
        self.mlp_bias = Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.emb_weight)
        glorot(self.mlp_weight)
        zeros(self.emb_bias)
        zeros(self.mlp_bias)

    def global_pooling(self, x, batch):
        x_emb = self.act(torch.matmul(x, self.emb_weight)+self.emb_bias)
        # global pooling
        if self.aggr == 'sum':
            x = global_add_pool(x_emb, batch)
        elif self.aggr == 'mean':
            x = global_mean_pool(x_emb, batch)
        elif self.aggr == 'max':
            x = global_max_pool(x_emb, batch)

        return x

    def forward(self, input=None, **kwargs):
        x = input
        batch = kwargs['batch']
        x = self.global_pooling(x, batch)
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.matmul(x, self.mlp_weight) + self.mlp_bias
        return x
