import torch
from torch_geometric.nn.inits import glorot, zeros, reset
from torch.nn import Parameter
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import sys
sys.path.append('/data/project/yinhuapark/scripts/models/ssl/ssl_graphmodels')
from utils.ATGCConv_GCN import ATGCConv_GCN, GINConv, GATConv, CombConv_GCN, LEConv, GCNConv
from utils.enhwa_utils import transform_note_batch, split_and_pad_to_seq
from torch.nn import Sequential as Seq, Linear as Lin
from torch_geometric.nn import BatchNorm as BN
from utils.sparse_softmax import Sparsemax
from torch_geometric.utils import softmax, dense_to_sparse, add_remaining_self_loops
from torch_scatter import scatter, segment_csr, gather_csr
from torch_geometric.nn.norm import GraphNorm


def statistic(edge_weights, temperature, weightss, thresholds):
    sample_prob = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature, probs=weightss)
    y = sample_prob.rsample()
    # plot(weights[edge_weight==0].detach().cpu(),weights[edge_weight==1].detach().cpu(), title=self.sparse)
    # plot(y[edge_weight==0].detach().cpu(),y[edge_weight==1].detach().cpu(), title=self.sparse)
    y_soft = y
    y_hard = (y > thresholds).to(y.dtype)
    y = (y_hard - y).detach() + y
    intra_edges = y[edge_weights == 1]
    inter_edges = y[edge_weights == 0]

    print('added inter_edges #{}'.format((inter_edges[inter_edges == 1].shape[0]) / (inter_edges.shape[0])))
    print('removed intra_edges #{}'.format((intra_edges[intra_edges == 0].shape[0]) / (intra_edges.shape[0])))

class StructureLearinng(torch.nn.Module):
    '''
    sparse: {'soft', 'gumbel_hard', 'sparse_hard'}
    emb_type: {'1_hop', '2_hop', 'context', 'comb'}
    '''
    def __init__(self, input_dim, sparse, emb_type='1_hop', threshold=0.5, temperature=0.5):
        super(StructureLearinng, self).__init__()
        self.att = Parameter(torch.Tensor(1, input_dim * 2))
        glorot(self.att.data)
        # self.sparse_attention = Sparsemax()
        self.emb_layer = int(emb_type.split('_')[0]) - 1
        if self.emb_layer > 0:
            self.gnns = torch.nn.ModuleList()
            for i in range(self.emb_layer):
                self.gnns.append(ATGCConv_GCN(input_dim, input_dim))

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

        # import matplotlib.pyplot as plt
        # def plot(inter, intra, title):
        #     if inter != None:
        #         plt.hist(inter, label='inter')
        #     if intra != None:
        #         plt.hist(intra, label='intra')
        #     plt.title(title)
        #     plt.legend()
        #     plt.show()

        if self.sparse == 'sparse_hard':
            edge_weight = self.sparse_attention(weights, col)
        elif self.sparse == 'soft':
            edge_weight = softmax(weights, col)
        elif 'gumbel' in self.sparse:
            # plot(weights[edge_weight==0].detach().cpu(),weights[edge_weight==1].detach().cpu(), title='raw edges')
            if self.sparse == 'gumbel_hard':
                weights = softmax(weights, col)
            elif self.sparse == 'gumbel_hard_sig':
                weights = torch.sigmoid(weights)

            # statistic(edge_weight, self.temperature, weights, self.threshold)
            sample_prob = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(self.temperature, probs=weights)
            y = sample_prob.rsample()
            # plot(weights[edge_weight==0].detach().cpu(),weights[edge_weight==1].detach().cpu(), title=self.sparse)
            # plot(y[edge_weight==0].detach().cpu(),y[edge_weight==1].detach().cpu(), title=self.sparse)
            y_soft = y
            y_hard = (y>self.threshold).to(y.dtype)
            y = (y_hard - y).detach() + y
            intra_edges = y[edge_weight == 1]
            inter_edges = y[edge_weight == 0]

            # print('added inter_edges #{}'.format((inter_edges[inter_edges==1].shape[0]) / (inter_edges.shape[0])))
            # print('removed intra_edges #{}'.format((intra_edges[intra_edges==0].shape[0]) / (intra_edges.shape[0])))

            edge_weight[edge_weight==0] = inter_edges
            edge_mask[(edge_mask==0) & (y==1)] = layer+1
            # print('final added edges #{}'.format((edge_weight[edge_weight==1].shape[0]-raw_edges) / raw_edges))
            intra_soft_edge = y_soft[edge_mask==-1]

            # intra_soft_edge = softmax(intra_soft_edge, col[edge_mask==-1])
            # plt.hist(intra_soft_edge.detach().cpu())
            # plt.show()

        assert edge_index.size(1) == edge_weight.size(0)

        torch.cuda.empty_cache()

        return edge_index, edge_weight, y_soft, edge_mask, intra_soft_edge


class GRAPHLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0, act=torch.nn.ReLU(), bias=True, num_layer=2, func='', threshold='', temperature=''):
        super(GRAPHLayer, self).__init__()
        self.dropout = dropout
        self.act = act
        self.num_layer = num_layer
        self.func = func
        self.improved = False
        self.threshold = threshold
        self.temperature = temperature
        if self.threshold == '':
            self.threshold = 0.5
        else:
            self.threshold = float(self.threshold)

        if self.temperature == '':
            self.temperature = 0.5
        else:
            self.temperature = float(self.temperature)

        if 'improved' in self.func:
            self.improved = True
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

            if 'gin' in self.func:
                self.convs.append(GINConv(Seq(Lin(output_dim, output_dim), BN(output_dim), torch.nn.ReLU())))
            elif 'gat' in self.func:
                self.convs.append(GATConv(output_dim, output_dim, heads=1))
            elif 'gcn' in self.func:
                self.convs.append(GCNConv(output_dim, output_dim))
            elif 'LE' in self.func:
                self.convs.append(LEConv(output_dim, output_dim))
            elif 'comb' in self.func:
                alpha = 0.
                share = False
                if 'comb_aug' in self.func:
                    alpha = 1.
                if 'share' in self.func:
                    share = True
                self.convs.append(CombConv_GCN(output_dim, output_dim, alpha, share))

            else:
                self.convs.append(ATGCConv_GCN(output_dim, output_dim, improved = self.improved, Lin=self.Lin))

            emb_type = '1_hop'
            if 'attn' in self.func:
                if 'soft' in self.func:
                    sparse = 'soft'
                elif 'mine' in self.func:
                    sparse = 'gumbel_hard'
                    if '2_hop' in self.func:
                        emb_type = '2_hop'
                elif 'sigmoid' in self.func:
                    sparse = 'gumbel_hard_sig'
                else:
                    sparse = 'sparse_hard'
                self.sls.append(StructureLearinng(output_dim, sparse, emb_type, self.threshold, self.temperature))

                torch.cuda.empty_cache()

            # else:
            #     print('variant gcn_{} is not available'.format(self.gnn_type))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        # glorot(self.lin.weight)
        # zeros(self.lin.bias)

    def forward(self, input, **kwargs):
        x = input
        batch = kwargs['batch']
        edge_index = kwargs['edge_index']

        '''
        update on 08/21/2021
        encoding module: {
            dropout,
            encode --> dense layer,
            dropout
            }
        '''

        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.act(torch.matmul(x, self.weight) + self.bias)
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)

        '''
        update on 08/21/2021
        GNN module: {
            GNN conv 
            }*num_layers
        '''

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
                if 'gin' in self.func:
                    x = self.act(self.convs[i](x, edge_index[:, edge_weight!=0]))
                elif 'comb' in self.func:
                    x = self.act(self.convs[i](x, edge_index[:, edge_weight!=0], edge_weight=edge_weight[edge_weight!=0], edge_mask=edge_mask[edge_weight!=0]))
                else:
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
                    # print(torch.mean(raw_edge_intra_weight * (log_p - log_q)))
                    # raw_edge_intra_weight = intra_soft_edge
            # added_edge = (edge_weight[edge_weight != 0].size(0)-raw_size)/raw_size
            # print('added_edges_{}'.format(added_edge))

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
        # glorot(self.lin.weight)
        # zeros(self.lin.bias)

    def forward(self, input, **kwargs):
        x = input
        # dropout
        x = torch.nn.functional.dropout(x, 0.7, training=self.training)
        # dense encode
        x = self.act(torch.matmul(x, self.weight) + self.bias)
        # x = torch.nn.functional.dropout(x, 0.5, training=self.training)

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
        if 'attn' in self.aggr:
            self.att_weight = Parameter(torch.Tensor(input_dim, 1))
            self.att_bias = Parameter(torch.Tensor(1))

        if self.func == 'two_tower':
            input_dim = input_dim*2

        self.mlp_weight = Parameter(torch.Tensor(input_dim, output_dim))
        self.mlp_bias = Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.emb_weight)
        glorot(self.mlp_weight)
        zeros(self.emb_bias)
        zeros(self.mlp_bias)
        if 'attn' in self.aggr:
            glorot(self.att_weight)
            zeros(self.att_bias)

    def global_pooling(self, x, batch):
        x_emb = self.act(torch.matmul(x, self.emb_weight)+self.emb_bias)
        # global pooling
        if self.aggr == 'sum':
            x = global_add_pool(x_emb, batch)
        elif self.aggr == 'mean':
            x = global_mean_pool(x_emb, batch)
        elif self.aggr == 'max':
            x = global_max_pool(x_emb, batch)
        elif self.aggr == 'attn':
            att = torch.nn.functional.sigmoid(torch.matmul(x, self.att_weight) + self.att_bias)
            x = x_emb*att
            x = global_add_pool(x, batch)
        elif self.aggr == 'attn_1':
            att = torch.nn.functional.sigmoid(torch.matmul(x, self.att_weight) + self.att_bias)
            x = x_emb*att
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = x_max + x_mean

        return x

    def forward(self, input=None, **kwargs):

        # if 'note' in self.func:
        #     x = input
        #     batch = kwargs['batch']
        #     # embedding
        #     x = self.act(torch.matmul(x, self.emb_weight) + self.emb_bias)
        #
        #     batch_n = kwargs['batch_n']
        #     x_n_batch_ = transform_note_batch(batch_n, batch)
        #     x = global_add_pool(x, x_n_batch_)
        #     x, seq_lens = split_and_pad_to_seq(x, batch_n, batch)
        #     x = torch.sum(x, dim=1)

        if self.func == 'two_tower':
            x_n = kwargs['x_n']
            x_n_batch = kwargs['x_n_batch']
            x_p = kwargs['x_p']
            x_p_batch = kwargs['x_p_batch']
            x_n = self.global_pooling(x_n, x_n_batch)
            x_p = self.global_pooling(x_p, x_p_batch)
            x = torch.cat([x_n, x_p], 1)

        else:
            x = input
            batch = kwargs['batch']
            # embedding
            x = self.global_pooling(x, batch)

        # dropout
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)

        # linear layer
        x = torch.matmul(x, self.mlp_weight) + self.mlp_bias
        return x
