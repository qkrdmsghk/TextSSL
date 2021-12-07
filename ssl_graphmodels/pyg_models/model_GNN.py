import sys, os
# sys.path.append('/data/project/yinhuapark/scripts/models/data_utils')
sys.path.append('/data/project/yinhuapark/scripts/models/ssl/ssl_graphmodels')

import torch
from utils.ATGCConv_GCN import *
from utils.AnchorPool import AnchorPool
from utils.enhwa_utils import *
from torch.nn import Sequential as Seq, Linear as Lin, functional as F, ReLU, init
from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap, global_max_pool as gmap
from torch_geometric.nn import BatchNorm as BN, GraphSizeNorm as GN
use_gpu = torch.cuda.is_available()

class ATGCNet(torch.nn.Module):
    def __init__(self, hidden_dim, num_layer, dropout, func, pre_trained_embedding=None, aggregate='sum', ratio=1.0):
        super(ATGCNet, self).__init__()
        if 'Conv' in func:
            self.func = getattr(self, 'GNN')
        else:
            self.func = getattr(self, func)
        self.aggregate = aggregate
        self.ratio = ratio
        if ',' in self.aggregate:
            self.aggregate = self.aggregate.split(',')
        # self.gnn_note = ATGCConv(Seq(Lin(emb_size, emb_size*2), # GN(), BN(128),
        #                         ReLU(), Lin(emb_size*2, emb_size) # GN(), BN(64)
        # ))
        # self.pool = AnchorPool(emb_size, ratio=0.8)
        # self.gnn_tax = ATGCConv(Seq(Lin(emb_size, emb_size), ReLU(), Lin(emb_size, emb_size)))
        # self.gnn_tax1 = ATGCConv(Seq(Lin(emb_size, emb_size), ReLU(), Lin(emb_size, emb_size)))
        # self.pre_trained_embedding = torch.nn.Embedding.from_pretrained(pre_trained_embedding)
        self.init_dim = pre_trained_embedding.shape[1]
        self.embeds = torch.nn.ModuleList()
        # self.embeds.append(Seq(Lin(self.init_dim, hidden_dim), BN(hidden_dim)))
        self.embeds.append(torch.nn.Embedding(self.init_dim, hidden_dim))
        self.embeds.append(torch.nn.Embedding(self.init_dim, hidden_dim))

        self.convs = torch.nn.ModuleList()
        if 'GNN' in func:
            for i in range(num_layer):
                self.convs.append(ATGCConv_GCN(hidden_dim, hidden_dim, score=False))
        elif 'GCNConv' in func:
            for i in range(num_layer):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif 'GINConv' in func:
            for i in range(num_layer):
                self.convs.append(GINConv(
                    Seq(Lin(hidden_dim, hidden_dim), BN(hidden_dim), ReLU())))
        elif 'GAT' in func:
            for i in range(num_layer):
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1))

        self.lins = torch.nn.ModuleList()
        self.lins.append(Seq(Lin(hidden_dim, hidden_dim), BN(hidden_dim), ReLU()))
        self.lins.append(Lin(hidden_dim, 2))
        self.hidden_dim = hidden_dim

        self.dropout = dropout
        self.data = ''
        # self.bn = BN(hidden_dim)

        # self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        #
        # '---'
        # self.pools = torch.nn.ModuleList()
        # for i in range(num_layer):
        #     self.pools.append(AnchorPool(hidden_dim, ratio=self.ratio))
        # self.pool_info = False

    def embedding(self, w_note):
        self.x_n_id, self.x_p_id = self.data.x_n_id.to(torch.long), self.data.x_p_id.to(torch.long)
        edge_index_p, x_p_batch = self.data.edge_index_p, self.data.x_p_batch
        edge_index_n, x_n_batch, batch_note = self.data.edge_index_n, self.data.x_n_batch, self.data.batch_n
        del self.data

        # x_p = self.pre_trained_embedding(x_p_id).reshape(-1, self.init_dim)
        x_p = self.embeds[0](self.x_p_id).reshape(-1, self.hidden_dim).relu()
        x_p = F.dropout(x_p, p=self.dropout, training=self.training)

        if w_note:
            # x_n = self.pre_trained_embedding(x_n_id).reshape(-1, self.init_dim)
            x_n = self.embeds[1](self.x_n_id).reshape(-1, self.hidden_dim).relu()
            x_n = F.dropout(x_n, p=self.dropout, training=self.training)
            return x_p, x_n, edge_index_p, edge_index_n, x_p_batch, x_n_batch, batch_note

        else: return x_p, edge_index_p, x_p_batch


    def GNN(self):
        x_p, edge_index_p, x_p_batch = self.embedding(w_note=False)

        ''' GNN start '''
        x_ps = []
        for conv in self.convs[:]:
            x_p = conv(x_p, edge_index_p).relu()
            x_p = F.dropout(x_p, p=self.dropout, training=self.training)
            if self.aggregate == 'sum':
                x_ps.append(gap(x_p, x_p_batch))
            if self.aggregate == 'mean':
                x_ps.append(gmp(x_p, x_p_batch))
            if self.aggregate == 'max':
                x_ps.append(gmap(x_p, x_p_batch))

        del edge_index_p, x_p_batch
        x_p = torch.zeros_like(x_ps[0])
        for x in x_ps[-1:]:
            x_p += x
        del x_ps
        ''' GNN end'''
        x_p = F.dropout(x_p, training=self.training)
        x_p = self.lins[0](x_p)
        # x_p = F.dropout(x_p, training=self.training)
        x_p = self.lins[1](x_p)
        x_p = F.softmax(x_p, dim=1)
        return x_p


    def GNN_norm(self):
        x_p, _, edge_index_p, _, x_p_batch, x_n_batch, _ = self.embedding(w_note=True)
        del _
        ''' GNN start '''
        deg_note = cal_note_degree(self.x_p_id, self.x_n_id, x_p_batch, x_n_batch)
        x_ps = []
        for conv in self.convs[:]:
            x_p = conv(x_p, edge_index_p, deg_note).relu()
            x_p = F.dropout(x_p, p=self.dropout, training=self.training)
            if self.aggregate == 'sum':
                x_ps.append(gap(x_p, x_p_batch))
            if self.aggregate == 'mean':
                x_ps.append(gmp(x_p, x_p_batch))
            if self.aggregate == 'max':
                x_ps.append(gmap(x_p, x_p_batch))

        del edge_index_p, x_p_batch, self.x_p_id, self.x_n_id
        x_p = torch.zeros_like(x_ps[0])
        for x in x_ps[-1:]:
            x_p += x
        del x_ps
        ''' GNN end'''
        x_p = F.dropout(x_p, training=self.training)
        x_p = self.lins[0](x_p)
        # x_p = F.dropout(x_p, training=self.training)
        x_p = self.lins[1](x_p)
        x_p = F.softmax(x_p, dim=1)
        return x_p

    def GNN_norm_trainable(self):
        x_p, _, edge_index_p, _, x_p_batch, x_n_batch, _ = self.embedding(w_note=True)
        del _
        ''' GNN start '''
        x_ps = []
        for conv in self.convs[:]:
            x_p = conv(x_p, edge_index_p, deg_note=None).relu()
            x_p = F.dropout(x_p, p=self.dropout, training=self.training)
            if self.aggregate == 'sum':
                x_ps.append(gap(x_p, x_p_batch))
            if self.aggregate == 'mean':
                x_ps.append(gmp(x_p, x_p_batch))
            if self.aggregate == 'max':
                x_ps.append(gmap(x_p, x_p_batch))

        del edge_index_p, x_p_batch, self.x_p_id, self.x_n_id
        x_p = torch.zeros_like(x_ps[0])
        for x in x_ps[-1:]:
            x_p += x
        del x_ps
        ''' GNN end'''
        x_p = F.dropout(x_p, training=self.training)
        x_p = self.lins[0](x_p)
        # x_p = F.dropout(x_p, training=self.training)
        x_p = self.lins[1](x_p)
        x_p = F.softmax(x_p, dim=1)
        return x_p


    def GNN_dert(self):
        x_p, x_n, edge_index_p, edge_index_n, x_p_batch, x_n_batch, _ = self.embedding(w_note=True)
        del _
        ''' GNN start '''
        dert = cal_dert_proportion(edge_index_p, edge_index_n, self.x_p_id, self.x_n_id, x_p_batch, x_n_batch)
        x_ps = []
        for conv in self.convs[:]:
            x_p = conv(x_p, edge_index_p, dert).relu()
            x_p = F.dropout(x_p, p=self.dropout, training=self.training)
            if self.aggregate == 'sum':
                x_ps.append(gap(x_p, x_p_batch))
            if self.aggregate == 'mean':
                x_ps.append(gmp(x_p, x_p_batch))
            if self.aggregate == 'max':
                x_ps.append(gmap(x_p, x_p_batch))

        del edge_index_p, x_p_batch, self.x_p_id, self.x_n_id
        x_p = torch.zeros_like(x_ps[0])
        for x in x_ps[-1:]:
            x_p += x
        del x_ps
        ''' GNN end'''
        x_p = F.dropout(x_p, training=self.training)
        x_p = self.lins[0](x_p)
        # x_p = F.dropout(x_p, training=self.training)
        x_p = self.lins[1](x_p)
        x_p = F.softmax(x_p, dim=1)
        return x_p


    def GNN_POOL(self):
        x_p, edge_index_p, x_p_batch = self.embedding(w_note=False)
        ''' GNN start '''
        x_ps = []
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            x_p = conv(x_p, edge_index_p).relu()
            if self.pool_info:
                x_p, edge_index_p, x_p_batch, _, _, _, perm, score = pool(x_p, edge_index_p, batch=x_p_batch)
                x_p_id = self.x_p_id[perm]
            else:
                x_p, edge_index_p, x_p_batch, _, _, _, _, _ = pool(x_p, edge_index_p, batch=x_p_batch)
            x_p = F.dropout(x_p, p=self.dropout, training=self.training)
            if self.aggregate == 'sum':
                x_ps.append(gap(x_p, x_p_batch))
            if self.aggregate == 'mean':
                x_ps.append(gmp(x_p, x_p_batch))
            if self.aggregate == 'max':
                x_ps.append(gmap(x_p, x_p_batch))

        del edge_index_p, x_p_batch, _
        x_p = torch.zeros_like(x_ps[0])
        for x in x_ps[-1:]:
            x_p += x
        del x_ps
        ''' GNN end'''
        x_p = F.dropout(x_p, training=self.training)
        x_p = self.lins[0](x_p)
        # x_p = F.dropout(x_p, training=self.training)
        x_p = self.lins[1](x_p)
        x_p = F.softmax(x_p, dim=1)
        if self.pool_info:
            return x_p, 'x_p', x_p_id
        else:
            return x_p

    def GNN_POOL_g(self):
        x_p, edge_index_p, x_p_batch = self.embedding(w_note=False)
        ''' GNN start '''
        x_ps = []
        for (i, conv) in enumerate(self.convs[:-1]):
            x_p = conv(x_p, edge_index_p).relu()
            x_p = F.dropout(x_p, p=self.dropout, training=self.training)

        x_p = self.convs[-1](x_p, edge_index_p).relu()
        x_p, edge_index_p, x_p_batch, _, _, _, _, _ = self.pools[-1](x_p, edge_index_p, batch=x_p_batch)
        x_p = F.dropout(x_p, p=self.dropout, training=self.training)

        if self.aggregate == 'sum':
            x_ps.append(gap(x_p, x_p_batch))
        if self.aggregate == 'mean':
            x_ps.append(gmp(x_p, x_p_batch))
        if self.aggregate == 'max':
            x_ps.append(gmap(x_p, x_p_batch))

        del edge_index_p, x_p_batch, _
        x_p = torch.zeros_like(x_ps[0])
        for x in x_ps[-1:]:
            x_p += x
        del x_ps
        ''' GNN end'''
        x_p = F.dropout(x_p, training=self.training)
        x_p = self.lins[0](x_p)
        # x_p = F.dropout(x_p, training=self.training)
        x_p = self.lins[1](x_p)
        x_p = F.softmax(x_p, dim=1)
        return x_p

    def GNN_Note(self):
        _, x_n, _, edge_index_n, _, x_n_batch, batch_note = self.embedding(w_note=True)
        ''' GNN start'''
        for conv in self.convs[:]:
            x_n = conv(x_n, edge_index_n).relu()
            x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        del edge_index_n
        ''' GNN end  '''
        ''' aggregation start'''
        x_n_batch_ = transform_note_batch(batch_note, x_n_batch)
        if self.aggregate[0] == 'sum':
            x_n = gap(x_n, x_n_batch_)
        elif self.aggregate[0] == 'max':
            x_n = gmap(x_n, x_n_batch_)
        x_n_seq_batch_padded, seq_lens = split_and_pad_to_seq(x_n, batch_note, x_n_batch)
        # packed_seqs = torch.nn.utils.rnn.pack_padded_sequence(x_n_seq_batch_padded, seq_lens, batch_first=True, enforce_sorted=False).to(x_n.device)
        # packed_out, _ = self.gru(packed_seqs)
        # out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        if self.aggregate[1] == 'sum':
            x_n = torch.sum(x_n_seq_batch_padded, dim=1) # x_n shape(batch, seq, dim)
        elif self.aggregate[1] == 'mean':
            x_n = torch.mean(x_n_seq_batch_padded, dim=1)
        elif self.aggregate[1] == 'max':
            x_n = torch.max(x_n_seq_batch_padded, dim=1)[0] # return (values, index)
        # del packed_out, packed_seqs, out
        ''' GRU end '''
        ''' MLP start'''
        x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[0](x_n)
        # x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[1](x_n)
        x_n = F.softmax(x_n, dim=1)
        ''' MLP end'''
        return x_n

    def GNN_Note_POOL(self):
        _, x_n, _, edge_index_n, _, x_n_batch, batch_note = self.embedding(w_note=True)
        ''' GNN start'''
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            x_n = conv(x_n, edge_index_n).relu()
            x_n_batch_ = transform_note_batch(batch_note, x_n_batch)
            if self.pool_info:
                x_n, edge_index_n, x_n_batch_, x_n_batch, batch_note, _, perm, score = pool(x_n, edge_index_n, batch=x_n_batch_, batch_=x_n_batch, batch_n=batch_note)
                x_n_id = self.x_n_id[perm]
            else:
                x_n, edge_index_n, x_n_batch_, x_n_batch, batch_note, _, _, _ = pool(x_n, edge_index_n, batch=x_n_batch_, batch_=x_n_batch, batch_n=batch_note)
            x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        del edge_index_n, _
        ''' GNN end  '''
        ''' aggregation start'''
        # x_n_batch_ = transform_note_batch(batch_note, x_n_batch)
        if self.aggregate[0] == 'sum':
            x_n = gap(x_n, x_n_batch_)
        elif self.aggregate[0] == 'max':
            x_n = gmap(x_n, x_n_batch_)
        x_n_seq_batch_padded, seq_lens = split_and_pad_to_seq(x_n, batch_note, x_n_batch)
        del x_n_batch_
        del seq_lens,x_n,x_n_batch
        # packed_seqs = torch.nn.utils.rnn.pack_padded_sequence(x_n_seq_batch_padded, seq_lens, batch_first=True, enforce_sorted=False).to(x_n.device)
        # packed_out, _ = self.gru(packed_seqs)
        # out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        if self.aggregate[1] == 'sum':
            x_n = torch.sum(x_n_seq_batch_padded, dim=1) # x_n shape(batch, seq, dim)
        elif self.aggregate[1] == 'mean':
            x_n = torch.mean(x_n_seq_batch_padded, dim=1)
        elif self.aggregate[1] == 'max':
            x_n = torch.max(x_n_seq_batch_padded, dim=1)[0] # return (values, index)
        del x_n_seq_batch_padded
        ''' GRU end '''
        ''' MLP start'''
        x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[0](x_n)
        # x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[1](x_n)
        x_n = F.softmax(x_n, dim=1)
        ''' MLP end'''
        if self.pool_info:
            return x_n, batch_note, x_n_id
        else:
            return x_n

    def GNN_Note_POOL_g(self):
        _, x_n, _, edge_index_n, _, x_n_batch, batch_note = self.embedding(w_note=True)
        ''' GNN start'''
        for (i, conv) in enumerate(self.convs[:-1]):
            x_n = conv(x_n, edge_index_n).relu()
            x_n_batch_ = transform_note_batch(batch_note, x_n_batch)
            x_n = F.dropout(x_n, p=self.dropout, training=self.training)

        x_n = self.convs[-1](x_n, edge_index_n).relu()
        x_n, edge_index_n, x_n_batch_, x_n_batch, batch_note, _, _, _ = self.pools[-1](x_n, edge_index_n, batch=x_n_batch_, batch_=x_n_batch, batch_n=batch_note)
        x_n = F.dropout(x_n, p=self.dropout, training=self.training)

        del edge_index_n, _
        ''' GNN end  '''
        ''' aggregation start'''
        # x_n_batch_ = transform_note_batch(batch_note, x_n_batch)
        if self.aggregate[0] == 'sum':
            x_n = gap(x_n, x_n_batch_)
        elif self.aggregate[0] == 'max':
            x_n = gmap(x_n, x_n_batch_)
        x_n_seq_batch_padded, seq_lens = split_and_pad_to_seq(x_n, batch_note, x_n_batch)
        del x_n_batch_
        del seq_lens,x_n,x_n_batch
        # packed_seqs = torch.nn.utils.rnn.pack_padded_sequence(x_n_seq_batch_padded, seq_lens, batch_first=True, enforce_sorted=False).to(x_n.device)
        # packed_out, _ = self.gru(packed_seqs)
        # out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        if self.aggregate[1] == 'sum':
            x_n = torch.sum(x_n_seq_batch_padded, dim=1) # x_n shape(batch, seq, dim)
        elif self.aggregate[1] == 'mean':
            x_n = torch.mean(x_n_seq_batch_padded, dim=1)
        elif self.aggregate[1] == 'max':
            x_n = torch.max(x_n_seq_batch_padded, dim=1)[0] # return (values, index)
        del x_n_seq_batch_padded
        ''' GRU end '''
        ''' MLP start'''
        x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[0](x_n)
        # x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[1](x_n)
        x_n = F.softmax(x_n, dim=1)
        ''' MLP end'''
        return x_n

    def GNN_Note_POOL_r(self):
        ''' [conv_n-pooling_n-reconstruction_p]*3 -> conv_p '''
        _, x_n, edge_index_p, edge_index_n, x_p_batch, x_n_batch, batch_note = self.embedding(w_note=True)
        x_p_id, x_n_id = self.x_p_id, self.x_n_id
        # batch_size = 64

        ''' GNN note & reconstruct start'''
        # print('\n===GNN note & reconstruct start')
        raw_x_n = x_n.size(0)
        for i, (conv, pool) in enumerate(zip(self.convs[:-1], self.pools[:-1])):
            x_n = conv(x_n, edge_index_n).relu()
            x_n_batch_ = transform_note_batch(batch_note, x_n_batch)
            x_n, edge_index_n, x_n_batch_, x_n_batch, batch_note, _, perm, _ = pool(x_n, edge_index_n, batch=x_n_batch_, batch_=x_n_batch, batch_n=batch_note)
            x_n = F.dropout(x_n, p=self.dropout, training=self.training)
            x_p, edge_index_p, x_p_batch, x_p_id, x_n_id = note2patient_reconstruction(
                x_p_id, x_n_id[perm], edge_index_p, x_n, x_n_batch, x_p_batch, aggregation=self.aggregate[0])
            # print('\t raw x_n:{} -> x_p:{}'.format(x_n.size(0), x_p.size(0)))

        # print('note graph pooling ratio {:.2f}%'.format(x_n.size(0)/raw_x_n*100))

        # print('GNN note & reconstruct end===')
        ''' GNN note & reconstruct end  '''

        del edge_index_n, _, x_p_id, x_n_id, perm
        del x_n_batch, batch_note

        ''' reconstructed graph conv start'''
        # print('===reconstructed graph conv start')
        for (conv, pool) in zip(self.convs[-1:], self.pools[-1:]):
            x_p = conv(x_p, edge_index_p).relu()
            x_p, edge_index_p, x_p_batch, _, _, _, _, _ = pool(x_p, edge_index_p, batch=x_p_batch)
            x_p = F.dropout(x_p, p=self.dropout, training=self.training)

        # print('patient graph pooling ratio: {:.2f}%'.format(x_p.size(0)/self.x_p_id.size(0)*100))

        # print('reconstructed graph conv end===')

        if self.aggregate[1] == 'sum':
            x_p = gap(x_p, x_p_batch)
        elif self.aggregate[1] == 'max':
            x_p = gmap(x_p, x_p_batch)
        ''' reconstructed graph conv start'''

        ''' MLP start'''
        x_p = F.dropout(x_p, p=self.dropout, training=self.training)
        x_p = self.lins[0](x_p)
        # x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_p = self.lins[1](x_p)
        x_p = F.softmax(x_p, dim=1)
        ''' MLP end'''
        if self.pool_info:
            return x_p, perm, 'x_p'
        else:
            return x_p

    def GNN_Note_POOL_r_g(self):
        ''' conv_n-conv_n-conv_n-pooling_n-reconstruction_p-conv_p '''
        _, x_n, edge_index_p, edge_index_n, x_p_batch, x_n_batch, batch_note = self.embedding(w_note=True)
        x_p_id, x_n_id = self.x_p_id, self.x_n_id
        # batch_size = 64

        ''' GNN note & reconstruct start'''
        # print('\n===GNN note & reconstruct start')
        raw_x_n = x_n.size(0)
        for i, conv in enumerate(self.convs[:-2]):
            x_n = conv(x_n, edge_index_n).relu()
            x_n = F.dropout(x_n, p=self.dropout, training=self.training)

        x_n = self.convs[-2](x_n, edge_index_n).relu()
        x_n_batch_ = transform_note_batch(batch_note, x_n_batch)
        x_n, edge_index_n, x_n_batch_, x_n_batch, batch_note, _, perm, _ = self.pools[-2](x_n, edge_index_n, batch=x_n_batch_, batch_=x_n_batch, batch_n=batch_note)
        x_n = F.dropout(x_n, p=self.dropout, training=self.training)

        x_p, edge_index_p, x_p_batch, x_p_id, x_n_id = note2patient_reconstruction(
            x_p_id, x_n_id[perm], edge_index_p, x_n, x_n_batch, x_p_batch, aggregation=self.aggregate[0])
        # print('\t raw x_n:{} -> x_p:{}'.format(x_n.size(0), x_p.size(0)))
        # print('GNN note & reconstruct end===')
        ''' GNN note & reconstruct end  '''

        del edge_index_n, _, x_p_id, x_n_id, perm
        del x_n_batch, batch_note

        ''' reconstructed graph conv start'''
        # print('===reconstructed graph conv start')
        x_p = self.convs[-1](x_p, edge_index_p).relu()
        x_p, edge_index_p, x_p_batch, _, _, _, _, _ = self.pools[-1](x_p, edge_index_p, batch=x_p_batch)
        x_p = F.dropout(x_p, p=self.dropout, training=self.training)
        # print('reconstructed graph conv end===')

        # print('note graph pooling ratio {:.2f}%'.format(x_n.size(0)/raw_x_n*100))
        # print('patient graph pooling ratio: {:.2f}%'.format(x_p.size(0)/self.x_p_id.size(0)*100))

        if self.aggregate[1] == 'sum':
            x_p = gap(x_p, x_p_batch)
        elif self.aggregate[1] == 'max':
            x_p = gmap(x_p, x_p_batch)
        ''' reconstructed graph conv start'''

        ''' MLP start'''
        x_p = F.dropout(x_p, p=self.dropout, training=self.training)
        x_p = self.lins[0](x_p)
        # x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_p = self.lins[1](x_p)
        x_p = F.softmax(x_p, dim=1)
        ''' MLP end'''
        if self.pool_info:
            return x_p, perm, 'x_p'
        else:
            return x_p

    def MLP(self):
        x_p, edge_index_p, x_p_batch = self.embedding(w_note=False)
        if self.aggregate == 'sum':
            x_p = gap(x_p, x_p_batch)
        elif self.aggregate == 'mean':
            x_p = gmp(x_p, x_p_batch)
        elif self.aggregate == 'max':
            x_p = gmap(x_p, x_p_batch)
        x_p = F.dropout(x_p, training=self.training)
        x_p = self.lins[0](x_p)
        # x_p = F.dropout(x_p, training=self.training)
        x_p = self.lins[1](x_p)
        x_p = F.softmax(x_p, dim=1)
        return x_p

    def MLP_Note(self):
        _, x_n, _, edge_index_n, _, x_n_batch, batch_note = self.embedding(w_note=True)
        ''' aggregation start'''
        x_n_batch_ = transform_note_batch(batch_note, x_n_batch)
        if self.aggregate[0] == 'sum':
            x_n = gap(x_n, x_n_batch_)
        elif self.aggregate[0] == 'max':
            x_n = gmap(x_n, x_n_batch_)
        x_n_seq_batch_padded, seq_lens = split_and_pad_to_seq(x_n, batch_note, x_n_batch)
        if self.aggregate[1] == 'sum':
            x_n = torch.sum(x_n_seq_batch_padded, dim=1)  # x_n shape(batch, seq, dim)
        elif self.aggregate[1] == 'mean':
            x_n = torch.mean(x_n_seq_batch_padded, dim=1)
        elif self.aggregate[1] == 'max':
            x_n = torch.max(x_n_seq_batch_padded, dim=1)[0]  # return (values, index)
        ''' MLP start'''
        x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[0](x_n)
        # x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[1](x_n)
        x_n = F.softmax(x_n, dim=1)
        ''' MLP end'''
        return x_n

    def GRU(self):
        _, x_n, _, edge_index_n, _, x_n_batch, batch_note = self.embedding(w_note=True)
        ''' GRU start'''
        x_n_batch_ = transform_note_batch(batch_note, x_n_batch)
        if self.aggregate[0] == 'sum':
            x_n = gap(x_n, x_n_batch_)
        elif self.aggregate[0] == 'max':
            x_n = gmap(x_n, x_n_batch_)
        x_n_seq_batch_padded, seq_lens = split_and_pad_to_seq(x_n, batch_note, x_n_batch)
        packed_seqs = torch.nn.utils.rnn.pack_padded_sequence(x_n_seq_batch_padded, seq_lens, batch_first=True, enforce_sorted=False).to(x_n.device)
        packed_out, _ = self.gru(packed_seqs)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        if self.aggregate[1] == 'sum':
            x_n = torch.sum(out, dim=1) # x_n shape(batch, seq, dim)
        elif self.aggregate[1] == 'mean':
            x_n = torch.mean(out, dim=1)
        elif self.aggregate[1] == 'max':
            x_n = torch.max(out, dim=1)[0] # return (values, index)

        del packed_out, packed_seqs, out
        ''' GRU end '''
        x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[0](x_n)
        # x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[1](x_n)
        x_n = F.softmax(x_n, dim=1)
        return x_n

    def Transformer(self):
        _, x_n, _, edge_index_n, _, x_n_batch, batch_note = self.embedding(w_note=True)
        ''' GRU start'''
        x_n_batch_ = transform_note_batch(batch_note, x_n_batch)
        if self.aggregate[0] == 'sum':
            x_n = gap(x_n, x_n_batch_)
        elif self.aggregate[0] == 'max':
            x_n = gmap(x_n, x_n_batch_)
        x_n_seq_batch_padded, seq_lens = split_and_pad_to_seq(x_n, batch_note, x_n_batch)
        packed_seqs = torch.nn.utils.rnn.pack_padded_sequence(x_n_seq_batch_padded, seq_lens, batch_first=True, enforce_sorted=False).to(x_n.device)
        packed_out, _ = self.gru(packed_seqs)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        if self.aggregate[1] == 'sum':
            x_n = torch.sum(out, dim=1) # x_n shape(batch, seq, dim)
        elif self.aggregate[1] == 'mean':
            x_n = torch.mean(out, dim=1)
        elif self.aggregate[1] == 'max':
            x_n = torch.max(out, dim=1)[0] # return (values, index)

        del packed_out, packed_seqs, out
        ''' GRU end '''
        x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[0](x_n)
        # x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[1](x_n)
        x_n = F.softmax(x_n, dim=1)
        return x_n

    def GNN_GRU(self):
        _, x_n, _, edge_index_n, _, x_n_batch, batch_note = self.embedding(w_note=True)

        ''' GNN start'''
        for conv in self.convs[:]:
            x_n = conv(x_n, edge_index_n).relu()
            x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        del edge_index_n
        ''' GNN end  '''
        ''' GRU start'''
        x_n_batch_ = transform_note_batch(batch_note, x_n_batch)
        x_n = gap(x_n, x_n_batch_)
        # x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n_seq_batch_padded, seq_lens = split_and_pad_to_seq(x_n, batch_note, x_n_batch)
        packed_seqs = torch.nn.utils.rnn.pack_padded_sequence(x_n_seq_batch_padded, seq_lens, batch_first=True, enforce_sorted=False).to(x_n.device)
        packed_out, _ = self.gru(packed_seqs)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        if self.aggregate[1] == 'sum':
            x_n = torch.sum(out, dim=1) # x_n shape(batch, seq, dim)
        elif self.aggregate[1] == 'mean':
            x_n = torch.mean(out, dim=1)
        elif self.aggregate[1] == 'max':
            x_n = torch.max(out, dim=1)[0] # return (values, index)

        del packed_out, packed_seqs, out
        ''' GRU end '''
        ''' MLP start'''
        x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[0](x_n)
        # x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[1](x_n)
        x_n = F.softmax(x_n, dim=1)
        ''' MLP end'''
        return x_n

    def GNN_POOL_GRU(self):
        _, x_n, _, edge_index_n, _, x_n_batch, batch_note = self.embedding(w_note=True)
        ''' GNN start'''

        for i, (conv, pool) in enumerate(zip(self.convs[:], self.pools[:])):
            # print('layer_{}', i)
            x_n = conv(x_n, edge_index_n).relu()
            x_n_batch_ = transform_note_batch(batch_note, x_n_batch)
            # pos_n = transform_pos_n(pos_n, x_p_batch, x_p, x_n_batch)
            x_n, edge_index_n, x_n_batch_, x_n_batch, batch_note, _, _, _ = pool(x_n, edge_index_n,
                                                                              batch=x_n_batch_,
                                                                              batch_=x_n_batch, batch_n=batch_note)
            x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        del edge_index_n, _,
        ''' GNN end  '''
        ''' GRU start'''
        x_n = gap(x_n, x_n_batch_)
        del x_n_batch_
        # x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n_seq_batch_padded, seq_lens = split_and_pad_to_seq(x_n, batch_note, x_n_batch)
        x_n_seq_batch_padded = torch.nn.utils.rnn.pack_padded_sequence(x_n_seq_batch_padded, seq_lens, batch_first=True, enforce_sorted=False).to(x_n.device)
        #
        x_n_seq_batch_padded, _ = self.gru(x_n_seq_batch_padded)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(x_n_seq_batch_padded, batch_first=True)
        del _, x_n_seq_batch_padded
        if self.aggregate[1] == 'sum':
            x_n = torch.sum(out, dim=1) # x_n shape(batch, seq, dim)
        elif self.aggregate[1] == 'mean':
            x_n = torch.mean(out, dim=1)
        elif self.aggregate[1] == 'max':
            x_n = torch.max(out, dim=1)[0] # return (values, index)


        # del x_n_seq_batch_padded, out, _, x_n_batch
        ''' GRU end '''
        ''' MLP start'''
        x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[0](x_n)
        # x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lins[1](x_n)
        x_n = F.softmax(x_n, dim=1)
        ''' MLP end'''
        return x_n



    def forward(self, data, pool_info=False):
        self.pool_info = pool_info
        self.data = data
        del data
        # print('-------------------------------------------')
        output = self.func()

        return output




