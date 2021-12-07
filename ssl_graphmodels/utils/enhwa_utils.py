
import torch
from torch_scatter import scatter_add, scatter_max, scatter_mean
from utils.AnchorPool import filter_adj
from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap, global_max_pool as gmap
from torch_geometric.utils import add_remaining_self_loops

def transform_pos_n(pos_n, x_p_batch, x_p, x_n_batch):

    num_nodes = scatter_add(x_p_batch.new_ones(x_p.size(0)), x_p_batch, dim=0)
    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
    pos_n = pos_n + cum_num_nodes[x_n_batch]
    del num_nodes, cum_num_nodes
    return pos_n


def transform_note_batch(note_batch, batch):

    num_notes = gmap(note_batch, batch) + 1
    cum_num_notes = torch.cat([num_notes.new_zeros(1), num_notes.cumsum(dim=0)[:-1]], dim=0)
    #print(cum_num_notes)
    #print(batch.shape)
    #print(note_batch.shape)
    note_batch = note_batch + cum_num_notes[batch]

    #print(note_batch.shape)
    #print(note_batch.unique().size())
    del cum_num_notes, num_notes
    return note_batch


def split_and_pad_to_seq(x, note_batch, batch):

    num_notes = gmap(note_batch, batch) + 1
    slice_index = torch.cat([num_notes.new_zeros(1), num_notes.cumsum(dim=0)[:]], dim=0)
    seq_batch_padded = []
    seq_lens = []
    max_len = max(num_notes)
    for i in range(slice_index.size(0)-1):
        seq_pad = torch.zeros(max_len, x.size(1), device=x.device)
        seq = x[slice_index[i]:slice_index[i+1]]
        seq_pad[:seq.shape[0]] = seq
        seq_batch_padded.append(seq_pad)
        seq_lens.append(seq.shape[0])
    seq_batch_padded = torch.stack(seq_batch_padded, dim=0)
    del num_notes, slice_index, max_len, seq_pad, seq
    return seq_batch_padded, seq_lens


def cal_note_degree(p_id, n_id, x_p_batch, x_n_batch):
    '''
    2021/05/25
    for calculating the note degree of each node.
    :param p_id:
    :param n_id:
    :param x_n_batch:
    :return:
    '''
    max_id = torch.cat([n_id.squeeze(), p_id.squeeze()]).max().item() + 1
    n_id_ = n_id.squeeze() + x_n_batch * max_id
    p_id_ = p_id.squeeze() + x_p_batch * max_id

    degree_note = scatter_add(torch.ones(n_id_.size(0), dtype=torch.float, device=p_id.device), n_id_)
    degree_note = degree_note[p_id_]
    assert degree_note.size(0) == p_id.size(0)

    return degree_note


def cal_deg(edge_index, num_nodes, improved=None, dtype=None):
    edge_weight_ = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
    fill_value = 1 if not improved else 2
    edge_index_, edge_weight_ = add_remaining_self_loops(edge_index, edge_weight_, fill_value, num_nodes)
    row, col = edge_index_
    deg = scatter_add(edge_weight_, row, dim=0, dim_size=num_nodes)
    return deg

def cal_dert_proportion(edge_index_p, edge_index_n, p_id, n_id, x_p_batch, x_n_batch):
    '''
    2021/06/08
    for calculating the increased degree proportion of each node.
    :param edge_index_n:
    :param n_id:
    :param edge_index_p:
    :param p_id:
    :param x_n_batch:
    :param x_p_batch:
    :return:
    '''
    max_id = torch.cat([n_id.squeeze(), p_id.squeeze()]).max().item() + 1
    n_id_ = n_id.squeeze() + x_n_batch * max_id
    p_id_ = p_id.squeeze() + x_p_batch * max_id
    deg_n_max = scatter_max(cal_deg(edge_index_n, n_id.size(0)), n_id_)[0][p_id_]

    deg_p = cal_deg(edge_index_p, p_id.size(0))
    assert deg_p.size(0) == deg_n_max.size(0)

    dert_deg = (deg_p-deg_n_max) / deg_p
    return dert_deg


def note2patient_reconstruction(p_id, n_id, edge_index_p, x_n, x_n_batch, x_p_batch, aggregation='max'):
    '''
    :date on 2021/03/30
    :param p_id: raw x_p_id.
    :param n_id: node id from x_n_id after pooling.
    :param edge_index_p: raw edge_index_p.
    :return: unique node id and edge_index_p after pooling.
    '''
    num_nodes_p = scatter_add(x_p_batch.new_ones(p_id.size(0)), x_p_batch, dim=0)
    batch_size = num_nodes_p.size(0)
    cum_num_nodes = torch.cat(
        [num_nodes_p.new_zeros(1),
         num_nodes_p.cumsum(dim=0)[:-1]], dim=0)

    perm = torch.cat([ torch.eq(n_id[x_n_batch == i].unique().unsqueeze(1), p_id[x_p_batch == i].squeeze()).nonzero()[:, 1]
             + cum_num_nodes[i]
             for i in range(batch_size)])

    for i in range(batch_size):
        if not torch.equal(n_id[x_n_batch == i].unique(), p_id[x_p_batch == i].sort()[0].squeeze()):
            ValueError('not good!')

    x_p_batch = x_p_batch[perm]
    edge_index_p, _ = filter_adj(edge_index_p, None, perm)
    p_id = p_id[perm]


    max_id = torch.cat([n_id.squeeze(), p_id.squeeze()]).max().item()+1
    n_id_ = n_id.squeeze() + x_n_batch * max_id
    x_n_ = torch.cat([x_n, torch.ones(x_n.size(0), 1).to(x_n.device)],1)
    if aggregation == 'sum':
        x_p  = scatter_add(x_n_, n_id_, dim=0)
    elif aggregation == 'mean':
        x_p = scatter_mean(x_n_, n_id_, dim=0)
    else:
        x_p = scatter_max(x_n_, n_id_, dim=0)[0]

    x_p = x_p[x_p[:, -1]!=0][:, :-1]
    assert x_p.size(0) == n_id_.unique().size(0) == x_p_batch.size(0)

    return x_p, edge_index_p, x_p_batch, p_id, n_id




