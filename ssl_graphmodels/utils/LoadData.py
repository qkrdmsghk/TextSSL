import sys, os
sys.path.append('/data/project/yinhuapark/scripts/models/ssl/ssl_make_graphs')
# from PygNotesGraphDataset import PygNotesGraphDataset as PNGD
from PygDocsGraphDataset import PygDocsGraphDataset as PDGD
from torch_geometric.data import DataLoader
import torch
import numpy as np
import random
import argparse
from gensim.models import Word2Vec
import pandas as pd
from tqdm import tqdm

def show_statisctic(train_set, test_set):
    # min_len = 10000
    # aver_len = 0
    # max_len = 0
    training_sent_num = []
    training_vocab = set()
    training_words = []
    training_jisjoint_words = []
    for data in tqdm(train_set):
        training_sent_num.append(data.batch_n[-1].item() + 1)
        training_words.append(data.x_p.size(0))
        training_jisjoint_words.append(data.x_n.size(0))
        for word in data.x_n_id.data.numpy().tolist():
            training_vocab.add(word)

    train_cs = pd.DataFrame(train_set.data.y_p.tolist())[0].value_counts().values.tolist()
    train_p = train_cs[-1] / train_cs[0]
    test_vocab = set()
    test_sent_num = []
    intersected_vocab = set()
    test_words = []
    test_disjoint_words = []
    for data in tqdm(test_set):
        test_sent_num.append(data.batch_n[-1].item()+1)
        test_words.append(data.x_p.size(0))
        test_disjoint_words.append(data.x_n.size(0))
        for word in data.x_n_id.data.numpy().tolist():
            test_vocab.add(word)
            if word in training_vocab:
                intersected_vocab.add(word)

    test_cs = pd.DataFrame(test_set.data.y_p.tolist())[0].value_counts().values.tolist()
    test_p = test_cs[-1] / test_cs[0]

    avg_trianing_sent_num = np.array(training_sent_num).mean()
    avg_test_sent_num= np.array(test_sent_num).mean()
    avg_sent_num = np.array(training_sent_num+test_sent_num).mean()

    avg_training_words = np.array(training_words).mean()
    avg_training_disjoint_words = np.array(training_jisjoint_words).mean()
    avg_test_words = np.array(test_words).mean()
    avg_test_disjoint_words = np.array(test_disjoint_words).mean()
    avg_words = np.array(training_words+test_words).mean()
    avg_disjoint_words = np.array(training_jisjoint_words+test_disjoint_words).mean()

    print('training_vocab {}, test_vocab {}, intersected_vocab {}, new word porportion {}'.format(len(training_vocab), len(test_vocab), len(intersected_vocab), 1-(len(intersected_vocab)/len(test_vocab))))
    print('training_sent_num {}, test_sent_num {}, all_sent_num {}'.format(avg_trianing_sent_num, avg_test_sent_num, avg_sent_num))
    print('training_joint_words {}, test_joint_words {}, all_joint_words {}'.format(avg_training_words, avg_test_words, avg_words))
    print('training_disjoint_words {}, test_disjoint_words {}, all_disjoint_words {}'.format(avg_training_disjoint_words, avg_test_disjoint_words, avg_disjoint_words))
    print('training_imbalanced_rate {}' 'test_imbalanced_rate {} all_imbalanced_rate {}'.format(train_p, test_p, (train_p+test_p)) )
    a = 0



class LoadDocsData():
    def __init__(self, name, type, pretrained=''):
        self.name = name
        self.type = type
        self.pretrained = pretrained
        super(LoadDocsData, self).__init__()
        self.train_set = []
        self.val_set = []
        self.test_set = []
        if '20ng' in name:
            dict_name = '20ng'
        else:
            dict_name = name
        self.dic_path = '{}_vocab.txt'.format(dict_name)
        if pretrained == 'bert':
            self.dic_path = 'bert_{}_vocab.txt'.format(dict_name)

        self.dictionary = open(os.path.join('---input directory---', dict_name, self.dic_path)).read().split()

    class Handle_data(object):
        def __init__(self, type, pretrained):
            self.type = type
            self.pretrained = pretrained
        def __call__(self, data):
            # pad = self.max_len - data.x_p.shape[0]
            # data.mask_p = torch.from_numpy(np.zeros((self.max_len, 1)))
            # data.mask_p[:data.x_p.shape[0], :] = 1.
            # data.x_p = torch.from_numpy(np.pad(data.x_p, ((0, pad), (0, 0)), mode='constant'))

            data.x_n_id = data.x_n[:, 0].to(torch.long)
            data.x_p_id = data.x_p[:, 0].to(torch.long)
            data.x_n = data.x_n[:, 1:].to(torch.float32)
            data.x_p = data.x_p[:, 1:].to(torch.float32)
            if self.type == 'pr':
                '''
                heterogeneous graph construction!
                need add pr(paragraph nodes) and pr-w edges info to PairData object!
                mask
                '''
                row, col = data.edge_index_p
                num_nodes = data.x_p.size(0)

                paragraph_node = data.batch_n + num_nodes
                word_node = data.pos_n

                row = torch.cat([row, paragraph_node, word_node]) # source node
                col = torch.cat([col, word_node, paragraph_node]) # target node

                paragraph_emb = torch.from_numpy(np.random.uniform(-0.01, 0.01, (data.y_n.size(0), data.x_p.size(1)))).to(data.x_p.dtype)

                data.x_pr = torch.cat([data.x_p, paragraph_emb])
                data.x_pr_mask = torch.cat([torch.full((data.x_p.shape[0],),-1).to(torch.long), data.y_n.squeeze(dim=1)], dim=0)
                data.edge_index_pr = torch.stack([row, col])
                # direct edge of word node -> paragraph node.
                data.edge_index_pr_d = torch.cat([data.edge_index_p, torch.stack([word_node, paragraph_node])], dim=1)
                data.edge_index_pr_d_ = torch.cat([data.edge_index_p, torch.stack([paragraph_node, word_node])], dim=1)
                assert data.x_pr.size(0) == data.edge_index_pr.max()+1 == data.edge_index_pr_d.max()+1 == data.x_pr_mask.size(0)

            elif self.type == 'gr':
                '''
                heterogeneous graph construction!
                need add gr(graph nodes) and gr-w edges info to PairData object!
                mask
                '''
                row, col = data.edge_index_p
                gr_emb = torch.from_numpy(np.random.uniform(-0.01, 0.01, (data.y_p.size(0), data.x_p.size(1)))).to(data.x_p.dtype)
                data.x_gr = torch.cat([data.x_p, gr_emb])
                data.x_gr_mask = torch.cat([torch.full((data.x_p.shape[0],),-1).to(torch.long), torch.tensor([1])], dim=0)

                data.edge_index_gr = torch.stack([row, col])
                num_nodes = torch.tensor([data.x_p.size(0)])
                word_nodes = torch.arange(data.x_p.size(0))
                row = torch.cat([row, num_nodes, word_nodes]) # source node
                col = torch.cat([col, word_nodes, num_nodes]) # target node
                data.edge_index_gr = torch.stack([row, col])

                assert data.x_gr.size(0) == data.edge_index_gr.max()+1

                
            elif self.type == 'inter_all':
                '''
                connect inter edge!
                connect all the words in 1-hop neighbor sentence!
                edge_mask == -1 --> intra_edge
                edge_mask ==  0 --> inter_edge
                '''
                row, col = data.edge_index_n
                edge_mask = torch.full((row.size(0),), -1).to(torch.long)
                edges = torch.combinations(torch.arange(data.x_n_id.size(0)), with_replacement=False).T
                row_, col_ = edges
                edges_attr = data.batch_n[row_] - data.batch_n[col_]
                row_ = row_[abs(edges_attr) == 1]
                col_ = col_[abs(edges_attr) == 1]
                edge_mask_ = torch.full((row_.size(0), ), 0).to(torch.long)
                row = torch.cat([row_, row])
                col = torch.cat([col_, col])
                data.edge_mask = torch.cat([edge_mask_, edge_mask])
                data.edge_index_n = torch.stack([row, col])

            # else:
            #     print('NO special data type!!')

            return data

    def split_train_val_data(self, seed, tr_split):
        np.random.seed(seed)

        cs = pd.DataFrame(self.train_set.data.y_p.tolist())[0].value_counts().to_dict()
        train_id = []
        cs_num = {}
        for c in cs:
            cs_num[c] = 0
        for i, data in enumerate(self.train_set):
            c = data.y_p.item()
            if cs_num[c] < round(cs[c]*tr_split):
                cs_num[c] += 1
                train_id.append(i)
        print('training perception->{}'.format(len(train_id)/len(self.train_set)))
        val_id = random.sample(train_id, int(len(train_id) * 0.1))
        train_id = [x for x in train_id if x not in val_id]

        self.val_set = self.train_set[val_id]
        self.train_set = self.train_set[train_id]

        return self.train_set, self.val_set

    def get_train_test(self, batch_size, seed, tr_split):

        print('load test data...')
        self.test_set = PDGD(name=self.name, split='test', dic=self.dictionary, pt=self.pretrained, transform=self.Handle_data(self.type, self.pretrained))
        # x = [data for data in self.test_set]
        print(len(self.test_set))
        print('load train data...')
        self.train_set = PDGD(name=self.name, split='train', dic=self.dictionary, pt=self.pretrained, transform=self.Handle_data(self.type, self.pretrained))


        # show_statisctic(self.train_set, self.test_set)
        self.train_set, self.val_set = self.split_train_val_data(seed, tr_split)
        print(len(self.train_set), len(self.val_set))
        assert self.val_set.data.y_p.unique().size(0) == \
               self.train_set.data.y_p.unique().size(0) == \
               self.test_set.data.y_p.unique().size(0)

        num_class = self.val_set.data.y_p.unique().size(0)
        follow_batch = ['x_n', 'x_p']
        if self.type != '':
            follow_batch.append('x_'+self.type)

        train_loader = DataLoader(self.train_set[:], batch_size=batch_size, follow_batch=follow_batch, shuffle=True)
        val_loader = DataLoader(self.val_set[:], batch_size=batch_size, follow_batch=follow_batch, shuffle=True)
        test_loader = DataLoader(self.test_set[:], batch_size=1, follow_batch=follow_batch, shuffle=False)

        return train_loader, val_loader, test_loader, num_class



def label_distribution(data_set):
    y_1 = 0
    y_0 = 0
    for d in data_set:
        if d.y_p == torch.tensor([1]):
            y_1 += 1
        else:
            y_0 += 1
    print('#y_1: ', str(y_1), ';#y_0: ', str(y_0))
    return y_1 + y_0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="GNN for MRSA prediction.")
    parser.add_argument('--name', type=str, default='mr')
    parser.add_argument('--type', type=str, default='inter_only')
    parser.add_argument('--pretrained', type=str, default='bert')
    # parser.add_argument('--memo', type=str, default='t-')

    args, _ = parser.parse_known_args()
    loader = LoadDocsData(name=args.name, type=args.type, pretrained='bert')
    train_loader, val_loader, test_loader, n_class = loader.get_train_test(batch_size=1, seed=2, tr_split=1.0)

