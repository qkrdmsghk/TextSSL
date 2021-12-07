from torch_geometric.data import InMemoryDataset
import pandas as pd
import shutil, os, sys
import os.path as osp
import torch
import numpy as np
from gensim.models import Word2Vec
sys.path.append('/data/project/yinhuapark/scripts/models/ssl/ssl_make_graphs')
from ConstructDatasetByDocs import *
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool as gap


IMDB_PATH = '/data/project/yinhuapark/IMDB'
PRE_PATH = '/data/project/yinhuapark/DATA_PRE'
RAW_PATH = '/data/project/yinhuapark/DATA_RAW'



class PygDocsGraphDataset(InMemoryDataset):
    def __init__(self, name, split, dic, pt, transform=None, pre_transform=None):
        if pt == '':
            self.imdb_path = osp.join(IMDB_PATH, name)
        else:
            self.imdb_path = osp.join(IMDB_PATH, name+'_'+pt)
        self.split = split
        self.dic = dic
        self.pt = pt
        self.pre_path = osp.join(PRE_PATH, name)
        super(PygDocsGraphDataset, self).__init__(self.imdb_path, transform, pre_transform)
        self.data, self.slices = torch.load(osp.join(self.processed_dir, f'{self.split}.pt'))

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return osp.join(self.imdb_path)

    @property
    def processed_file_names(self):
        return [f'{self.split}.pt']

    def process(self):
        # construct graph by note list.
        if self.pt == "":
            cdbn = ConstructDatasetByDocs(pre_path=self.pre_path, split=self.split, dictionary=self.dic, pt=self.pt)
            data_list = cdbn.construct_datalist()
        else:
            cdbn = ConstructDatasetByDocs(pre_path=self.pre_path, split=self.split, dictionary=self.dic, pt=self.pt)
            data_list = cdbn.construct_datalist_bert()
        data, slices = self.collate(data_list)
        torch.save((data, slices), osp.join(self.processed_dir, f'{self.split}.pt'))
        print('Saving...')

    def get_max_len(self, split):
        slices = torch.load(osp.join(self.processed_dir, f'{split}.pt'))

        return max([(slices['x_p'][i + 1] - slices['x_p'][i]).item() for i in range(len(slices['x_p']) - 1)])




def statistic(train_loader, test_loader):
    note_dist = []
    x_n_dist = []
    x_p_dist = []
    # x_p_1_dist = []
    # x_p_0_dist = []
    edge_n_dist = []
    edge_p_dist = []
    note_node_dist = []
    y_dist = []

    for data in train_loader:
        note_words = (gap(torch.ones_like(data.batch_n), data.batch_n).numpy().tolist())
        # assert data.y_n.shape[0] == len(gap(torch.ones_like(data.batch_n), data.batch_n).numpy().tolist())
        note_dist.append(data.y_n.shape[0])
        x_n_dist.append(data.x_n.shape[0])
        x_p_dist.append(data.x_p.shape[0])
        edge_n_dist.append(data.edge_index_n.shape[1])
        edge_p_dist.append(data.edge_index_p.shape[1])
        note_node_dist += note_words
        # print(data.y_p[:, 1].item())
        # if data.y_p[:, 1].item() == 1:
        #     x_p_1_dist.append(data.x_p.shape[0])
        # else:
        #     x_p_0_dist.append(data.x_p.shape[0])
        y_dist.append(data.y_p.item())

    for data in test_loader:
        note_words = (gap(torch.ones_like(data.batch_n), data.batch_n).numpy().tolist())
        # assert data.y_n.shape[0] == len(gap(torch.ones_like(data.batch_n), data.batch_n).numpy().tolist())
        note_dist.append(data.y_n.shape[0])
        x_n_dist.append(data.x_n.shape[0])
        x_p_dist.append(data.x_p.shape[0])
        edge_n_dist.append(data.edge_index_n.shape[1])
        edge_p_dist.append(data.edge_index_p.shape[1])
        note_node_dist += note_words
        # print(data.y_p[:, 1].item())
        # if data.y_p[:, 1].item() == 1:
        #     x_p_1_dist.append(data.x_p.shape[0])
        # else:
        #     x_p_0_dist.append(data.x_p.shape[0])
        y_dist.append(data.y_p.item())
    nodes = np.array(x_p_dist)
    print('..........# Max Vocab: {}...........'.format(np.max(nodes)))
    print('..........# Min Vocab: {}...........'.format(np.min(nodes)))
    print('..........# Avg Vocab: {}...........'.format(np.mean(nodes)))



    title = 'nodes distribution'
    sns.set_style('darkgrid')
    sns.set_style('ticks')

    # sns.distplot(y_dist)
    sns.distplot(x_p_dist, label='#joint words')
    sns.distplot(x_n_dist, label='#disjoint words')
    plt.legend(title='task: '+args.name)
    plt.title(title)
    # plt.xlim(-1000,6000)
    sns.despine(trim=True)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create in-memory dataset.")

    parser.add_argument('--raw_path', type=str, default='/data/project/yinhuapark/DATA_RAW/')
    parser.add_argument('--task', type=str, default='mr' , help='task name: [20ng]')
    args, _ = parser.parse_known_args()

    dict_path = '{}_vocab.txt'.format(args.task)
    dictionary = open(os.path.join(args.raw_path, args.task, dict_path)).read().split()

    print('load test data...')
    test_set = PygDocsGraphDataset(name=args.task, split='test', dic=dictionary)
    print('load train data...')
    train_set = PygDocsGraphDataset(name=args.task, split='train', dic=dictionary)