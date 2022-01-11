from torch_geometric.data import InMemoryDataset
import os, sys
import os.path as osp
import torch
Your_path = '/data/project/yinhuapark/ssl/'
sys.path.append(Your_path+'ssl_make_graphs')


from ConstructDatasetByDocs import *
import argparse


class PygDocsGraphDataset(InMemoryDataset):
    def __init__(self, name, split, dic, pt, transform=None, pre_transform=None):
        imdb_path = Your_path+'re-extract_data/IMDB'
        pre_path = Your_path+'re-extract_data/DATA_PRE'
        if pt == '':
            self.imdb_path = osp.join(imdb_path, name)
        else:
            self.imdb_path = osp.join(imdb_path, name+'_'+pt)
        self.split = split
        self.dic = dic
        self.pt = pt
        self.pre_path = osp.join(pre_path, name)
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Construct In-Memory Datasets")
    parser.add_argument('--name', type=str, default='R52')
    parser.add_argument('--raw_path', type=str, default=Your_path+'re-extract_data/DATA_RAW')
    parser.add_argument('--imdb_path', type=str, default=Your_path+'re-extract_data/IMDB')


    args, _ = parser.parse_known_args()
    if not os.path.exists(args.imdb_path):
        os.makedirs(args.imdb_path)

    vocab = open(os.path.join(args.raw_path, args.name, args.name+'_vocab.txt')).read().split()
    test_set = PygDocsGraphDataset(name=args.name, split='test', dic=vocab, pt='')