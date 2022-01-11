
from __future__ import absolute_import
from __future__ import print_function

import os, sys
import argparse
from tqdm import tqdm
sys.path.append('../ssl_make_graphs')
from document_utils import *


def process_partition(partition, window_size, max_len):

    output_dir = os.path.join(args.pre_path, args.name, partition+'_cooc')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for y in os.listdir(os.path.join(args.raw_path, args.name, partition))[:]:
        if not os.path.exists(os.path.join(output_dir, y)):
            os.mkdir(os.path.join(output_dir, y))
        docs = list(filter(lambda x: x.find('s') != -1, os.listdir(os.path.join(args.raw_path, args.name, partition, y))))
        for name in tqdm(docs[:], desc='Iterating over docs in {}_{}_{}'.format(args.name, partition, y)):
            doc = open(os.path.join(args.raw_path, args.name, partition, y, name), 'r')
            doc_cooc = document_cooc(doc, window_size=window_size, MAX_TRUNC_LEN=max_len)
            if len(doc_cooc) > 0:
                doc_cooc.to_csv(os.path.join(output_dir, y, name), sep='\t', index=False)
            else:
                print(y+'_'+name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create document cooc graph")

    parser.add_argument('--raw_path', type=str, default='../re-extract_data/DATA_RAW')
    parser.add_argument('--pre_path', type=str, default='../re-extract_data/DATA_PRE')
    parser.add_argument('--name', type=str, default='R52')
    parser.add_argument('--window_size', type=str, default='3')
    parser.add_argument('--max_len', type=str, default='350')

    args, _ = parser.parse_known_args()

    if not os.path.exists(args.pre_path):
        os.makedirs(args.pre_path)
    if not os.path.exists(os.path.join(args.pre_path, args.name)):
        os.makedirs(os.path.join(args.pre_path, args.name))
    process_partition('train', window_size=int(args.window_size), max_len=int(args.max_len))
    process_partition('test', window_size=int(args.window_size), max_len=int(args.max_len))

    print('----done----,\n'
          ' please check your `DATA_PRE/{}/train_cooc or test_cooc` directory;\n'
          ' the processed file is *** with \'s\' in dataframe format'.format(args.name))
