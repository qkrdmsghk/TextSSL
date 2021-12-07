
from __future__ import absolute_import
from __future__ import print_function

import os, sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import random
random.seed(49297)
from tqdm import tqdm
sys.path.append('../ssl_make_graphs')
from document_utils import *
from pytorch_pretrained_bert import BertTokenizer, BertModel


def process_partition(partition, action, window_size, max_len):
    if window_size !=3:
        output_dir = os.path.join(args.pre_path, args.task+'_ws_{}'.format(window_size))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir = os.path.join(output_dir, partition + '_cooc')
    elif max_len != 350:
        output_dir = os.path.join(args.pre_path, args.task + '_ml_{}'.format(max_len))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir = os.path.join(output_dir, partition + '_cooc')
    else:
        output_dir = os.path.join(args.pre_path, args.task, partition+'_cooc')
    if action == 'make':
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for y in os.listdir(os.path.join(args.raw_path, args.task, partition))[:]:
            if not os.path.exists(os.path.join(output_dir, y)):
                os.mkdir(os.path.join(output_dir, y))
            docs = list(filter(lambda x: x.find('s') != -1, os.listdir(os.path.join(args.raw_path, args.task, partition, y))))
            for name in tqdm(docs[:], desc='Iterating over docs in {}_{}_{}'.format(args.task, partition, y)):
                doc = open(os.path.join(args.raw_path, args.task, partition, y, name), 'r')
                doc_cooc = document_cooc(doc, window_size=window_size, MAX_TRUNC_LEN=max_len)
                if len(doc_cooc) > 0:
                    doc_cooc.to_csv(os.path.join(output_dir, y, name), sep='\t', index=False)
                else:
                    print(y+'_'+name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create data for decompensation prediction task.")

    parser.add_argument('raw_path', type=str)
    parser.add_argument('pre_path', type=str)
    parser.add_argument('task', type=str, help='task name: [20ng]')
    parser.add_argument('--partition', type=str, default='train')
    parser.add_argument('--action', type=str, default='make')
    parser.add_argument('--window_size', type=str, default='3')
    parser.add_argument('--max_len', type=str, default='350')

    args, _ = parser.parse_known_args()

    if not os.path.exists(os.path.join(args.pre_path, args.task)):
        os.makedirs(os.path.join(args.pre_path, args.task))

    process_partition(args.partition, action=args.action, window_size=int(args.window_size), max_len=int(args.max_len))
