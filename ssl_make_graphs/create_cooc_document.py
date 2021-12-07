
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
sys.path.append('/data/project/yinhuapark/scripts/models/ssl/ssl_make_graphs')
from document_utils import *
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


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


def process_partition_bert(partition, action, window_size):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    output_dir = os.path.join(args.pre_path, args.task, partition+'_cooc_bert')
    if action == 'make':
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for y in os.listdir(os.path.join(args.raw_path, args.task, partition))[:]:
            if not os.path.exists(os.path.join(output_dir, y)):
                os.mkdir(os.path.join(output_dir, y))
            docs = list(filter(lambda x: x.find('s') != -1, os.listdir(os.path.join(args.raw_path, args.task, partition, y))))
            for name in tqdm(docs[:], desc='Iterating over docs in {}_{}_{}'.format(args.task, partition, y)):
                doc = open(os.path.join(args.raw_path, args.task, partition, y, name), 'r')
                doc_cooc, doc_bert_embs = document_cooc_bert(doc, tokenizer, model, window_size=window_size)
                if len(doc_cooc) > 0:
                    doc_cooc.to_csv(os.path.join(output_dir, y, name), sep='\t', index=False)
                    doc_bert_embs.to_csv(os.path.join(output_dir, y, name+'_bert_embs'), sep='\t', index=False)
                else:
                    print(y+'_'+name)

# def create_train_vocab():
#     output_dir = os.path.join(args.pre_path, args.task, 'train_vocab.txt')
#     vocab = set()
#     for y in os.listdir(os.path.join(args.raw_path, args.task, 'train'))[:]:
#         docs = list(filter(lambda x: x.find('s')!= -1, os.listdir(os.path.join(args.raw_path, args.task, 'train', y))))
#         for name in tqdm(docs[:], desc='Iterating over docs in {}_{}_{}'.format(args.task, 'train', y)):
#             doc = open(os.path.join(args.raw_path, args.task, 'train', y, name), 'r')
#             for line in doc.readlines():
#                 for word in line.split():
#                     vocab.add(word)
#
#     f = open(output_dir, 'w')
#     f.write('\n'.join(vocab))
#     f.close()

def create_train_vocab_bert():

    vocab_path = os.path.join('/data/project/yinhuapark/DATA_RAW', args.task, 'bert_'+args.task+'_vocab.txt')
    vocab = set()
    for partition in ['train', 'test']:
        output_dir = os.path.join(args.pre_path, args.task, partition+'_cooc_bert')
        for y in os.listdir(os.path.join(output_dir))[:]:
            docs = list(filter(lambda x: x.find('_bert_embs') == -1, os.listdir(os.path.join(output_dir, y))))
            for name in tqdm(docs[:], desc='Iterating over docs in {}_{}_{}'.format(args.task, partition, y)):
                doc = pd.read_csv(os.path.join(output_dir, y, name), sep='\t', header=0)
                for word in doc['word1'].tolist():
                    vocab.add(str(word))

    f = open(vocab_path, 'w')
    f.write('\n'.join(vocab))
    f.close()

    f = open(vocab_path, 'w')
    f.write('\n'.join(vocab))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create data for decompensation prediction task.")

    parser.add_argument('--raw_path', type=str, default='/data/project/yinhuapark/DATA_RAW/')
    parser.add_argument('--pre_path', type=str, default='/data/project/yinhuapark/DATA_PRE/')
    parser.add_argument('--task', type=str, default='R8' , help='task name: [20ng]')
    parser.add_argument('--partition', type=str, default='train')
    parser.add_argument('--action', type=str, default='make')
    parser.add_argument('--window_size', type=str, default='3')
    parser.add_argument('--max_len', type=str, default='350')

    args, _ = parser.parse_known_args()

    if not os.path.exists(os.path.join(args.pre_path, args.task)):
        os.makedirs(os.path.join(args.pre_path, args.task))

    # process_partition(args.partition, action=args.action, window_size=int(args.window_size), max_len=int(args.max_len))
    # process_partition_bert(args.partition, action=args.action, window_size=int(args.window_size))
    create_train_vocab_bert()

    '''
    for EHR dataset.
    '''
    # create_train_vocab()
    # for test
    # new_model = Word2Vec.load(os.path.join(args.pre_path, args.task, 'word2vec_' + args.dimension))
    # vocabulary = new_model.wv.vocab
    # all_sentences = np.load(os.path.join(args.pre_path, args.task, 'all_sentences_length.npy'))
    # print('task: {} --> vocab_size: {} --> len(sentences): {}'.format(args.task, len(vocabulary), all_sentences.shape[0]))
    # process_partition(args.partition, action=False)