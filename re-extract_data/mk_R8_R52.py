import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
import os
import shutil
import argparse
from tqdm import tqdm

'''
make R8 and R52 dataset!
using filtering methods in https://ana.cachopo.org/datasets-for-single-label-text-categorization;
by eliminating document with less than one or with more than one topic in file "Reuters21578-Apte-115Cat".
'''



def filter_samples(path):
    s = {}
    for split in ['train', 'test']:
        for cla in os.listdir(os.path.join(path, split)):
            for file in os.listdir(os.path.join(path, split, cla)):
                if file in s:
                    s[file] += 1
                else:
                    s[file] = 1

    df = pd.DataFrame([[k, v] for k,v in s.items()], columns=['sample', 'freq'])
    more_than_1_samples = df[df['freq']==1]['sample'].tolist()
    return more_than_1_samples


def move_samples(samples, source_path, target_path, split, labels):
    dic = {}
    all = 0
    source_path = source_path
    for cla in tqdm(labels):
        if not os.path.exists(os.path.join(target_path, split, cla)):
            os.makedirs(os.path.join(target_path, split, cla))
        dic[cla] = 0
        for file in os.listdir(os.path.join(source_path, split, cla)):
            if file in samples:
                shutil.copyfile(os.path.join(source_path, split, cla, file), os.path.join(target_path, split, cla, file))
                dic[cla] += 1
                all+=1
    return dic, all


def cleaned_vocab(name, textgcn_path):
    clean_vocab = set()
    path = os.path.join(textgcn_path, 'corpus', name+'.clean.txt')
    docs = open(path)
    for line in docs.readlines():
        for word in line.split():
            clean_vocab.add(word)

    f = open(os.path.join(args.raw_path, name, '{}_vocab.txt'.format(name)), 'w')
    f.write('\n'.join(clean_vocab))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create data for decompensation prediction task.")

    parser.add_argument('--source_path', type=str, default='Reuters21578-Apte-115Cat')
    parser.add_argument('--raw_path', type=str, default='DATA_RAW')
    parser.add_argument('--textgcn_path', type=str, default='textgcn_data')

    parser.add_argument('--name', type=str, default='R8')

    args, _ = parser.parse_known_args()

    labels = os.path.join(args.textgcn_path,  args.name + '.txt')
    labels = open(labels).readlines()
    labels = [x.strip().split('\t') for x in labels]
    labels = pd.DataFrame(labels, columns=['index', 'split', 'class'])
    labels = labels['class'].unique().tolist()

    if args.name == 'R52':
        assert len(labels) == 52
    else:
        assert len(labels) == 8

    samples = filter_samples(args.source_path)
    print('moving training samples...')
    move_samples(samples, source_path=args.source_path, target_path=os.path.join(args.raw_path, args.name), split='test', labels=labels)
    print('moving test samples...')
    move_samples(samples, source_path=args.source_path, target_path=os.path.join(args.raw_path, args.name), split='train', labels=labels)

    cleaned_vocab(name=args.name, textgcn_path=args.textgcn_path)

    print('----done----, please check your `DATA_RAW/{}` directory'.format(args.name))

