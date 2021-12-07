import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
import os
import shutil

'''
date: 2021/07/01
make R8 and R52 dataset!
using filtering methods in https://ana.cachopo.org/datasets-for-single-label-text-categorization;
by eliminating document with less than one or with more than one topic in file "Reuters21578-Apte-115Cat".
'''


name = 'R8'


raw_path = '/data/project/yinhuapark/DATA_RAW/Reuters21578-Apte-115Cat/'
path = '/data/project/yinhuapark/DATA_RAW/'+name
labels = '/data/project/yinhuapark/text_gcn/data/'+name+'.txt'
labels = open(labels).readlines()
labels = [x.strip().split('\t') for x in labels]
labels = pd.DataFrame(labels, columns=['index', 'split', 'class'])
labels = labels['class'].unique().tolist()
if name=='R52':
    assert len(labels) == 52
else:
    assert len(labels) == 8


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
    for cla in labels:
        if not os.path.exists(os.path.join(target_path, split, cla)):
            os.makedirs(os.path.join(target_path, split, cla))
        dic[cla] = 0
        for file in os.listdir(os.path.join(source_path, split, cla)):
            if file in samples:
                # shutil.copyfile(os.path.join(source_path, split, cla, file), os.path.join(target_path, split, cla, file))
                dic[cla] += 1
                all+=1
    print(dic, all)


    return dic, all

samples = filter_samples(raw_path)

raw_path = '/data/project/yinhuapark/DATA_RAW/Reuters21578-Apte-90Cat/'
move_samples(samples, source_path=raw_path, target_path=path, split='test', labels=labels)
move_samples(samples, source_path=raw_path, target_path=path, split='train', labels=labels)

