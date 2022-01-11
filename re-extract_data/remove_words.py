import os
import re
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import argparse



def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def filtering(line, vocab):
    line = clean_str(line)
    words = []
    for word in line.split():
        if word in vocab:
            words.append(word)
        else:
            # print(word)
            continue
    line = ' '.join(words)
    return line

def processing_dataset(dataset, raw_path):
    '''
    dataset= 20ng, ohsumed, R8, R52
    :return:
    '''
    vocab = open(os.path.join(raw_path, dataset, '{}_vocab.txt'.format(dataset))).read().split('\n')

    for split in ['train', 'test']:
        for y in os.listdir(os.path.join(raw_path, dataset, split)):
            sent_len = []
            for name in tqdm(os.listdir(os.path.join(raw_path, dataset, split, y)), desc='processing split {}: category {}...'.format(split, y)):
                if '_s' not in name:
                    doc_content_list = []
                    f = open(os.path.join(raw_path, dataset, split, y, name), 'r', encoding='utf8', errors='ignore')
                    strs = f.read()
                    sentences = sent_tokenize(strs)
                    sent_len.append(len(sentences))
                    for sent in sentences:
                        doc_content_list.append(filtering(sent, vocab))
                    f = open(os.path.join(raw_path, dataset, split, y, name + "_s"), 'w')
                    doc_content_str = '\n'.join(doc_content_list)
                    f.write(doc_content_str)
                    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="remove the infrequent words")

    parser.add_argument('--name', type=str, default='R52')
    parser.add_argument('--raw_path', type=str, default='DATA_RAW')

    args, _ = parser.parse_known_args()
    processing_dataset(dataset=args.name, raw_path=args.raw_path)
    print('----done----, please check your `DATA_RAW/{}/train or test` directory; the processed file is *** with \'s\''.format(args.name))
