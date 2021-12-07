
import numpy as np
from tqdm import tqdm
import os


class GloveTokenizer:
    def __init__(self, filename, unk='<unk>', pad='<pad>'):
        self.filename = filename
        self.unk = unk
        self.pad = pad
        self.stoi = dict()
        self.itos = dict()
        self.embedding_matrix = list()
        with open(filename, 'r', encoding='utf8') as f: # Read tokenizer file
            for i, line in enumerate(tqdm(f.readlines())):
                values = line.split()
                self.stoi[values[0]] = i
                self.itos[i] = values[0]
                self.embedding_matrix.append([float(v) for v in values[1:]])
        if self.unk is not None: # Add unk token into the tokenizer
            i += 1
            self.stoi[self.unk] = i
            self.itos[i] = self.unk
            self.embedding_matrix.append(np.random.rand(len(self.embedding_matrix[0])))
        if self.pad is not None: # Add pad token into the tokenizer
            i += 1
            self.stoi[self.pad] = i
            self.itos[i] = self.pad
            self.embedding_matrix.append(np.zeros(len(self.embedding_matrix[0])))
        self.embedding_matrix = np.array(self.embedding_matrix).astype(np.float32) # Convert if from double to float for efficiency

    def encode(self, sentence):
        if type(sentence) == str:
            sentence = sentence.split(' ')
        elif len(sentence): # Convertible to list
            sentence = list(sentence)
        else:
            raise TypeError('sentence should be either a str or a list of str!')
        encoded_sentence = []
        for word in sentence:
            encoded_sentence.append(self.stoi.get(word, self.stoi[self.unk]))
        return encoded_sentence

    def decode(self, encoded_sentence):
        try:
            encoded_sentence = list(encoded_sentence)
        except Exception as e:
            print(e)
            raise TypeError('encoded_sentence should be either a str or a data type that is convertible to list type!')
        sentence = []
        for encoded_word in encoded_sentence:
            sentence.append(self.itos[encoded_word])
        return sentence

    def embedding(self, encoded_sentence):
        return self.embedding_matrix[np.array(encoded_sentence)]


class Pretrained_embedding_for_dataset:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, tokenizer, train_vocab):
        self.tokenizer = tokenizer
        self.train_vocab = train_vocab
        self.stoi = {'<unk>': 0, '<pad>': 1}  # Re-index
        self.itos = {0: '<unk>', 1: '<pad>'}  # Re-index
        self.vocab_count = len(self.stoi)
        self.embedding_matrix = None
        self.build_vocab()

    def build_vocab(self):
        for vocab in self.train_vocab:
            if vocab in self.tokenizer.stoi.keys():
                self.stoi[vocab] = self.vocab_count
                self.itos[self.vocab_count] = vocab
                self.vocab_count += 1
        self.embedding_matrix = self.tokenizer.embedding(self.tokenizer.encode(list(self.stoi.keys())))


dataset = 'ohsumed'
path = os.path.join('/data/project/yinhuapark/DATA_PRE', dataset)
tokenizer = GloveTokenizer('glove.6B.300d.txt')
train_vocab = open(os.path.join(path, 'train_vocab.txt'), 'r').read().split()
dataset = Pretrained_embedding_for_dataset(tokenizer=tokenizer, train_vocab=train_vocab)
np.save(os.path.join(path, 'train_glove_embedding'), dataset.embedding_matrix)

f = open(os.path.join(path, 'train_glove_vocab.txt'), 'w')
f.write('\n'.join(list(dataset.stoi.keys())))
f.close()