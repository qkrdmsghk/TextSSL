import pandas as pd
import torch
import numpy as np


class COOC():
    def __init__(self, window_size):
        # self.item_lists = item_lists
        self.window_size = window_size
        super(COOC, self).__init__()
        self.item_pair_set = {}
        self.windows = 0

    def make_item_pair(self, item_set):
        item_pair_set = []
        for i in range(1, len(item_set)):
            for j in range(0, i):
                if item_set[i] != item_set[j]:
                    item_pair = (item_set[i], item_set[j])
                    item_pair_ = (item_set[j], item_set[i])
                    item_pair_set.append(item_pair)
                    item_pair_set.append(item_pair_)
                else:
                    continue
        return item_pair_set

    def sliding_find_cooc(self, item_list):
        # for item_list in self.item_lists:
        if len(item_list) <= self.window_size:
            sub_pair_set = self.make_item_pair(item_list)
            for pair in sub_pair_set:
                if pair not in self.item_pair_set:
                    self.item_pair_set[pair] = 1
                else:
                    self.item_pair_set[pair] += 1
        else:
            for i in range(len(item_list)-self.window_size+1):
                sub_list = item_list[i: i+self.window_size]
                sub_pair_set = self.make_item_pair(sub_list)
                for pair in sub_pair_set:
                    if pair not in self.item_pair_set:
                        self.item_pair_set[pair] = 1
                    else:
                        self.item_pair_set[pair] += 1
        self.windows += len(item_list)-self.window_size+1

        return self.item_pair_set, self.windows

    def dict2df(self, all_cooc_pairs, windows):
        if len(all_cooc_pairs) > 0:
            df = pd.DataFrame.from_dict(all_cooc_pairs, orient='index').reset_index()
            df = df.sort_values(by=0, ascending=False).reset_index(drop=True)
            df['word1'] = df['index'].apply(lambda x: x[0])
            df['word2'] = df['index'].apply(lambda x: x[1])
            df = df.iloc[:, 1:]
            df.columns = ['freq', 'word1', 'word2']
            df['freq'] = df['freq'] / windows
            return df
        else:
            # print('no pairs')
            return ''

def document_cooc(doc, window_size, MAX_TRUNC_LEN=350):
    par_id = 0
    dfs = []
    doc_len = 0
    flag = False
    for line in doc.readlines():
        line = line.split()
        if MAX_TRUNC_LEN != None and doc_len+len(line) >= MAX_TRUNC_LEN:
            break
        doc_len += len(line)
        cooc = COOC(window_size=window_size)
        item_pair_set, windows = cooc.sliding_find_cooc(line)
        if len(set(item_pair_set)) > 1:
            ddf = cooc.dict2df(item_pair_set, windows)
            ddf['paragraph_id'] = par_id
            par_id += 1
            dfs.append(ddf)

    assert doc_len <= MAX_TRUNC_LEN
    if len(dfs)>0:
        dfs = pd.concat(dfs)
    else:
        dfs = ''
    return dfs

def document_cooc_bert(doc, tokenizer, model, window_size, MAX_TRUNC_LEN=400):
    par_id = 0
    dfs = []
    embs = []
    doc_len = 0
    flag = False
    for line in doc.readlines():
        # line = line.split()
        # raw_len = len(line.split())
        tokens = tokenizer.tokenize(line)
        # print(len(line)-raw_len)
        if MAX_TRUNC_LEN != None and doc_len+len(tokens) >= MAX_TRUNC_LEN:
            break
        doc_len += len(tokens)
        cooc = COOC(window_size=window_size)
        item_pair_set, windows = cooc.sliding_find_cooc(tokens)
        if len(set(item_pair_set)) > 1:
            ddf = cooc.dict2df(item_pair_set, windows)
            ddf['paragraph_id'] = par_id
            dfs.append(ddf)


            indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
            encoded_layers, _ = model(torch.tensor([indexed_tokens]))
            emb = pd.DataFrame(np.array(encoded_layers[-1].squeeze(0).data))
            emb['tokens'] = tokens
            emb_ = []
            for t, tdf in emb.groupby('tokens'):
                if len(tdf) > 1:
                    tdf = tdf.iloc[:, :-1].sum()
                    # todo!! one word!!
                    tdf['tokens'] = t
                    tdf = pd.DataFrame(tdf).transpose()
                assert len(tdf) ==1
                emb_.append(tdf)
            emb_ = pd.concat(emb_)
            # print('concat #{} same words'.format(emb.shape[0]-emb_.shape[0]))
            assert ddf['word1'].unique().shape[0] == emb_.shape[0]
            emb_['paragraph_id'] = par_id
            embs.append(emb_)

            par_id += 1

    assert doc_len <= MAX_TRUNC_LEN
    if len(dfs)>0:
        dfs = pd.concat(dfs)
        embs = pd.concat(embs)
    else:
        dfs = ''
        embs = ''
    return dfs, embs

if __name__ == '__main__':
    pass
