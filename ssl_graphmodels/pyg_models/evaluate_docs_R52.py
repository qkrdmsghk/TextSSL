import sys, os
sys.path.append('/data/project/yinhuapark/scripts/models/ssl/ssl_graphmodels')

from comet_ml import Experiment
from comet_ml.experiment import InterruptedExperiment
from config.conf import arg_config
from utils.LoadData import LoadDocsData
from models_docs import DocNet
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
use_gpu = torch.cuda.is_available()
import random
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


label_dict = {}
patient_dict = {}
p_id = 0
for y_id, y in enumerate(os.listdir('/data/project/yinhuapark/DATA_PRE/R52/test_cooc')):
    label_dict[y_id] = y
    for p in os.listdir('/data/project/yinhuapark/DATA_PRE/R52/test_cooc/{}'.format(y)):
        p_id += 1
        patient_dict[p_id] = p
print(label_dict)


def node_embedding_pca(data, hidden_emb=None, exp_dict=None, method='gnn_note'):

    if 'note' not in  method:
        x = hidden_emb[data.x_p_batch==0]
        id = data.x_p_id[data.x_p_batch==0]
        pos = torch.zeros_like(id)
    else:
        x = hidden_emb[data.x_n_batch==0]
        id = data.x_n_id[data.x_n_batch==0]
        pos = data.batch_n[data.x_n_batch==0]

    x = x.detach().cpu().numpy()
    id = id.detach().cpu().numpy()
    pos = pos.detach().cpu().numpy()


    if exp_dict != None:
        row, col = exp_dict['edge_index']
        row_id, col_id = data.x_n_id[row], data.x_n_id[col]
        row_id = row_id.detach().cpu().numpy()
        col_id = col_id.detach().cpu().numpy()

        layer = exp_dict['edge_mask'].squeeze().detach().cpu().numpy()
        weight = exp_dict['edge_weight'].squeeze().detach().cpu().numpy()

        dictionary = pd.DataFrame(loader.dictionary)
        row_word = dictionary.iloc[row_id, :].reset_index(drop=True)
        col_word = dictionary.iloc[col_id, :].reset_index(drop=True)
        df = pd.concat([row_word, col_word], 1)
        df.columns = ['word1', 'word2']
        df['layer'] = layer
        df['weight'] = weight


    pca = PCA(n_components=2)
    pca.fit(x, id)
    res = pca.transform(x)
    freq = pd.Series(id).value_counts()[id].reset_index()

    inter_id1 = freq[freq[0]>0].index.tolist()
    inter_id2 = freq[freq[0]>0]['index'].tolist()


    plt.figure(figsize=(5,4))
    sns.color_palette("Paired")
    hue = ['blue']
    sns.set_context("paper", font_scale=1.0)

    # sns.scatterplot(x=res[:, 0], y=res[:, 1], hue=freq[0])

    pos = list(map(str, pos))
    sns.scatterplot(x=res[:, 0], y=res[:, 1], hue=pos, palette=hue, s=30, alpha=0.6)

    key_words = ['april', 'finance', 'minister', 'ottawa', 'billion', 'ltd', 'offer', 'pipelines', 'dome', 'early', 'implications',
                 'tax', 'completed', 'large', 'revenue', 'takeover', 'transcanada', 'credits', 'period', 'elements', 'petroleum',
                 'loss', 'opposition', 'parties']
    texts = []
    for i1, i2 in zip(inter_id1, inter_id2):
        if loader.dictionary[i2] in key_words:
            if loader.dictionary[i2] == 'takeover':
                bbox_args = dict(boxstyle="round4", fc="lightgreen")
                plt.annotate('takeover', xy=(res[i1, 0], res[i1, 1]), xytext=((res[i1, 0]-2, res[i1, 1]+0.4)),
                             bbox=bbox_args, fontsize=11,
                             arrowprops=dict(alpha=0.5, arrowstyle='wedge',
                                             connectionstyle='arc3, rad=0.5', color='r'))
            else:
                texts.append(plt.text(x=res[i1, 0]+i1*0.01, y=res[i1, 1]+i1*0.01,
                                      s=loader.dictionary[i2], size=8))

    adjust_text(texts, only_move={'points': 'y', 'texts': 'y'}, lw=0.5)
    plt.tight_layout()
    plt.grid()
    plt.show()
    # plt.savefig('GNN.png', dpi=500)

    # a = 0

    return df




def evaluate(loader):
    model.eval()
    nb_correct, nb_total = 0, 0
    labels = []
    preds = []
    # s_ids = pd.read_csv('visualization/correct_samples_our_R52', sep='\t', header=0)['sample'].tolist()
    s_ids = ['0010823_s']
    for i, data in enumerate(tqdm(loader)):
        data = data.to(args.device)
        pred, hidden_emb, exp_dict = model(data, explain=True)
        y = data.y_p.data
        labels.append(y)
        preds.append(pred.data.argmax(-1))
        nb_correct += (pred.data.argmax(dim=-1) == y).sum().item()
        nb_total += len(y)

        if patient_dict[i+1] in s_ids:
            #
            #     # print(patient_dict[i+1], label_dict[y.item()])
            #     # plt.title('True_{}'.format(patient_dict[i+1]))
            df = node_embedding_pca(data, hidden_emb, exp_dict, method=methods)
            # df.to_csv('exp_info_{}'.format(patient_dict[i+1]), sep='\t', index=False)
            # print(df)
        #     # plt.title('True_{}'.format(patient_dict[i+1]))
        #     # node_embedding_pca(data, hidden_emb, added_inter_edge)
        #
        else:
            print(patient_dict[i+1], label_dict[y.item()])
        #     # plt.title('Error_{}'.format(patient_dict[i+1]))
        #     # node_embedding_pca(data)
        #     # plt.title('Error_{}'.format(patient_dict[i+1]))
        #     # node_embedding_pca(data, hidden_emb)


    labels = torch.cat(labels).detach().cpu().numpy()
    preds = torch.cat(preds).detach().cpu().numpy()

    return labels, preds



def get_model_name(num_layer, aggr, seed, tr_split):
    if tr_split != '':
        model_name = 'num_layer_{}_aggr_{}_seed_{}_tr_{}'.format(num_layer, aggr, seed, tr_split)
    else:
        model_name = 'num_layer_{}_aggr_{}_seed_{}'.format(num_layer, aggr, seed)

    return model_name



def assign_params(params, dd):

    params['hidden_dim'] = int(dd['hidden_dimension'])
    params['dropout'] = float(dd['drop_out'])
    params['lr'] = float(dd['lr'])
    params['patience'] = int(dd['patience'])
    params['weight_decay'] = float(dd['weight_decay'])
    params['batch_size'] = int(dd['batch_size'])

    return params



def make_result(labels, preds):
    test_acc = (labels==preds).sum()/labels.shape[0]
    print("Test set results:{}".format(test_acc))

    p, r, f1, s = metrics.precision_recall_fscore_support(labels, preds, average=None)
    df = pd.DataFrame(np.concatenate([[p], [r], [f1], [s]]).T, columns=['precision', 'recall', 'f1', 'support'])
    # df.to_csv(os.path.join(args.result_output, name, methods, file[0][file[0].find('tune'):file[0].find('.pt')]+'_prfs_{}'.format(SEED)), sep='\t', index=True)
    print(df)
    a = pd.DataFrame([labels, preds]).T
    a.columns = ['True', 'Pred']
    a['True_name'] = a['True'].apply(lambda x: label_dict[x])
    a['Pred_name'] = a['Pred'].apply(lambda x: label_dict[x])
    a['sample'] = patient_dict.values()
    a.to_csv('R_52_prediction_labels_{}'.format(args.methods), sep='\t', index=True)
    return a
    # a.to_csv(os.path.join(args.result_output, name, methods, file[0][file[0].find('tune'):file[0].find('.pt')]+'_labels_{}'.format(SEED)), sep='\t', index=True)
    # print("Macro average Test Precision, Recall and F1-Score...")
    # macro_p, macro_r, macro_f, _ = metrics.precision_recall_fscore_support(labels, preds, average='macro')
    # print(metrics.precision_recall_fscore_support(labels, preds, average='macro'))
    #
    # print("\nMicro average Test Precision, Recall and F1-Score...\n")
    # micro_p, micro_r, micro_f, _ = metrics.precision_recall_fscore_support(labels, preds, average='micro')
    # print(metrics.precision_recall_fscore_support(labels, preds, average='micro'))
    # converged_epoch = file[0][6:file[0].find('tune')-1]
    #
    # result = [[-1, True, args.num_layer, args.aggregate, converged_epoch, test_acc, macro_p, macro_r, macro_f, micro_p, micro_r, micro_f]]
    # result = pd.DataFrame(result, columns=['model_id', 'pre_train', 'num_layer', 'aggregation', 'converged_epoch', 'test_acc',
    #                                        'macro_precision', 'macro_recall', 'macro_f1', 'micro_precision', 'micro_recall', 'micro_f1'])
    # result.to_csv(os.path.join(args.result_output, args.name, args.methods, file[0][file[0].find('tune'):file[0].find('.pt')]), sep='\t', index=False)


if __name__ == '__main__':
    args = arg_config()
    name = 'R52'
    best_hp = 7
    final_model_output =  '/data/project/yinhuapark/final_models'
    # SEEDS = [456, 2222, 239874, 456, 2222, 239874, 239784]
    SEED = 1
    num_layer = 3
    params_path = '../config/params_{}'.format(name)
    hps_list = pd.read_csv(params_path, sep='\t', header=0)
    hps = hps_list.iloc[best_hp, :].to_dict()
    # METHODS = ['gnn_note_attn_mine_lin', 'gnn_note', 'gnn']
    temperature = ''
    type = ''
    methods = 'gnn'
    tr_split = '1.0'
    '''
    TODO: METHODS = ['gnn_note', 'gnn'] --> type = ''
          METHODS = ['gnn_note_attn_mine_lin?', 'gnn_note_attn_mine_lin_reg?'] --> temperature = 0.1, 1.0
          METHODS = ['best model with best temperature'] + training set rate [10%, 20%, 30%. 40%, 50%. 60%, ]?? paper./
    '''

    loader = LoadDocsData(name=name, type=type)

    SEED = int(SEED)
    torch.set_num_threads(1)
    if torch.cuda.is_available() and int(args.gpu) >= 0:
        args.device = torch.device('cuda:' + args.gpu)
        setup_seed(SEED)
        torch.cuda.empty_cache()
    else:
        args.device = torch.device('cpu')
    # methods = args.methods
    if type != "":
        methods += '_' + type
    if temperature != "":
        methods += "_temperature_" + temperature


    model_name = get_model_name(num_layer, args.aggregate, SEED, tr_split)
    train_loader, val_loader, test_loader, num_class = loader.get_train_test(batch_size=int(hps['batch_size']),
                                                                             seed=SEED, tr_split=args.tr_split)
    dictionary = loader.dictionary

    params = vars(args)
    params['input_dim'] = 300
    params['output_dim'] = num_class
    params = assign_params(params, hps)
    params['seed'] = SEED
    params['num_layer'] = num_layer
    params['methods'] = methods
    params['type'] = type

    model_path = os.path.join(final_model_output, name, methods, model_name+'.pt')
    print(model_path)
    model = DocNet(params=params).to(params['device'])
    model.load_state_dict(torch.load(model_path))

    labels, preds = evaluate(test_loader)
    make_result(labels, preds)

    # plot_imbalanced_result()