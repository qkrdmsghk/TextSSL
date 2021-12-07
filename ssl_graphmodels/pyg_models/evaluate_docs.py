import sys, os
sys.path.append('/data/project/yinhuapark/scripts/models/ssl/ssl_graphmodels')

from config.conf import arg_config
from utils.LoadData import LoadDocsData
from models_docs import DocNet
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
use_gpu = torch.cuda.is_available()
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

label_dict = {}
patient_dict = {}
p_id = 0
for y_id, y in enumerate(os.listdir('/data/project/yinhuapark/DATA_PRE/ohsumed/test_cooc')):
    label_dict[y_id] = y
    for p in os.listdir('/data/project/yinhuapark/DATA_PRE/ohsumed/test_cooc/{}'.format(y)):
        p_id += 1
        patient_dict[p_id] = p


def evaluate(loader):
    model.eval()
    nb_correct, nb_total = 0, 0
    labels = []
    preds = []
    for i, data in enumerate(loader):
        data = data.to(args.device)
        pred, hidden_emb = model(data, explain=True)
        y = data.y_p.data
        labels.append(y)
        preds.append(pred.data.argmax(-1))
        nb_correct += (pred.data.argmax(dim=-1) == y).sum().item()
        nb_total += len(y)

        if y==pred.data.argmax(-1) and patient_dict[i+1]=='0018596_s':
            print(patient_dict[i+1], label_dict[y.item()])
            plt.title('True_{}'.format(patient_dict[i+1]))
            node_embedding_pca(data)
            plt.title('True_{}'.format(patient_dict[i+1]))
            node_embedding_pca(data, hidden_emb)

        else:
            print(patient_dict[i+1], label_dict[y.item()])
            # plt.title('Error_{}'.format(patient_dict[i+1]))
            # node_embedding_pca(data)
            # plt.title('Error_{}'.format(patient_dict[i+1]))
            # node_embedding_pca(data, hidden_emb)





    labels = torch.cat(labels).detach().cpu().numpy()
    preds = torch.cat(preds).detach().cpu().numpy()

    return labels, preds



def node_embedding_pca(data, hidden_emb=None):

    if hidden_emb == None:
        x = data.x_n[data.x_n_batch==0].detach().cpu().numpy()

    else:
        x = hidden_emb[data.x_n_batch==0].detach().cpu().numpy()
    id = data.x_n_id[data.x_n_batch==0].detach().cpu().numpy()
    pos = data.pos_n[data.x_n_batch==0].detach().cpu().numpy()

    pca = PCA(n_components=2)
    pca.fit(x, id)
    res = pca.transform(x)
    freq = pd.Series(id).value_counts()[id].reset_index()

    inter_id1 = freq[freq[0]>1].index.tolist()
    inter_id2 = freq[freq[0]>1]['index'].tolist()



    # sns.scatterplot(x=res[:, 0], y=res[:, 1], hue=freq[0])
    pos = list(map(str, pos))
    sns.scatterplot(x=res[:, 0], y=res[:, 1], hue=pos)


    for i1, i2 in zip(inter_id1, inter_id2):
        # print(dictionary[i2])
        # print(res[i1, :])
        plt.text(x=res[i1, 0], y=res[i1, 1], s=dictionary[i2])
    plt.show()





def make_result(labels, preds):
    test_acc = (labels==preds).sum()/labels.shape[0]
    print("Test set results:{}".format(test_acc))

    p, r, f1, s = metrics.precision_recall_fscore_support(labels, preds, average=None)
    df = pd.DataFrame(np.concatenate([[p], [r], [f1], [s]]).T, columns=['precision', 'recall', 'f1', 'support'])
    df.to_csv(os.path.join(args.result_output, name, methods, file[0][file[0].find('tune'):file[0].find('.pt')]+'_prfs_{}'.format(SEED)), sep='\t', index=True)
    print(df)
    a = pd.DataFrame([labels, preds]).T
    a.columns = ['True', 'Pred']
    a['True_name'] = a['True'].apply(lambda x: label_dict[x])
    a['Pred_name'] = a['Pred'].apply(lambda x: label_dict[x])
    a['smaple'] = patient_dict.values()

    a.to_csv(os.path.join(args.result_output, name, methods, file[0][file[0].find('tune'):file[0].find('.pt')]+'_labels_{}'.format(SEED)), sep='\t', index=True)
    print("Macro average Test Precision, Recall and F1-Score...")
    macro_p, macro_r, macro_f, _ = metrics.precision_recall_fscore_support(labels, preds, average='macro')
    print(metrics.precision_recall_fscore_support(labels, preds, average='macro'))

    print("\nMicro average Test Precision, Recall and F1-Score...\n")
    micro_p, micro_r, micro_f, _ = metrics.precision_recall_fscore_support(labels, preds, average='micro')
    print(metrics.precision_recall_fscore_support(labels, preds, average='micro'))
    converged_epoch = file[0][6:file[0].find('tune')-1]

    result = [[-1, True, args.num_layer, args.aggregate, converged_epoch, test_acc, macro_p, macro_r, macro_f, micro_p, micro_r, micro_f]]
    result = pd.DataFrame(result, columns=['model_id', 'pre_train', 'num_layer', 'aggregation', 'converged_epoch', 'test_acc',
                                           'macro_precision', 'macro_recall', 'macro_f1', 'micro_precision', 'micro_recall', 'micro_f1'])
    result.to_csv(os.path.join(args.result_output, args.name, args.methods, file[0][file[0].find('tune'):file[0].find('.pt')]), sep='\t', index=False)


def plot_imbalanced_result():
    import matplotlib.pyplot as plt
    import seaborn as sns

    path = '/data/project/yinhuapark/model_results'
    name = 'ohsumed'
    methods = ['gnn', 'gnn_note']
    num_layer = '2'
    dfs = []
    for method in methods:
        path_ = os.path.join(path, name, method)
        file = 'tune_0_w_pt_0_num_layer_{}_aggr_sum_prfs'.format(num_layer)
        df = pd.read_csv(os.path.join(path_, file), sep='\t', header=0)
        df = df.sort_values(by='support').reset_index(drop=True)
        dfs.append(df)

    diff_dfs = dfs[0] - dfs[1]
    diff_dfs['support'] = dfs[0]['support']
    diff_dfs['label'] = diff_dfs.index

    sns.set_style('whitegrid')
    plt.title('{} label distribution'.format(name))
    plt.bar(diff_dfs['label'], diff_dfs['support'])
    plt.show()

    plt.cla()
    sns.set_style('whitegrid')
    plt.scatter(x=diff_dfs.index, y=diff_dfs['precision'], c='blue')
    plt.scatter(x=diff_dfs.index, y=diff_dfs['f1'], c='green')
    plt.scatter(x=diff_dfs.index, y=diff_dfs['recall'], c='red')

    plt.plot(diff_dfs['recall'], c='red', label='recall')
    plt.plot(diff_dfs['f1'], c='green', label='f1')
    plt.plot(diff_dfs['precision'], c='blue', label='precision')
    plt.plot([0 for i in range(len(diff_dfs))], c='black')
    plt.title('GNN-GNN_Note ({}, layer={})'.format(name, num_layer))
    plt.legend()
    plt.show()

def assign_params(params, dd):

    params['hidden_dim'] = int(dd['hidden_dimension'])
    params['dropout'] = float(dd['drop_out'])
    params['lr'] = float(dd['lr'])
    params['patience'] = int(dd['patience'])
    params['weight_decay'] = float(dd['weight_decay'])
    params['batch_size'] = int(dd['batch_size'])

    return params


if __name__ == '__main__':
    args = arg_config()
    SEED = 123
    num_layer = 2
    name = 'ohsumed'
    type = ''
    batch_size = 1
    methods = 'gnn_note'

    if torch.cuda.is_available() and int(args.gpu) >= 0:
        args.device = torch.device('cuda:'+ args.gpu)
        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
    else:
        args.device = torch.device('cpu')



    loader = LoadDocsData(name=name, type=type)
    _, _, test_loader, num_class = loader.get_train_test(batch_size=batch_size, seed=SEED)

    dictionary = loader.dictionary

    files =  os.listdir(os.path.join(args.model_output, name, methods, str(SEED)))
    file = list(filter(lambda x: x.find('num_layer_'+str(num_layer))!=-1, files))
    assert len(file) == 1
    model_path = os.path.join(args.model_output, name, methods, str(SEED), file[0])
    print(model_path)

    p = model_path[model_path.find('tune') + 5:model_path.find('tune') + 6]
    params_path = '../config/params_{}'.format(name)
    hps_list = pd.read_csv(params_path, sep='\t', header=0)
    hps_list = hps_list.to_dict('records')
    hps = hps_list[int(p)]

    params = vars(args)
    params['methods'] = methods
    params['type'] = type
    params['num_layer'] = num_layer
    params['input_dim'] = 300
    params['output_dim'] = num_class
    params = assign_params(params, hps)
    params['seed'] = SEED

    # dictionary = open(os.path.join('/data/project/yinhuapark/DATA_PRE', 'ohsumed', 'all_vocab.txt')).read().split()

    model = DocNet(params=params).to(params['device'])
    model.load_state_dict(torch.load(model_path))
    labels, preds = evaluate(test_loader)

    make_result(labels, preds)

    plot_imbalanced_result()



