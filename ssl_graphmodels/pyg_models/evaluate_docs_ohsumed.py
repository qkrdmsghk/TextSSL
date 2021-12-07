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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


label_dict = {}
patient_dict = {}
p_id = 0
for y_id, y in enumerate(os.listdir('/data/project/yinhuapark/DATA_PRE/ohsumed/test_cooc')):
    label_dict[y_id] = y
    for p in os.listdir('/data/project/yinhuapark/DATA_PRE/ohsumed/test_cooc/{}'.format(y)):
        p_id += 1
        patient_dict[p_id] = p


def node_embedding_pca(data, hidden_emb=None, added_inter_edge=None):

    if hidden_emb == None:
        x = data.x_n[data.x_n_batch==0].detach().cpu().numpy()

    else:
        x = hidden_emb[data.x_n_batch==0].detach().cpu().numpy()
    id = data.x_n_id[data.x_n_batch==0].detach().cpu().numpy()
    pos = data.batch_n[data.x_n_batch==0].detach().cpu().numpy()

    if added_inter_edge != None:
        row, col = added_inter_edge
        row_id, col_id = data.x_n_id[row], data.x_n_id[col]
        row_id = row_id.detach().cpu().numpy()
        col_id = col_id.detach().cpu().numpy()


    pca = PCA(n_components=2)
    pca.fit(x, id)
    res = pca.transform(x)
    freq = pd.Series(id).value_counts()[id].reset_index()

    inter_id1 = freq[freq[0]>1].index.tolist()
    inter_id2 = freq[freq[0]>1]['index'].tolist()



    # sns.scatterplot(x=res[:, 0], y=res[:, 1], hue=freq[0])
    pos = list(map(str, pos))
    sns.scatterplot(x=res[:, 0], y=res[:, 1], hue=pos, palette='tab10')


    # for i1, i2 in zip(inter_id1, inter_id2):
    #     print(dictionary[i2])
    #     print(res[i1, :])
        # plt.text(x=res[i1, 0], y=res[i1, 1], s=dictionary[i2])
    plt.show()

    a = 0


def evaluate(loader):
    model.eval()
    nb_correct, nb_total = 0, 0
    labels = []
    preds = []
    for i, data in enumerate(tqdm(loader)):
        data = data.to(args.device)
        pred, hidden_emb = model(data, explain=True)
        y = data.y_p.data
        labels.append(y)
        preds.append(pred.data.argmax(-1))
        nb_correct += (pred.data.argmax(dim=-1) == y).sum().item()
        nb_total += len(y)

        if y==pred.data.argmax(-1) and patient_dict[i+1]=='0019537_s':

            # print(patient_dict[i+1], label_dict[y.item()])
            # plt.title('True_{}'.format(patient_dict[i+1]))
            node_embedding_pca(data, hidden_emb)
            # plt.title('True_{}'.format(patient_dict[i+1]))
            # node_embedding_pca(data, hidden_emb, added_inter_edge)

        else:
            print(patient_dict[i+1], label_dict[y.item()])
            # plt.title('Error_{}'.format(patient_dict[i+1]))
            # node_embedding_pca(data)
            # plt.title('Error_{}'.format(patient_dict[i+1]))
            # node_embedding_pca(data, hidden_emb)


    labels = torch.cat(labels).detach().cpu().numpy()
    preds = torch.cat(preds).detach().cpu().numpy()

    return labels, preds



def get_model_name(num_layer, aggr, seed):
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
    a['smaple'] = patient_dict.values()
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
    name = 'ohsumed'
    best_hp = 0
    final_model_output =  '/data/project/yinhuapark/final_models'
    # SEEDS = [456, 2222, 239874, 456, 2222, 239874, 239784]
    SEED = '6'
    num_layer = 2
    params_path = '../config/params_{}'.format(name)
    hps_list = pd.read_csv(params_path, sep='\t', header=0)
    hps = hps_list.iloc[best_hp, :].to_dict()
    # METHODS = ['gnn_note_attn_mine_lin', 'gnn_note_attn_mine', 'gnn_note']
    methods = 'gnn_note_attn_mine_lin'
    '''
    TODO: METHODS = ['gnn_note', 'gnn'] --> type = ''
          METHODS = ['gnn_note_attn_mine_lin?', 'gnn_note_attn_mine_lin_reg?'] --> temperature = 0.1, 1.0
          METHODS = ['best model with best temperature'] + training set rate [10%, 20%, 30%. 40%, 50%. 60%, ]?? paper./
    '''

    loader = LoadDocsData(name=name, type=args.type)

    SEED = int(SEED)
    torch.set_num_threads(1)
    if torch.cuda.is_available() and int(args.gpu) >= 0:
        args.device = torch.device('cuda:' + args.gpu)
        setup_seed(SEED)
        torch.cuda.empty_cache()
    else:
        args.device = torch.device('cpu')
    # methods = args.methods
    if args.type != "":
        methods += '_' + args.type
    if args.variant != "":
        methods += '_' + args.variant
    if args.threshold != "":
        methods += "_threshold_" + args.threshold
    if args.temperature != "":
        methods += "_temperature_" + args.temperature

    if os.path.exists(os.path.join(final_model_output, name)):
        pass
    else:
        os.mkdir(os.path.join(final_model_output, name))
    if os.path.exists(os.path.join(final_model_output, name, methods)):
        pass
    else:
        os.mkdir(os.path.join(final_model_output, name, methods))

    model_name = get_model_name(num_layer, args.aggregate, SEED)
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

    model_path = os.path.join(final_model_output, name, methods, model_name+'.pt')
    print(model_path)
    model = DocNet(params=params).to(params['device'])
    model.load_state_dict(torch.load(model_path))

    labels, preds = evaluate(test_loader)

    a = make_result(labels, preds)
    a.to_csv(os.path.join(final_model_output, name, methods, model_name+"_detail"), sep='\t', index=False)

    # plot_imbalanced_result()