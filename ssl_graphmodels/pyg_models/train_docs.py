import sys, os
sys.path.append('/data/project/yinhuapark/scripts/models/ssl/ssl_graphmodels')

from comet_ml import Experiment
from comet_ml.experiment import InterruptedExperiment

#experiment = Experiment(
#    api_key= 'DiBTlYoVBF3vBHX605LM0y6Vk',
#    project_name = 'textgcn',
#    workspace='qkrdmsghk',
#)
import random
from config.conf import arg_config
from utils.LoadData import LoadDocsData
from models_docs import DocNet
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
use_gpu = torch.cuda.is_available()


def train(loader, training=True):
    total_loss = 0
    total_kl = 0
    if training:
        model.train()
    else:
        model.eval()
    for index, data in enumerate(loader):
        if training:
            optimizer.zero_grad()
        data = data.to(args.device)
        out_p, kl_term = model(data)#.to(args.device)
        y_p = data.y_p
        loss_p = multi_crit(out_p, y_p)

        total_loss += loss_p.item() + (kl_term.item())
        # print(loss_p.item(), (kl_term.item()))
        if training:
            loss_p.backward()
            optimizer.step()
    total_loss = total_loss / len(loader)
    torch.cuda.empty_cache()

    return total_loss


def test(loader):
    model.eval()
    nb_correct, nb_total = 0, 0
    labels = []
    preds = []
    for data in loader:
        data = data.to(args.device)
        with torch.no_grad():
            pred, _ = model(data)
            y = data.y_p.data
            nb_correct += (pred.data.argmax(dim=-1) == y).sum().item()
            nb_total += len(y)
            labels.append(y)
            preds.append(pred.data)
    labels = torch.cat(labels).detach().cpu().numpy()
    preds = torch.cat(preds).detach().cpu().numpy()

    return nb_correct / nb_total, labels, preds

def train_main(train_loader, val_loader, test_loader, patience):
    min_val_loss = 20
    max_val_acc = 0
    step = 0
    best_epoch = 0
    best_model = ''
    best_test_results = 0
    all_results = []
    with tqdm(total=args.epoch, bar_format='{desc}{n_fmt}/{total_fmt} |{bar}|{postfix}', ncols=80) as t:

        for epoch in range(args.epoch):
            t.set_description(desc='train and validate')

            train_loss = train(train_loader, training=True)
            val_loss = train(val_loader, training=False)
            val_results, _, _ = test(val_loader)

            if val_results >= max_val_acc:
                step = 0
                test_results, labels, preds = test(test_loader)
                max_val_acc = val_results
                best_model = model.state_dict()
                best_epoch = epoch
                best_test_results = test_results
                best_preds = preds

            elif val_results < max_val_acc:
                step += 1
            if step > patience and patience != -1:
                break

            all_results.append([epoch, train_loss, val_loss, val_results, test_results])
            #print('val_loss={:^7.3f};val_acc={:^7.3f};test_acc={:^7.3f}'.format(val_loss, val_results, test_results))
            #t.set_postfix_str('lr={:^7.6f}; val_loss={:^7.3f};test_acc={:^7.3f}'.format(scheduler.get_lr()[0], val_loss, test_results))
            t.set_postfix_str('val_loss={:^7.3f};val_acc={:^7.3f};test_acc={:^7.3f}'.format(val_loss, val_results, test_results))
            t.update()

            experiment.log_metric('val_loss', val_loss, step=epoch)
            experiment.log_metric('val_acc', val_results, step=epoch)
            experiment.log_metric('test_acc', test_results, step=epoch)


            experiment.log_metric('best_test_acc', best_test_results)
            experiment.log_metric('best_epoch', best_epoch)

    del train_loader, test_loader, val_loader, val_loss, train_loss
    return best_model, best_epoch, best_test_results, all_results, best_preds, labels

#
# def get_embeddings():
#     if args.type == "glove":
#         embeddings = np.load(os.path.join('/data/project/yinhuapark/DATA_PRE', args.name, 'train_glove_embedding.npy'))
#     else:
#         embeddings = np.array(open(os.path.join('/data/project/yinhuapark/DATA_PRE', args.name, 'train_vocab.txt')).read().split())
#
#     if args.pre_trained == 'False':
#         embeddings = torch.eye(embeddings.shape[0])
#     return embeddings


def get_model_name(tune_id):
    if args.pre_trained == 'False':
        pre_train=0
    else:
        pre_train=1
    model_name = 'tune_{}_w_pt_{}_num_layer_{}_aggr_{}'.format(
        tune_id, pre_train, args.num_layer, args.aggregate)

    return model_name

def get_hps(hps, methods, name, seed):
    if '20ng' in name:
        name = '20ng'
    params_path = '../config/params_{}'.format(name)
    hps_list = pd.read_csv(params_path, sep='\t', header=0)
    if hps != 'all':
        if 'top' in hps:
            base_methods = 'gnn_note_attn_mine_inter_all'
            dfs = []
            for f in list(filter(lambda x: x.find('all_results') == -1, os.listdir(os.path.join(args.result_output, name, base_methods)))):
                dfs.append(pd.read_csv(os.path.join(args.result_output, name, base_methods, f), sep='\t'))
            dfs = pd.concat(dfs).sort_values(by='test_acc', ascending=False)
            hps = int(hps.split('_')[1])
            if len(dfs) > hps:
                hps_id = dfs['model_id'].unique().tolist()[:hps]
            else:
                hps_id = dfs['model_id'].unique().tolist()

        elif ',' in hps:
            hps_id = list(map(int, list(filter(lambda x: len(x)>0, hps.split(',')))))

        hps_list = hps_list.iloc[hps_id, :]

    if  'new' in params_path:
        hps_list['model_id'] = hps_list.index + 100
    else:
        hps_list['model_id'] = hps_list.index

    hps_list = hps_list.to_dict('records')
    hps_list_= []
    for hps in hps_list:
        model_name = get_model_name(hps['model_id'])
        flag = True
        for exist in os.listdir(os.path.join(args.model_output, args.name, methods, str(SEED))):
            if model_name in exist:
                flag = False
        if flag:
            hps_list_.append(hps)
        else:
            print('existing results!')
    del hps_list, flag, model_name
    return hps_list_

def assign_params(params, dd):

    params['hidden_dim'] = int(dd['hidden_dimension'])
    params['dropout'] = float(dd['drop_out'])
    params['lr'] = float(dd['lr'])
    params['patience'] = int(dd['patience'])
    params['weight_decay'] = float(dd['weight_decay'])
    params['batch_size'] = int(dd['batch_size'])

    return params


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = arg_config()
    for SEED in tqdm(args.seed.split(',')):
        SEED = int(SEED)
        torch.set_num_threads(1)

        if torch.cuda.is_available() and int(args.gpu) >= 0:
            args.device = torch.device('cuda:'+ args.gpu)
            # torch.backends.cudnn.benchmark = True
            # torch.backends.cudnn.deterministic = True
            setup_seed(SEED)
            # torch.cuda.empty_cache()
        else:
            args.device = torch.device('cpu')

        loader = LoadDocsData(name=args.name, type=args.type)


        result_dfs = []

        methods = args.methods
        if args.type != "":
            methods += '_' + args.type

        if args.variant != "":
            methods += '_' + args.variant
        if args.threshold != "":
            methods += "_threshold_" + args.threshold
        if args.temperature != "":
            methods += "_temperature_" + args.temperature

        if os.path.exists(os.path.join(args.result_output, args.name, methods)):
            pass
        else:
            os.mkdir(os.path.join(args.result_output, args.name, methods))
            os.mkdir(os.path.join(args.model_output, args.name, methods))


        if os.path.exists(os.path.join(args.model_output, args.name, methods, str(SEED))):
            pass
        else:
            os.mkdir(os.path.join(args.model_output, args.name, methods, str(SEED) ))

        if args.hps == '-1':
            hps = vars(args)
            hps['model_id'] = -1
            hps_list = [hps]
        else:
            hps_list = get_hps(args.hps, methods, args.name, SEED)

        for i, hps in enumerate(hps_list[:]):
            experiment = Experiment(
                        api_key= 'DiBTlYoVBF3vBHX605LM0y6Vk',
                        project_name = 'textgcn',
                        workspace='qkrdmsghk',
                        )
            tune_id = hps['model_id']
            model_name = get_model_name(tune_id)
            print('{}/{}th random search : {}'.format(tune_id, len(hps_list[:])-1, model_name))

            train_loader, val_loader, test_loader, num_class = loader.get_train_test(batch_size=int(hps['batch_size']), seed=SEED)

            params = vars(args)
            params['input_dim'] = 300
            params['output_dim'] = num_class
            params = assign_params(params, hps)
            params['seed'] = SEED
            # params['vocab_size'] = len(loader.dictionary) + 1

            print(pd.DataFrame.from_dict(params, orient='index'))

            experiment.log_parameters(params)

            model = DocNet(params=params).to(params['device'])
            experiment.set_model_graph(str(model))

            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: hps['lr_decay'] ** epoch, last_epoch=-1)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

            multi_crit = torch.nn.CrossEntropyLoss()


            best_model, best_epoch, test_results, all_results, best_preds, labels = train_main(train_loader, val_loader, test_loader, params['patience'])

            experiment.log_confusion_matrix(labels, best_preds, title='Test Confusion Matrix')


            all_results = pd.DataFrame(all_results, columns=['epoch', 'train_loss', 'val_loss', 'val_acc', 'test_acc'])
            all_results.to_csv(os.path.join(params['result_output'], params['name'], methods, model_name+'_all_results'), sep='\t', index=False)

            print('best_epoch:{}; acc:{:.4f}'.format(best_epoch, test_results))
            result_df = pd.DataFrame([[tune_id, params['pre_trained'], params['num_layer'], params['aggregate'], best_epoch, test_results]],
                                     columns=['model_id', 'pre_train', 'num_layer', 'aggregation', 'converged_epoch', 'test_acc'])
            result_dfs.append(result_df)


            result_df.to_csv(os.path.join(params['result_output'], params['name'], methods, model_name), sep='\t', index=False)
            torch.save(best_model, os.path.join(params['model_output'], params['name'], methods, str(SEED), 'epoch_'+str(best_epoch)+'_'+model_name+'.pt'))

            del best_model, best_epoch, test_results, model

    # result_dfs = pd.concat(result_dfs).sort_values(by='test_acc', ascending=False)
    # result_dfs.to_csv(os.path.join(args.result_output, args.name, args.methods, 'test_results'), sep='\t')
