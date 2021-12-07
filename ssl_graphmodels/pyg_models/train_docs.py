import sys, os
sys.path.append('/data/project/yinhuapark/scripts/models/ssl/ssl_graphmodels')

from comet_ml import Experiment
from comet_ml.experiment import InterruptedExperiment

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
        out_p, kl_term = model(data)
        y_p = data.y_p
        loss_p = multi_crit(out_p, y_p)

        total_loss += loss_p.item() + (kl_term.item())
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
            t.set_postfix_str('val_loss={:^7.3f};val_acc={:^7.3f};test_acc={:^7.3f}'.format(val_loss, val_results, test_results))
            t.update()

    del train_loader, test_loader, val_loader, val_loss, train_loss
    return best_model, best_epoch, best_test_results, all_results, best_preds, labels


def get_model_name(tune_id):
    if args.pre_trained == 'False':
        pre_train=0
    else:
        pre_train=1
    model_name = 'tune_{}_w_pt_{}_num_layer_{}_aggr_{}'.format(
        tune_id, pre_train, args.num_layer, args.aggregate)

    return model_name


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
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            setup_seed(SEED)
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

        hps = vars(args)
        hps['model_id'] = -1
        hps_list = [hps]

        for i, hps in enumerate(hps_list[:]):
            tune_id = hps['model_id']
            model_name = get_model_name(tune_id)
            print('{}/{}th random search : {}'.format(tune_id, len(hps_list[:])-1, model_name))

            train_loader, val_loader, test_loader, num_class = loader.get_train_test(batch_size=int(hps['batch_size']), seed=SEED)

            params = vars(args)
            params['input_dim'] = 300
            params['output_dim'] = num_class
            params = assign_params(params, hps)
            params['seed'] = SEED

            print(pd.DataFrame.from_dict(params, orient='index'))
            model = DocNet(params=params).to(params['device'])
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            multi_crit = torch.nn.CrossEntropyLoss()

            best_model, best_epoch, test_results, all_results, best_preds, labels = train_main(train_loader, val_loader, test_loader, params['patience'])
            all_results = pd.DataFrame(all_results, columns=['epoch', 'train_loss', 'val_loss', 'val_acc', 'test_acc'])
            all_results.to_csv(os.path.join(params['result_output'], params['name'], methods, model_name+'_all_results'), sep='\t', index=False)

            print('best_epoch:{}; acc:{:.4f}'.format(best_epoch, test_results))
            result_df = pd.DataFrame([[tune_id, params['pre_trained'], params['num_layer'], params['aggregate'], best_epoch, test_results]],
                                     columns=['model_id', 'pre_train', 'num_layer', 'aggregation', 'converged_epoch', 'test_acc'])
            result_dfs.append(result_df)


            result_df.to_csv(os.path.join(params['result_output'], params['name'], methods, model_name), sep='\t', index=False)
            torch.save(best_model, os.path.join(params['model_output'], params['name'], methods, str(SEED), 'epoch_'+str(best_epoch)+'_'+model_name+'.pt'))

            del best_model, best_epoch, test_results, model