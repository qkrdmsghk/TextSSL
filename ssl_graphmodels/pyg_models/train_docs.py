import sys, os

Your_path = '/data/project/yinhuapark/ssl/'
sys.path.append(Your_path+'ssl_graphmodels')
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = arg_config()
    SEED = args.seed
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
    train_loader, val_loader, test_loader, num_class = loader.get_train_test(batch_size=args.batch_size, seed=SEED, tr_split=args.tr_split)
    params = vars(args)
    params['input_dim'] = 300
    params['output_dim'] = num_class
    params['seed'] = SEED
    print(pd.DataFrame.from_dict(params, orient='index'))
    model = DocNet(params=params).to(params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    multi_crit = torch.nn.CrossEntropyLoss()

    best_model, best_epoch, test_results, all_results, best_preds, labels = train_main(train_loader, val_loader, test_loader, params['patience'])
    # del best_model, best_epoch, test_results, model