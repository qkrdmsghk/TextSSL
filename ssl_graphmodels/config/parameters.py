import numpy as np
import random

import pandas as pd


param_grid = {
    'patience': list(range(20, 21)),
    'lr': list(np.logspace(np.log10(0.0005), np.log10(0.1), base=10, num=100)),
    'lr_decay': list(np.linspace(0.6, 1, num=8)),
    'weight_decay': [5e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3],
    'drop_out': [0.5, 0.6, 0.7, 0.8, 0.9],
    'batch_size': [64],
    'hidden_dimension': [128]
}


params_20ng = {
    'patience': [-1],
    'lr': [0.0005, 0.0001],
    'lr_decay': [1],
    'weight_decay': [0],
    'drop_out': [0.5],
    'hidden_dimension': [256, 512],
    'batch_size': [128, 256]
}
params_aclImdb = {
    'patience': [-1],
    'lr': [0.0001, 0.0005],
    'lr_decay': [1],
    'weight_decay': [0],
    'drop_out': [0.5],
    'hidden_dimension': [256, 512],
    'batch_size': [128, 256]
}

params_ohsumed = {
    'patience': [-1],
    'lr': [0.001],
    'lr_decay': [1],
    'weight_decay': [0],
    'drop_out': [0.5],
    'hidden_dimension': [256, 512],
    'batch_size': [128, 256]
}
params_R52 = {
    'patience': [-1],
    'lr': [0.001, 0.0005],
    'lr_decay': [1],
    'weight_decay': [0],
    'drop_out': [0.5],
    'hidden_dimension': [256, 512],
    'batch_size': [128, 256]
}
params_R8 = {
    'patience': [-1],
    'lr': [0.001, 0.0005],
    'lr_decay': [1],
    'weight_decay': [0],
    'drop_out': [0.5],
    'hidden_dimension': [96, 128],
    'batch_size': [64, 128]
}
params_mr = {
    'patience': [-1],
    'lr': [0.001, 0.0005],
    'lr_decay': [1],
    'weight_decay': [0],
    'drop_out': [0.5],
    'hidden_dimension': [96, 128],
    'batch_size': [64, 128]
}

# def save_parameters():
#     '''
#     random search
#     :return:
#     '''
#     MAX_EVALS = 10
#     dfs = []
#     for tune_id in range(MAX_EVALS):
#         np.random.seed(tune_id)
#         hps = {k: random.sample(v, 1) for k, v in param_grid_for_docs.items()}
#         dfs.append(pd.DataFrame.from_dict(hps))
#     dfs = pd.concat(dfs).reset_index(drop=True)
#     dfs.to_csv('parameters_for_tuning_docs_new', sep='\t', index=False)
#     print(dfs)

from sklearn.model_selection import ParameterGrid
def save_parameters():
    '''
    grid search
    :return:
    '''
    dataset = 'ohsumed'
    dfs = []
    grids = list(ParameterGrid(params_ohsumed))
    for grid in grids:
        print(pd.DataFrame.from_dict(grid, orient='index').T)
        dfs.append(pd.DataFrame.from_dict(grid, orient='index').T)
    dfs = pd.concat(dfs).reset_index(drop=True)
    dfs.to_csv('params_{}'.format(dataset), sep='\t', index=False)
    print(dfs)

save_parameters()



# hps_list = pd.read_csv('parameters_for_tuning_docs', sep='\t', header=0)
# hps_list = hps_list[(hps_list['drop_out']==0.8) & (hps_list['weight_decay']==0.00001)]
# hps_list = hps_list.to_dict('records')
# print(hps_list)
# import os
# df = pd.read_csv(os.path.join('../../mimic3benchmark/evaluation', 'MLP', 'aggr_sum'), sep='\t').sort_values(by='AUC of PRC_value', ascending=False)[:10]
#
# hps_list = hps_list[hps_list['drop_out'].isin(df['dropout'])]
# hps_list = hps_list[hps_list['weight_decay'].isin(df['weight_decay'])]
# hps_list['lr_decay_2'] = hps_list['lr_decay'].apply(lambda x: round(x, 2))
# hps_list['lr_4'] = hps_list['lr'].apply(lambda x: round(x, 4))
#
# hps_list = hps_list[hps_list['lr_decay_2'].isin(df['lr_decay'])]
# hps_list = hps_list[hps_list['lr_4'].isin(df['lr'])]


