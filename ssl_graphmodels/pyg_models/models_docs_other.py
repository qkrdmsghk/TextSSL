import sys, os
# sys.path.append('/data/project/yinhuapark/scripts/models/data_utils')
sys.path.append('/data/project/yinhuapark/scripts/models/ssl/ssl_graphmodels')

import torch
from utils.ATGCConv_GCN import *
from utils.enhwa_utils import *
from layers_docs import READOUTLayer, DENSELayer, GRAPHLayer, READOUT_cat_Layer
use_gpu = torch.cuda.is_available()



class DocNet(torch.nn.Module):
    def __init__(self, func, params):
        super(DocNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.params = params
        self.func = func
        print('building {} model...'.format(self.func))
        self.gnn_type = ''
        if self.func == 'gnn':
            getattr(self, 'gnn_build')()
        elif 'gnn' in self.func:
            self.gnn_type = self.func.split('_')[1]
            getattr(self, 'gnn_build')()
        elif 'cat' == self.func:
            self.aggr_ = 'cat'
            getattr(self, 'cat_build')()
        elif 'cat' in self.func:
            self.aggr_ = self.func.split('_')[1]
            getattr(self, 'cat_build')()
        else:
            getattr(self, self.func+'_build')()
        print(self.layers)

    def gnn_build(self):
        self.layers.append(GRAPHLayer(self.params['input_dim'],
                                      self.params['hidden_dim'],
                                      dropout=self.params['dropout'],
                                      act=torch.nn.ReLU(),
                                      bias=True,
                                      gnn_type = self.gnn_type,
                                      num_layer=self.params['num_layer']
                                      ))
        self.layers.append(READOUTLayer(self.params['hidden_dim'],
                                        self.params['output_dim'],
                                        func=self.func,
                                        dropout=self.params['dropout'],
                                        act=torch.nn.ReLU()
                                        ))
    def cat_build(self):
        self.layers.append(GRAPHLayer(self.params['input_dim'],
                                      self.params['hidden_dim'],
                                      dropout=self.params['dropout'],
                                      act=torch.nn.ReLU(),
                                      bias=True,
                                      num_layer=self.params['num_layer']
                                      ))
        self.layers.append(READOUT_cat_Layer(self.params['hidden_dim'],
                                        self.params['output_dim'],
                                        func=self.func,
                                        dropout=self.params['dropout'],
                                        aggr_ = self.aggr_,
                                        act=torch.nn.ReLU()
                                        ))

    def mlp_build(self):

        self.layers.append(DENSELayer(self.params['input_dim'],
                                      self.params['hidden_dim'],
                                      dropout=self.params['dropout'],
                                      act=torch.nn.ReLU(),
                                      bias=True
                                      ))
        self.layers.append(READOUTLayer(self.params['hidden_dim'],
                                        self.params['output_dim'],
                                        func=self.func,
                                        dropout=self.params['dropout'],
                                        act=torch.nn.ReLU()
                                        ))
    def docpr_all_build(self):
        self.layers.append(GRAPHLayer(self.params['input_dim'],
                                      self.params['hidden_dim'],
                                      dropout=self.params['dropout'],
                                      act=torch.nn.ReLU(),
                                      bias=True,
                                      num_layer=self.params['num_layer']
                                      ))
        self.layers.append(READOUTLayer(self.params['hidden_dim'],
                                        self.params['output_dim'],
                                        func=self.func,
                                        dropout=self.params['dropout'],
                                        act=torch.nn.ReLU()
                                        ))
    def docgr_only_build(self):
        self.layers.append(GRAPHLayer(self.params['input_dim'],
                                      self.params['hidden_dim'],
                                      dropout=self.params['dropout'],
                                      act=torch.nn.ReLU(),
                                      bias=True,
                                      num_layer=self.params['num_layer']
                                      ))
        self.layers.append(READOUTLayer(self.params['hidden_dim'],
                                        self.params['output_dim'],
                                        func=self.func,
                                        dropout=self.params['dropout'],
                                        act=torch.nn.ReLU()
                                        ))

    def docpr_only_build(self):
        self.layers.append(GRAPHLayer(self.params['input_dim'],
                                      self.params['hidden_dim'],
                                      dropout=self.params['dropout'],
                                      act=torch.nn.ReLU(),
                                      bias=True,
                                      num_layer=self.params['num_layer']
                                      ))
        self.layers.append(READOUTLayer(self.params['hidden_dim'],
                                        self.params['output_dim'],
                                        func=self.func,
                                        dropout=self.params['dropout'],
                                        act=torch.nn.ReLU()
                                        ))

    def d_docpr_all_build(self):
        self.layers.append(GRAPHLayer(self.params['input_dim'],
                                      self.params['hidden_dim'],
                                      dropout=self.params['dropout'],
                                      act=torch.nn.ReLU(),
                                      bias=True,
                                      num_layer=self.params['num_layer']
                                      ))
        self.layers.append(READOUTLayer(self.params['hidden_dim'],
                                        self.params['output_dim'],
                                        func=self.func,
                                        dropout=self.params['dropout'],
                                        act=torch.nn.ReLU()
                                        ))

    def d_docpr_only_build(self):
        self.layers.append(GRAPHLayer(self.params['input_dim'],
                                      self.params['hidden_dim'],
                                      dropout=self.params['dropout'],
                                      act=torch.nn.ReLU(),
                                      bias=True,
                                      num_layer=self.params['num_layer']
                                      ))
        self.layers.append(READOUTLayer(self.params['hidden_dim'],
                                        self.params['output_dim'],
                                        func=self.func,
                                        dropout=self.params['dropout'],
                                        act=torch.nn.ReLU()
                                        ))
    def d_docpr_cat_build(self):
        self.layers.append(GRAPHLayer(self.params['input_dim'],
                                      self.params['hidden_dim'],
                                      dropout=self.params['dropout'],
                                      act=torch.nn.ReLU(),
                                      bias=True,
                                      num_layer=self.params['num_layer']
                                      ))
        self.layers.append(READOUTLayer(self.params['hidden_dim'],
                                        self.params['output_dim'],
                                        func=self.func,
                                        dropout=self.params['dropout'],
                                        act=torch.nn.ReLU()
                                        ))

    def d_inv_docpr_all_build(self):
        self.layers.append(GRAPHLayer(self.params['input_dim'],
                                      self.params['hidden_dim'],
                                      dropout=self.params['dropout'],
                                      act=torch.nn.ReLU(),
                                      bias=True,
                                      num_layer=self.params['num_layer']
                                      ))
        self.layers.append(READOUTLayer(self.params['hidden_dim'],
                                        self.params['output_dim'],
                                        func=self.func,
                                        dropout=self.params['dropout'],
                                        act=torch.nn.ReLU()
                                        ))
    def d_inv_docpr_only_build(self):
        self.layers.append(GRAPHLayer(self.params['input_dim'],
                                      self.params['hidden_dim'],
                                      dropout=self.params['dropout'],
                                      act=torch.nn.ReLU(),
                                      bias=True,
                                      num_layer=self.params['num_layer']
                                      ))
        self.layers.append(READOUTLayer(self.params['hidden_dim'],
                                        self.params['output_dim'],
                                        func=self.func,
                                        dropout=self.params['dropout'],
                                        act=torch.nn.ReLU()
                                        ))
    def d_inv_docpr_cat_build(self):
        self.layers.append(GRAPHLayer(self.params['input_dim'],
                                      self.params['hidden_dim'],
                                      dropout=self.params['dropout'],
                                      act=torch.nn.ReLU(),
                                      bias=True,
                                      num_layer=self.params['num_layer']
                                      ))
        self.layers.append(READOUTLayer(self.params['hidden_dim'],
                                        self.params['output_dim'],
                                        func=self.func,
                                        dropout=self.params['dropout'],
                                        act=torch.nn.ReLU()
                                        ))



    def forward(self, data, explain=False):
        if 'note' in self.func:
            output = data.x_n
            if explain:
                hidden_emb = self.layers[0](data.x_n, edge_index=data.edge_index_n, batch=data.x_n_batch)

            for layer in self.layers:
                output = layer(output, edge_index=data.edge_index_n, batch=data.x_n_batch, batch_n=data.batch_n)

        elif 'docpr' in self.func:
            output = data.x_pr
            if self.func.startswith('d_'):
                edge_index = data.edge_index_pr_d
                if 'inv' in self.func:
                    edge_index = data.edge_index_pr_d_
            else:
                edge_index = data.edge_index_pr

            for layer in self.layers:
                output = layer(output, edge_index=edge_index, batch=data.x_pr_batch, x_pr_mask=data.x_pr_mask)

        elif 'docgr' in self.func:
            output = data.x_gr
            edge_index = data.edge_index_gr
            for layer in self.layers:
                output = layer(output, edge_index=edge_index, batch=data.x_gr_batch, x_gr_mask=data.x_gr_mask)

        elif 'cat' in self.func:
            x_p = self.layers[0](data.x_p, edge_index=data.edge_index_p, batch=data.x_p_batch)
            x_n = self.layers[0](data.x_n, edge_index=data.edge_index_n, batch=data.x_n_batch)
            output = self.layers[1](x_p=x_p, x_n=x_n, x_p_batch=data.x_p_batch, x_n_batch=data.x_n_batch)

        else:
            output = data.x_p
            for layer in self.layers:
                output = layer(output, edge_index=data.edge_index_p, batch=data.x_p_batch)

        if explain:
            return output, hidden_emb
        else:
            return output


