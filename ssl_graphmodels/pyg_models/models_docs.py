import sys, os
sys.path.append('../ssl_graphmodels')

import torch
from utils.ATGCConv_GCN import *
from utils.enhwa_utils import *
from layers_docs import READOUTLayer, DENSELayer, GRAPHLayer
use_gpu = torch.cuda.is_available()



class DocNet(torch.nn.Module):
    def __init__(self, params):
        super(DocNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.params = params
        self.func = params['methods']
        print('building {} model...'.format(self.func))
        self.type = params['type']

        if 'gnn' in self.func:
            getattr(self, 'gnn_build_bert')()
        else:
            getattr(self, self.func+'_build')()
        # print(self.layers)

    def gnn_build(self):
        self.layers.append(GRAPHLayer(self.params['input_dim'],
                                      self.params['hidden_dim'],
                                      dropout=self.params['dropout'],
                                      act=torch.nn.ReLU(),
                                      bias=True,
                                      func = self.func,
                                      num_layer=self.params['num_layer'],
                                      temperature=self.params['temperature'],
                                      threshold=self.params['threshold']
                                      ))
        self.layers.append(READOUTLayer(self.params['hidden_dim'],
                                        self.params['output_dim'],
                                        func=self.func,
                                        dropout=self.params['dropout'],
                                        act=torch.nn.ReLU(),
                                        aggr=self.params['aggregate']))

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
                                        act=torch.nn.ReLU(),
                                        aggr=self.params['aggregate']
                                        ))

    def forward(self, data, explain=False):
        if explain:
            exp_dict = None
            hidden_emb = None
        if self.func == 'gnn_note':
            output = data.x_n
            if explain:
                hidden_emb = self.layers[0](data.x_n, edge_index=data.edge_index_n, batch=data.x_n_batch)
            for layer in self.layers:
                output = layer(output, edge_index=data.edge_index_n, batch=data.x_n_batch, batch_n=data.batch_n)

        elif 'gnn_note_attn' in self.func:
            if 'inter' in self.type:
                if explain:
                    hidden_emb, exp_dict = self.layers[0](data.x_n, edge_index=data.edge_index_n, batch=data.x_n_batch, edge_mask=data.edge_mask, explain=explain)
                output = data.x_n
                for layer in self.layers:
                    output = layer(output, edge_index=data.edge_index_n, batch=data.x_n_batch, edge_mask=data.edge_mask)

        else:
            output = data.x_p
            if explain:
                hidden_emb = self.layers[0](data.x_p, edge_index=data.edge_index_p, batch=data.x_p_batch)
            for layer in self.layers:
                output = layer(output, edge_index=data.edge_index_p, batch=data.x_p_batch)


        if 'reg' in self.func:
            kl_term = torch.mean(torch.Tensor(self.layers[0].kl_terms))
        else:
            kl_term = torch.zeros(1)

        if explain:
            return output, hidden_emb, exp_dict
        else:
            return output, kl_term


