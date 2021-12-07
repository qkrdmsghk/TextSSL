
import os
import json
import argparse

#
# def path_config(data_path):
#     with open(data_path, 'r', encoding='utf-8') as fin:
#         opts = json.load(fin)
#     print(opts)
#     return opts
# === hihi === #



def arg_config():
    parse = argparse.ArgumentParser('text-level gnn')

    parse.add_argument('--name', type=str, default='R8')
    parse.add_argument('--pre_trained', type=str, default='False')
    parse.add_argument('--type', type=str, default='inter_all', help='')
    parse.add_argument('--variant', type=str, default='', help='')
    parse.add_argument('--tr_split', type=float, default=1.0, help='0.001, 0.0001')
    parse.add_argument('--weight_decay', type=str, default='0', help='0, 0.001, 0.0001')
    parse.add_argument('--drop_out', type=str, default='0.5', help='0, 0.5')


    parse.add_argument('--methods', type=str, default='', help='gnn, gnn_note')
    parse.add_argument('--num_layer', type=int, default='2', help='1,2,3,4')
    parse.add_argument('--aggregate', type=str, default='sum', help='sum, max, mean, attn')


    parse.add_argument('--hps', type=str, default='-1', help='10')
    parse.add_argument('--threshold', type=str, default='', help='0,0.5, 1')
    parse.add_argument('--temperature', type=str, default='0.01', help='0,0.5, 1')


    parse.add_argument('--gpu', type=str, default='0')
    parse.add_argument('--epoch', type=int, default=200)
    parse.add_argument('--patience', type=int, default=-1)
    parse.add_argument('--seed', type=str, default='239874,123,456,789,194408,2222,3333,7778,1944,23')


    parse.add_argument('--model_output', type=str, default='/data/project/yinhuapark/model_save')
    parse.add_argument('--result_output', type=str, default='/data/project/yinhuapark/model_results')
    parse.add_argument('--evaluate_output', type=str,
                        default='/data/project/yinhuapark/scripts/models/mimic3benchmark/evaluation/')
    parse.add_argument('--tb_run_path', type=str, default='/data/project/yinhuapark/tb_run_results/models/mimic',
                        help="Directory where the tensorboardX should be stored.")

    parse.add_argument('--action', type=bool, default=False)
    parse.add_argument('--batch_size', type=int, default=1, help='batch size of source inputs')

    parse.add_argument('--lr', type=str, default='', help='1,2,3,4')

    args = parse.parse_args()


    return args