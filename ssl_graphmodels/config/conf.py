
import argparse


def arg_config():
    parse = argparse.ArgumentParser('TextSSL Parameters')

    parse.add_argument('--name', type=str, default='R8')
    parse.add_argument('--pre_trained', type=str, default='')
    parse.add_argument('--type', type=str, default='inter_all', help='')
    parse.add_argument('--tr_split', type=float, default=1.0, help='0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0')

    parse.add_argument('--hidden_dim', type=int, default=96, help='96, 256, 512')
    parse.add_argument('--dropout', type=float, default=0, help='0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0')
    parse.add_argument('--lr', type=float, default=1e-3, help='1e-4, 5e-4, 1e-3')
    parse.add_argument('--weight_decay', type=float, default='0', help='0, 0.001, 0.0001')
    parse.add_argument('--batch_size', type=int, default='16', help='16, 64, 128, 256')

    parse.add_argument('--methods', type=str, default='gnn_note_attn_gumbel',
                       help='gnn(WordCooc), gnn_note(Disjoint), gnn_note_attn_soft(Complete), '
                            'gnn_note_attn_gumbel(ours), gnn_note_attn_gumbel_reg(ours w/reg)')
    parse.add_argument('--num_layer', type=int, default='2', help='1,2,3,4')
    parse.add_argument('--aggregate', type=str, default='sum', help='sum, max, mean, attn')


    parse.add_argument('--threshold', type=float, default=0.5, help='0,0.5,0.7,1')
    parse.add_argument('--temperature', type=float, default=0.01, help='0, 0.1, 0.2, 0.5, 1')

    parse.add_argument('--gpu', type=str, default='0')
    parse.add_argument('--epoch', type=int, default=200)
    parse.add_argument('--patience', type=int, default=-1)
    parse.add_argument('--seed', type=int, default=123)

    args = parse.parse_args()


    return args