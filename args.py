import argparse
import sys


def get_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('ASTGN')
    parser.add_argument('--tasks', type=str, default="LP", choices=["LP", "NC"],
                        help='task name link prediction or node classification')
    parser.add_argument('--bandit', action='store_true', help='use bandit sampler or not')
    parser.add_argument('--valid_path', action='store_true', help='make sample path valid or not')
    parser.add_argument('--eta', type=float, default="0.4",help='the parameter of bandit sampler')
    parser.add_argument('--T', type=int, default="40", help='the parameter of bandit sampler')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch_size')
    parser.add_argument('--emb_dimension', type=int, default=172, help='emb_dimension')
    parser.add_argument('--time_dimension', type=int, default=172, help='dimension of time-encoding')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=30, help='Number of epochs')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
    parser.add_argument('--n_worker', type=int, default=0, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=-1, help='Random Seed')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--warmup', action='store_true', help='')
    parser.add_argument('--edge_feat_dim', type=int, default=172, help='Dimensions of the edge feature')

    try:
        args = parser.parse_args()
        assert args.n_worker == 0, "n_worker must be 0, etherwise dataloader will cause bug and results very bad performance (this bug will be fixed soon)"
        args.feat_dim = 172
        args.no_time = True
        # args.no_pos = True

    except:
        parser.print_help()
        sys.exit(0)

    return args

