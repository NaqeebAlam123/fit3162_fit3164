import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion_pretrain",
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dtw",
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--landmark",
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--actor",
                        type=str,
                        default='M030') ## NOTE M030
    parser.add_argument("--lr",
                        type=float,
                        default=0.0002)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--lambda1",
                        type=int,
                        default=100)
    parser.add_argument("--batch_size",
                        type=int,
                        default=16)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=100)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument('--device_ids', type=str, default='0')

    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrained_dir', type=str)
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')

    parser.add_argument('--triplet_margin', type=int, default=1)
    parser.add_argument('--triplet_weight', type=int, default=10)
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--pretrain_sep', type=bool, default=False) ## NOTE: pretrain seperate
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--use_triplet', type=bool, default=False)
    parser.add_argument('--rnn', type=bool, default=True)

    return parser.parse_args()

config = parse_args()
