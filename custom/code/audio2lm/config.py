import argparse


DATASET_DIR = 'data/mfcc/M030'
MODEL_DIR = 'saved_models/EmotionNet/M030/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion_pretrain",
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dtw",
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--landmark",
                        default=False, action=argparse.BooleanOptionalAction)
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

    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--triplet_margin', type=int, default=1)
    parser.add_argument('--triplet_weight', type=int, default=10)
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--use_triplet', type=bool, default=False)
    parser.add_argument('--rnn', type=bool, default=True)

    return parser.parse_args()

config = parse_args()


# def parse_args_Lm_encoder():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--lr",
#                         type=float,
#                         default=0.0002)
#     parser.add_argument("--beta1",
#                         type=float,
#                         default=0.5)
#     parser.add_argument("--beta2",
#                         type=float,
#                         default=0.999)
#     parser.add_argument("--lambda1",
#                         type=int,
#                         default=100)
#     parser.add_argument("--batch_size",
#                         type=int,
#                         default=16)#192,96
#     parser.add_argument("--max_epochs",
#                         type=int,
#                         default=100)
#     parser.add_argument("--cuda",
#                         default=True)
#     parser.add_argument("--dataset_dir",
#                         type=str,
#                         default="../dataset_M003/")
#                         # default="/mnt/ssd0/dat/lchen63/grid/pickle/")
#                         # default = '/media/lele/DATA/lrw/data2/pickle')
#     parser.add_argument("--model_dir",
#                         type=str,
#                         #default="../model_M030_mouth_close/")
#                         default="../model_M030/")
#     parser.add_argument('--pretrained_dir',
#                         type=str,
#                         #default='/media/asus/840C73C4A631CC36/MEAD/ATnet_emotion/pretrain/M003/90_pretrain.pth'
#                         default='train/disentanglement/model_M030/99_pretrain.pth')
#     parser.add_argument('--atpretrained_dir', type=str,default='train/disentanglement/atnet_lstm_18.pth') ## NOTE: used if pretrain_sep = True
#     parser.add_argument('--serpretrained_dir', type=str,default='train/emotion_pretrain/model_M030/SER_99.pkl') ## NOTE: used if pretrain_sep = True
#     parser.add_argument('--device_ids', type=str, default='0')
#     parser.add_argument('--num_thread', type=int, default=0)
#     parser.add_argument('--weight_decay', type=float, default=4e-4)
#     parser.add_argument('--load_model', action='store_true')
#     parser.add_argument('--pretrain', type=bool, default=True) ## NOTE: 
#     parser.add_argument('--pretrain_sep', type=bool, default=False) ## NOTE: pretrain seperate
#     parser.add_argument('--pretrained_epoch', type=int)
#     parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
#     parser.add_argument('--rnn', type=bool, default=True)

#     return parser.parse_args()

# train_Lm_encoder_config = parse_args_Lm_encoder()
