# -*- coding: utf-8 -*-

import argparse
from data_process import *
from model import Text_CNN
from train import train, plot

parser = argparse.ArgumentParser(description='Parse of Text-CNN')
parser.add_argument('-epochs', type=int, default=50, help='Number of epochs (default: 50)')
parser.add_argument('-batch_size', type=int, default=128, help='Batch size (default: 128)')
parser.add_argument('-lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
parser.add_argument('-dropout', type=float, default=0.5, help='Possibility of dropout (default: 0.5)')
parser.add_argument('-embed_dim', type=int, default=128, help='Embedding dimension (default: 128)')
parser.add_argument('-filter_dim', type=int, default=100, help='Number of filter (default: 100)')
parser.add_argument('-filter_size', type=str, default='3,4,5', help='Filter size (default: "3,4,5")')
parser.add_argument('-class_num', type=int, default=3, help='Number of class (default: 3)')
parser.add_argument('-pre_train_wv', type=bool, default=False,
                    help='Whether to use pre_trained word vector (default: False)')
parser.add_argument('-pre_train_name', type=str, default='xxx',
                    help='File name of pre_trained word vector (default: xxx)')
parser.add_argument('-max_sen_len', type=int, default=40, help='Max length of sentence')
parser.add_argument('-device', type=int, default=-1, help='device used for training, -1 represents cpu (default: -1)')
parser.add_argument('-split_rate', type=float, default=0.8, help='Rate of training data (default: 0.8)')
parser.add_argument('-save_model', type=bool, default=True, help='Whether to save the model (default: True)')
parser.add_argument('-interval', type=int, default=5, help='Number of batch to wait before logging status (default: 5)')
args = parser.parse_args()

print('Organizing dataset...')
train_num = split(args)

print('Loading dataset...')
iter_train, iter_validation, args.embed_dim, wv, vocab_len = dataset_load(args)
args.filter_size = [int(x) for x in args.filter_size.split(',')]
cuda = args.device != -1 and torch.cuda.is_available()

print('Training data...')
text_cnn = Text_CNN(wv, vocab_len, args)
print(text_cnn)
Loss, Acc, val_Loss, val_Acc = train(iter_train, iter_validation, text_cnn, cuda, train_num, args)
if args.save_model:
    torch.save(text_cnn, 'model/model.pkl')
plot(Loss, Acc, val_Loss, val_Acc, args)
