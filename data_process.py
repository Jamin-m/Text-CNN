# -*- coding: utf-8 -*-

import pandas as pd
import re
import jieba
import logging
jieba.setLogLevel(logging.INFO)
import torch
from torchtext import data
from torchtext.vocab import Vectors


def split(args):
    merge_bad = pd.read_table('data/bad.txt', header=None, encoding='utf-8')[:1000]
    merge_bad['label'] = 0
    merge_med = pd.read_table('data/medium.txt', header=None, encoding='utf-8')[:1000]
    merge_med['label'] = 1
    merge_good = pd.read_table('data/good.txt', header=None, encoding='utf-8')[:1000]
    merge_good['label'] = 2

    merge = pd.concat([merge_bad, merge_med, merge_good])
    merge = merge.sample(frac=1).reset_index(drop=True)

    split_pos = int(len(merge) * args.split_rate)
    merge[:split_pos].to_csv('data/train.csv', header=False)
    merge[split_pos:].to_csv('data/validation.csv', header=False)
    return split_pos


def word_div(sentence):
    regex = re.compile('[^\u4e00-\u9fa5a-zA-Z0-9]')
    sentence = regex.sub('', sentence)
    return [word.strip() for word in jieba.cut(sentence)]


def load_wv(name):
    return Vectors(name=name, cache='pretrain')


def dataset_load(args):
    text = data.Field(sequential=True, tokenize=word_div, fix_length=args.max_sen_len)
    label = data.Field(sequential=False, use_vocab=False)
    train, val = data.TabularDataset.splits(
        path='data/', train='train.csv',
        validation='validation.csv', format='csv',
        fields=[('Index', None), ('Text', text), ('Label', label)])
    if args.pre_train_wv:
        text.build_vocab(train, val, vectors=load_wv('pretrain/' + args.pre_train_name))
        args.embed_dim = text.vocab.vectors.size()[-1]
        wv = text.vocab.vectors
    else:
        text.build_vocab(train, val)
        wv = None

    iter_train, iter_validation = data.Iterator.splits(
        (train, val),
        batch_sizes=(args.batch_size, len(val)),
        sort_key=lambda x: len(x.Text),
    )
    return iter_train, iter_validation, args.embed_dim, wv, len(text.vocab)
