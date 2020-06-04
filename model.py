# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Text_CNN(nn.Module):
    def __init__(self, wv, vocab_len, args):
        super(Text_CNN, self).__init__()
        self.args = args

        embedding_num = vocab_len
        embedding_dim = args.embed_dim
        feature_dim = args.filter_dim
        class_ = args.class_num
        kernel_size = args.filter_size
        dropout_rate = args.dropout
        max_sen_len = args.max_sen_len

        if args.pre_train_wv:
            self.embed = nn.Embedding(embedding_num, embedding_dim).from_pretrained(wv)
        else:
            self.embed = nn.Embedding(embedding_num, embedding_dim)
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, feature_dim, (h, embedding_dim)),
            nn.ReLU(),
            nn.MaxPool2d((max_sen_len - h + 1, 1)))
            for h in kernel_size
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(feature_dim * len(kernel_size), class_)

    def forward(self, x):
        x = x.permute(1, 0)
        embed_x = self.embed(x).unsqueeze(1)
        out = [conv(embed_x) for conv in self.convs]
        out = torch.cat(out, 1)
        out = out.view(-1, out.size(1))
        out = self.dropout(out)
        out = self.fc(out)
        return out
