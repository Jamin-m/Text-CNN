# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def train(iter_train, iter_validation, model, cuda, train_num, args):
    if cuda:
        torch.cuda.set_device(args.device)
        model = model.cuda()
    Loss, val_Loss = [], []
    Acc, val_Acc = [], []
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        i = 0
        print('Epoch: ', epoch)
        for batch in iter_train:
            text, label = batch.Text, batch.Label
            if cuda:
                text, label = text.cuda(), label.cuda()
            opt.zero_grad()
            predict = model(text)
            loss = F.cross_entropy(predict, label)
            loss.backward()
            opt.step()
            i += 1
            if i % args.interval == 0:
                correct = (torch.max(predict, 1)[1].view(label.size()).data == label.data).sum()
                acc = 100. * correct / batch.batch_size
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, accuracy: {:.6f}'.format(epoch, i * args.batch_size, train_num, 100. * i * args.batch_size / train_num, loss.item(), acc))
        Loss.append(loss.item())
        Acc.append(acc)
        val_loss, val_acc = eval(iter_validation, model, cuda)
        val_Loss.append(val_loss)
        val_Acc.append(val_acc)
    return Loss, Acc, val_Loss, val_Acc


def eval(iter_validation, model, cuda):
    model.eval()
    with torch.no_grad():
        # one batch
        for batch in iter_validation:
            text, label = batch.Text, batch.Label
            if cuda:
                text, label = text.cuda(), label.cuda()
            predict = model(text)
            val_Loss = F.cross_entropy(predict, label).item()
            correct = (torch.max(predict, 1)[1].view(label.size()).data == label.data).sum()
            val_Acc = 100. * correct / batch.batch_size
    return val_Loss, val_Acc


def plot(Loss, Acc, val_Loss, val_Acc, args):
    plt.figure()
    plt.plot(range(1, args.epochs + 1), Loss, 'r', linewidth=2, label='Training Loss')
    plt.plot(range(1, args.epochs + 1), val_Loss, 'b', linewidth=2, label='Validation Loss')
    plt.legend()
    plt.grid()
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('picture/loss.png')
    plt.figure()
    plt.plot(range(1, args.epochs + 1), Acc, 'r', linewidth=2, label='Training Acc')
    plt.plot(range(1, args.epochs + 1), val_Acc, 'b', linewidth=2, label='Validation Acc')
    plt.legend()
    plt.grid()
    plt.title('Training Acc')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.savefig('picture/acc.png')
