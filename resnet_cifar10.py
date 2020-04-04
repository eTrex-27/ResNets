import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torch.backends.cudnn as cudnn
from models import model_dict

from LoaderDataset import cifar10_dataloaders

from utils import adjust_learning_rate
from train_test import train, test
from plot import ploting

def parser_params():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=164, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='82,123', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet20,resnet32,resnet44,resnet56,resnet110', choices=['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'], help='dataset')
    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    models = opt.model.split(',')
    opt.model = list([])
    for m in models:
        opt.model.append(m)

    return opt


loss_tr20, loss_tr32, loss_tr44, loss_tr56, loss_tr110 = [], [], [], [], []
list_loss_train = [loss_tr20, loss_tr32, loss_tr44, loss_tr56, loss_tr110]
loss_tt20, loss_tt32, loss_tt44, loss_tt56, loss_tt110 = [], [], [], [], []
list_loss_test = [loss_tt20, loss_tt32, loss_tt44, loss_tt56, loss_tt110]


def main(m):
    best_error = 100
    opt = parser_params()

    if opt.dataset == 'cifar10':
        train_loader, test_loader = cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
    else:
        raise NotImplementedError(opt.dataset)

    print(opt.model[m])
    model = model_dict[opt.model[m]](num_classes=n_cls)

    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    for epoch in range(1, opt.epochs + 1):

        if m == 4 and epoch == 1:
            opt.learning_rate = 0.01
        else:
            opt.learning_rate = 0.1

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        train_error, train_loss = train(epoch, train_loader, model, criterion, optimizer, list_loss_train[m])
        print('epoch {} | train_loss: {}'.format(epoch, train_loss))
        print('epoch {} | train_error: {}'.format(epoch, train_error))

        test_error, test_loss = test(test_loader, model, criterion, list_loss_test[m])
        print('epoch {} | test_loss: {}'.format(epoch, test_loss))
        print('epoch {} | test_error: {}'.format(epoch, test_error))
        print('iterations: {}'.format(epoch * len(train_loader)))

        if best_error > test_error:
            best_error = test_error

    print('Min error: ', best_error)


if __name__ == '__main__':
    for m in np.arange(5):
        main(m)
    ploting(list_loss_train, list_loss_test)
