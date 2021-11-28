#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import torch.nn.functional as F
import copy
from resnet import *


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.targets = dataset.targets
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        # self.loader = DataLoader(DatasetSplit(dataset, pre_idx), batch_size=len(pre_idx), shuffle=True)


    def train(self, net,vnet,feature,snet):

        net.train()
        optimizer = torch.optim.SGD(net.params(), self.args.lr,momentum=self.args.momentum, nesterov=self.args.nesterov,
                                      weight_decay=self.args.weight_decay)


        epoch_loss = []
        for ep in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                cost_w=F.cross_entropy(log_probs,labels,reduce=False)
                cost_v = torch.reshape(cost_w, (len(cost_w), 1))
                with torch.no_grad():
                    main_input = feature.index_select(0, labels.long())
                    w_new = vnet(main_input)
                    w_new1=snet(cost_v)
                    w_multi=w_new*w_new1

                norm_v = torch.sum(w_multi)
                if norm_v != 0:
                    w_v = w_multi / norm_v
                else:
                    w_v = w_multi
                l_f = torch.sum(cost_v * w_v)
                optimizer.zero_grad()
                l_f.backward()
                optimizer.step()
                batch_loss.append(l_f.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(),sum(epoch_loss) / len(epoch_loss)




def test_img(net_g, datatest, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.test_bs)
    l = len(data_loader)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct.item() / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg



