#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from sampling import  get_train_data,Divide_groups
from options import args_parser
from Update import *
from util import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from resnet import *
from picture import *
import time


if __name__ == '__main__':


    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)
    # log
    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
    TAG = 'exp/final_acc/fedren_{}_seed{}_epoch{}_imb{}_slr{}_{}'.format( args.dataset,args.seed, args.epochs,
                                                                                args.imb_factor,args.s_lr,current_time)
    logdir = f'runs/{TAG}' if not args.debug else f'runs2/{TAG}'
    writer = SummaryWriter(logdir)

    # load dataset and split users
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)


    train_groups, idx_to_meta,dict_per_cls,img_num_list = get_train_data(train_dataset,args.dataset, args)
    validloader = DataLoader(DatasetSplit(train_dataset, idx_to_meta), batch_size=len(idx_to_meta), shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False)
    dict_users = Divide_groups(train_dataset,train_groups, dict_per_cls,args.num_users,args)
    mkdirs("./runs/data_pic")
    mkdirs("./runs/lossclass_multi")


    model = ResNet32(args.num_classes).cuda(args.device)
    model.train()
    w_glob = model.state_dict()


    vnet = VNet(args.embedding_dim, 100, 1).to(args.device)
    vnet.train()
    feature=torch.randn(args.num_classes,args.embedding_dim).cuda()
    feature.requires_grad=True

    snet = VNet(1, 100, 1).to(args.device)
    snet.train()


    loss_train=[]
    w_epoch = []

    for epoch in range(args.epochs):


        w_locals, loss_locals = [], []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:

            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_users[idx])
            w, loss= local_model.train(net=copy.deepcopy(model).to(args.device),
                                    vnet=copy.deepcopy(vnet).to(args.device),
                                    feature=copy.deepcopy(feature),snet=copy.deepcopy(snet).to(args.device))

            w_locals.append(w)
            loss_locals.append(loss)

        # update global weights
        w_glob = FedAvg(w_locals)
        w_temp=FedAvg(w_locals)
        model.load_state_dict(w_glob)
        meta_vnet = VNet(args.embedding_dim, 100, 1).cuda()
        meta_vnet.load_state_dict(vnet.state_dict())
        meta_vnet.train()
        meta_snet = VNet(1, 100, 1).cuda()
        meta_snet.load_state_dict(snet.state_dict())
        meta_snet.train()

        optimizer_w = torch.optim.SGD(meta_vnet.params(), args.v_lr, momentum=args.momentum,
                                      nesterov=args.nesterov, weight_decay=args.weight_decay)
        optimizer_w.add_param_group({"params": feature, 'lr': args.v_lr})
        optimizer_s = torch.optim.SGD(meta_snet.params(), args.s_lr, momentum=args.momentum,
                                      nesterov=args.nesterov, weight_decay=args.weight_decay)




        for i in range(5):

            meta_model = ResNet32(args.num_classes).cuda()
            meta_model.load_state_dict(w_temp)
            meta_model.train()
            images_val, labels_val = next(iter(validloader))
            images, labels = images_val.to(args.device), labels_val.to(args.device)
            y_f_hat = meta_model(images)
            cost = F.cross_entropy(y_f_hat, labels, reduce=False)
            cost_v = torch.reshape(cost, (len(cost), 1))
            f_input=feature.index_select(0,labels.long())
            v_lambda1 = meta_vnet(f_input)
            v_lambda2 = meta_snet(cost_v)
            v_lambda=v_lambda1*v_lambda2
            norm_c = torch.sum(v_lambda)
            if norm_c != 0:
                v_lambda_norm = v_lambda / norm_c
            else:
                v_lambda_norm = v_lambda
            l_f_meta = torch.sum(cost_v * v_lambda_norm)
            meta_model.zero_grad()
            grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
            meta_lr = args.lr
            meta_model.update_params(lr_inner=meta_lr, source_params=grads)
            del grads
            y_g_hat = meta_model(images)
            l_g_meta = F.cross_entropy(y_g_hat, labels)
            optimizer_w.zero_grad()
            optimizer_s.zero_grad()
            l_g_meta.backward()
            optimizer_w.step()
            optimizer_s.step()
            w_temp=meta_model.state_dict()

        model.load_state_dict(w_glob)
        v_glob = meta_vnet.state_dict()
        vnet.load_state_dict(v_glob)
        s_glob = meta_snet.state_dict()
        snet.load_state_dict(s_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Train loss {:.3f}'.format(epoch+1, loss_avg))
        loss_train.append(loss_avg)
        writer.add_scalar('train_loss', loss_avg,epoch+1)
        test_acc, test_loss = test_img(model, test_dataset, args)
        writer.add_scalar('test_loss', test_loss, epoch + 1)
        writer.add_scalar('test_acc', test_acc, epoch + 1)

    # testing
    model.eval()
    acc_train, loss_train = test_img(model, train_dataset, args)
    acc_test, loss_test = test_img(model, test_dataset, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    writer.close()
