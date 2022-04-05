#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/13 3:21 PM
# @Author    : Junhyung Kwon
# @Site      :
# @File      : HSCTrainer.py
# @Software  : PyCharm

import time
from copy import deepcopy

import numpy as np
import torch
from base import BaseTrainer, BaseNet
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from utils import AverageMeter, concatenate


class RHSCTrainerV2(BaseTrainer):
    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple,
                 weight_decay: float, device: str, verbose: bool, init_steps: int, cid: str, patience: int,
                 include_pert: bool, gamma: float, is_abnormal: bool, r_lr, c_lr,
                 radius_thresh: float, pert_steps: int, pert_step_size: float, pert_duration: int, val_set=0,
                 client_wise=False, update_c=False, lamb: float=0., return_best=True):
        super(RHSCTrainerV2, self).__init__(optimizer_name, lr, n_epochs, lr_milestones, weight_decay, device)
        self.r_lr = r_lr
        self.c_lr = c_lr

        self.val_set = val_set
        self.return_best = return_best
        self.is_abnormal = is_abnormal
        self.client_wise = client_wise
        self.update_c = update_c
        self.verbose = verbose
        self.lamb = lamb
        # print(self.verbose)

        self.patience = patience
        self.pert_steps = pert_steps
        self.pert_step_size = pert_step_size
        self.include_pert = include_pert

        self.init_steps = init_steps
        self.c_idx = cid
        self.pert_gamma = gamma  # DROCC perturbation upper bound
        self.pert_radius = radius_thresh  # DROCC perturbation lower bound
        self.pert_duration = pert_duration

        self.train_loss_list = []
        self.valid_loss_list = []
        self.R_hist = []
        self.c = None
        self.R = None

    def init_optim(self, net: BaseNet) -> optim.Optimizer:
        return getattr(optim, self.optimizer_name)(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def initialize_c(self, net, dataloader, eps=0.1):
        n_samples = 0
        net.eval()
        c = torch.zeros(net.rep_dim, device=self.device)

        for X, y in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)

            abnormal_idx_list = y.type(torch.bool)
            normal_X = X[~abnormal_idx_list]
            encoded = net(normal_X)

            n_samples += encoded.shape[0]
            c += torch.sum(encoded, dim=0)
        c /= n_samples
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c >= 0)] = eps

        if self.update_c:
            self.c = torch.tensor(c, requires_grad=True, device=self.device)
        else:
            self.c = c.detach()
        # self.c.requires_grad = True

    def init_radius(self, net, dataloader):
        # _, radiuses = self.get_low_confidence(net, dataloader)
        # print(radiuses)
        # r, _ = self.personalized_r_gamma(radiuses)
        self.R = torch.tensor([0.1], requires_grad=True, device=self.device)
        print('done radius init')

    def mean_radius(self, net, dataloader):
        net.to(self.device)
        net.eval()

        n_samples = 0
        r = 0

        for X, y in dataloader:
            X = X.to(self.device)

            encoded = net(X)

            dist = torch.sum((encoded - self.c) ** 2, dim=1)
            n_samples += encoded.shape[0]
            r += torch.sum(dist, dim=0).item()

        r /= n_samples
        return r

    def get_low_confidence(self, net, dataloader, portion=0.9):
        # deprecated
        radiuses = None
        norm_Xs = None
        net.to(self.device)
        net.eval()
        for X, y in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)

            abnormal_idx_list = y.type(torch.bool)
            normal_X = X[~abnormal_idx_list]
            # if self.c_idx == '{11EEC3EB-2A31-4B80-BF40-0ECC2BB53EE4}':
            #     print(normal_X)
            encoded = net(normal_X)

            dist = torch.sqrt(torch.sum((encoded - self.c) ** 2, dim=1))

            norm_Xs = concatenate(norm_Xs, normal_X.detach().cpu().numpy())
            radiuses = concatenate(radiuses, dist.detach().cpu().numpy())

        if radiuses is None:
            radiuses = np.array([0., 0.])

        sorted_idxs = radiuses.argsort()

        norm_Xs = norm_Xs[sorted_idxs]
        low_confidence_X = norm_Xs[round(len(norm_Xs) * portion):]
        # print(low_confidence_X)
        try:
            tense = torch.tensor(low_confidence_X, dtype=torch.float32)
        except Exception as e:
            print(e)

        # print('done low confidence')
        return tense, radiuses

    # deprecated
    def personalized_r_gamma(self, radiuses):
        radiuses.sort()
        r = radiuses[round(len(radiuses) * self.pert_radius)]
        r_max = np.max(radiuses)
        r_gamma = 1 + ((r_max - r) / r) * self.pert_gamma
        return r, r_gamma

    # deprecated
    def perturb(self, net, X, radiuses, num_gradient_steps=7, step_size=0.07):  # r=0.1, gamma=1.1
        net.to(self.device)
        net.eval()

        r, r_gamma = self.personalized_r_gamma(radiuses)

        z = net.encode(X)
        zeta = torch.randn(z.shape).to(self.device).detach().requires_grad_()
        z_adv_sampled = z + zeta

        for step in range(num_gradient_steps):
            zeta.requires_grad_()
            with torch.enable_grad():
                svdd_map = net.svdd_mapping(z_adv_sampled)
                pert_dist = torch.abs(svdd_map - self.c)
                loss = torch.sum(pert_dist)
                grad = torch.autograd.grad(loss, [zeta])[0]
                grad_flattened = torch.reshape(grad, (grad.shape[0], -1))
                grad_norm = torch.norm(grad_flattened, p=2, dim=1)

                for u in range(grad.ndim - 1):
                    grad_norm = torch.unsqueeze(grad_norm, dim=u + 1)
                if grad.ndim == 2:
                    grad_norm = grad_norm.repeat(1, grad.shape[1])
                if grad.ndim == 3:
                    grad_norm = grad_norm.repeat(1, grad.shape[1], grad.shape[2])
                grad_normalized = grad / grad_norm

            with torch.no_grad():
                zeta.add_(step_size * grad_normalized)

            if (step + 1) % 3 == 0 or step == num_gradient_steps - 1:
                norm_zeta = torch.sqrt(torch.sum(zeta ** 2, dim=tuple(range(1, zeta.dim()))))
                alpha = torch.clamp(norm_zeta, r, r * r_gamma).to(self.device)

                proj = (alpha / norm_zeta).view(-1, *[1] * (zeta.dim() - 1))
                zeta = proj * zeta

                z_adv_sampled = z + zeta

        return z_adv_sampled

    def load_pretrained_model(self, ae, net):
        encoder_state = ae.encoder.rnn_layer.state_dict()
        net.encoder.rnn_layer.load_state_dict(encoder_state)

    def train(self, dataset: DataLoader, validset: DataLoader, net: BaseNet):
        # print(self.verbose)

        start_time = time.time()
        # if self.pretrain_e:
        #     self.pretrain(dataset, validset, net, lr=0.001, weight_decay=1e-6)
        net.to(self.device)

        optimizer = self.init_optim(net)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        if self.verbose:
            print('Starting {} client training...'.format(self.c_idx))

        if self.c is None:
            if self.verbose:
                print('init c..')
            self.initialize_c(net, dataset)

        if self.R is None:
            print('init R..')
            self.init_radius(net, dataset)

        opt_r = optim.Adam([self.R], lr=self.r_lr)

        if self.update_c:
            opt_c = optim.Adam([self.c], lr=self.c_lr)

        best_net = None
        best_loss = -10
        best_loss2 = 999999
        best_R = 0
        best_c = None
        num_examples_train = 1

        print('start epoch')
        for epoch in range(self.n_epochs):
            train_dist = AverageMeter()
            train_loss = AverageMeter()
            valid_loss = AverageMeter()
            num_examples_train = 0

            net.train()
            self.R.requires_grad = True
            if self.update_c:
                self.c.requires_grad = True

            """
            Training Step
            """
            if self.client_wise:
                pert_bool = epoch >= self.init_steps
            else:
                pert_bool = epoch == self.init_steps

            dataloader = dataset

            if self.is_abnormal:
                for i, data in enumerate(dataloader):
                    X = data[0].to(self.device)
                    y = data[1].to(self.device)

                    svdd_out = net(X)

                    abnormal_idx_list = y.type(torch.bool)
                    normal_len = len(X[~abnormal_idx_list])
                    abnormal_len = len(X[abnormal_idx_list])
                    total_len = normal_len + abnormal_len

                    optimizer.zero_grad()
                    opt_r.zero_grad()

                    if self.update_c:
                        opt_c.zero_grad()

                    """
                    objective
                    """
                    if self.update_c:
                        self.c.requires_grad = True
                    num_examples_train += svdd_out.size(0)
                    dist = torch.sum((svdd_out - self.c) ** 2, dim=1)

                    self.R.requires_grad = False
                    losses = (1 - y) * ((dist - self.R) ** 2) - (y * torch.log(
                        1 - torch.exp(-((dist - self.R) ** 2)))) + self.lamb * torch.abs(self.R)
                    loss = torch.mean(losses)
                    train_dist.update(torch.mean(dist).item(), X.size(0))
                    train_loss.update(loss.item(), X.size(0))

                    loss.backward()
                    optimizer.step()
                    if self.update_c:
                        opt_c.step()

                    # r loss update
                    if self.is_abnormal:
                        if self.update_c:
                            self.c.requires_grad = False
                        self.R.requires_grad = True
                        r_loss = ((1 - y) * ((dist.data - self.R) ** 2)) - (y * torch.log(
                            1 - torch.exp(-((dist.data - self.R) ** 2)))) + self.lamb * torch.abs(self.R)
                        r_loss = torch.mean(r_loss)

                        r_loss.backward()
                        opt_r.step()

            if not self.is_abnormal:
                self.initialize_c(net, dataloader)
                self.R.data = self.mean_radius(net, dataloader)
            """
            validation step
            """
            net.eval()
            self.R.requires_grad = False
            if self.update_c:
                self.c.requires_grad = False

            y_list = None
            score_list = None

            for data in validset:
                X = data[0].to(self.device)
                y_list = concatenate(y_list, data[1].detach().cpu().numpy())

                svdd_out = net(X)

                # dist = torch.sqrt(torch.sum((svdd_out - self.c) ** 2, dim=1))
                dist = torch.sum((svdd_out - self.c) ** 2, dim=1)
                score_list = concatenate(score_list, dist.detach().cpu().numpy())
            f1 = 0

            if self.is_abnormal:
                pred = np.zeros_like(score_list)
                pred[score_list > self.R.detach().cpu().numpy()] = 1

                f1 = f1_score(y_list, pred)

            dist_mean = np.mean(score_list)

            if self.val_set == 2:
                self.initialize_c(net, dataset)  # update c
            if self.verbose:
                if self.is_abnormal:
                    print("%s Epoch %d \t| training dist: %.4f, training loss: %.8f, validation f1: %.8f, R: %.8f" % (
                        self.c_idx, epoch + 1, train_dist.avg, train_loss.avg, f1, self.R))
                else:
                    print("%s Epoch %d \t| training dist: %.4f, training loss: %.8f, validation dist: %.8f, R: %.8f" % (
                        self.c_idx, epoch + 1, train_dist.avg, train_loss.avg, dist_mean, self.R))

            self.train_loss_list.append(train_loss.avg)
            self.valid_loss_list.append(valid_loss.avg)
            self.R_hist.append(deepcopy(self.R.detach().cpu().numpy().tolist()))

            if self.is_abnormal:
                if f1 > best_loss:
                    if self.verbose:
                        print('best model updated')
                    last_update = epoch
                    best_loss = f1
                    best_net = deepcopy(net)
                    best_R = deepcopy(self.R.detach().cpu().numpy())
                    best_c = deepcopy(self.c)

            if best_loss2 > dist_mean and not self.is_abnormal:
                if self.verbose:
                    print('best model updated')
                last_update = epoch
                best_loss2 = dist_mean
                best_net = deepcopy(net)
                best_R = deepcopy(self.R.detach().cpu().numpy())
                best_c = deepcopy(self.c)

            scheduler.step()

            # if self.patience < epoch - last_update:
            #     print(f'Early stopping.. epoch{epoch}')
            #     print('Train duration: %.4f s'% (time.time() - start_time))
            #     break

        if self.return_best:
            return best_c, best_R, best_net, num_examples_train
        else:
            return deepcopy(self.c), self.R.detach().cpu().numpy(), deepcopy(net), num_examples_train


    def test(self, dataset: DataLoader, net: BaseNet, c=None, r=None):
        start_time = time.time()
        net.to(self.device)
        net.eval()

        scores = None
        labels = None

        num_examples_test = 0

        if c is None:
            c = self.c

        if r is None:
            r = self.R

        with torch.no_grad():
            for X, y in dataset:
                X = X.to(self.device)

                svdd_out = net(X)
                num_examples_test += X.size(0)
                dist = torch.sum((svdd_out - c) ** 2, dim=1)
                scores = concatenate(scores, dist.detach().cpu().numpy())
                labels = concatenate(labels, y.detach().cpu().numpy())

        return_type = 0
        if labels.sum() > 0:
            y_prob_pred = (scores >= r).astype(bool)
            obj = classification_report(labels, y_prob_pred, output_dict=True)
            test_auc = roc_auc_score(labels, scores)
            accuracy = accuracy_score(labels, y_prob_pred)

            print('Testing duration: %.4f s' % (time.time() - start_time))
        else:
            return_type = 1
            y_prob_pred = (scores >= r).astype(bool)
            obj = y_prob_pred.sum() / len(scores)
            test_auc = obj
            accuracy = accuracy_score(labels, y_prob_pred)

        return num_examples_test, obj, test_auc, accuracy, return_type


    def test_with_threshold(self, dataset: DataLoader, net: BaseNet, c=None, r=None):
        start_time = time.time()
        net.to(self.device)
        net.eval()

        scores = None
        labels = None

        num_examples_test = 0

        if c is None:
            c = self.c

        if r is None:
            r = self.R

        with torch.no_grad():
            for X, y in dataset:
                X = X.to(self.device)

                outputs = net(X)
                num_examples_test += X.size(0)

                # print(num_examples_test)
                dist = torch.sqrt(torch.sum((outputs - c) ** 2, dim=1))
                # print('test3')
                scores = concatenate(scores, dist.detach().cpu().numpy())
                # print('test4')
                labels = concatenate(labels, y.detach().cpu().numpy())
                # print('test5')

        return_type = 0
        if labels.sum() > 0:

            # fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
            # J = tpr - fpr
            # ix = np.argmax(J)
            # best_thresh = thresholds[ix]
            y_prob_pred = (scores >= r).astype(bool)
            obj = classification_report(labels, y_prob_pred, output_dict=True)
            test_auc = roc_auc_score(labels, scores)
            accuracy = accuracy_score(labels, y_prob_pred)

            print('Testing duration: %.4f s' % (time.time() - start_time))
        else:
            return_type = 1
            y_prob_pred = (scores >= r).astype(bool)
            obj = y_prob_pred.sum() / len(scores)
            test_auc = obj
            accuracy = accuracy_score(labels, y_prob_pred)

        return num_examples_test, obj, test_auc, accuracy, return_type
