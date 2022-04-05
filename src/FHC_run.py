#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/21 11:49 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : FedAvg.py
# @Software  : PyCharm

import argparse
import multiprocessing as mp
import os
import shutil
import time
from abc import ABC
from collections import OrderedDict
from functools import reduce
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from datasets.image_dataset import MNISTDatasetModule, CIFAR10DatasetModule
# from datasets import MNISTDataModule, MNISTDatasetModule, CIFAR10DatasetModule  # ADDRepDataset
# from fl import weight_to_state_dict, state_dict_to_weight
from model import SVDD
from optimizer import RHSCTrainerV2
from utils import set_random_seed, make_logger, load_data

import math


# import torch
# from collections import OrderedDict

def weight_to_state_dict(keys, weights):
    params_dict = zip(keys, weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    return state_dict


def state_dict_to_weight(state_dict):
    return [values.cpu().numpy() for _, values in state_dict.items()]


class RHSCFedTraining(ABC):
    def __init__(self, loger, num_clients, num_abnormal_clients, num_normal_clients, num_rounds: int,
                 eval_rounds: int, trainer_params, model_params, descriptor_params, cuda_num_list,
                 abnormal_client_list=None, file_name='mnist', model_name='RHSC', abnormal_in_val=False,
                 save_path=None, rep_model='RNNAE', eval_device='cuda:0', personalization=True, global_init_steps=0):
        super(RHSCFedTraining, self).__init__()

        self.loger = loger
        self.global_init_steps = global_init_steps
        self.file_name = file_name
        self.save_path = save_path
        self.model_name = model_name
        self.abnormal_in_val = abnormal_in_val
        self.rep_model = rep_model
        self.cuda_num_list = cuda_num_list
        self.trainer_params = trainer_params
        self.personalization = personalization
        self.eval_rounds = eval_rounds
        self.abnormal_client_list = abnormal_client_list

        self.model_params = model_params
        self.eval_device = eval_device
        self.descriptor_params = descriptor_params

        self.num_clients = num_clients
        self.num_abnormal_clients = num_abnormal_clients
        self.num_normal_clients = num_normal_clients
        self.num_rounds = num_rounds

        self.current_rnd = 0
        self.g_weights = None
        self.g_c = None
        self.g_R = None
        self.keys = None
        self.initial_dict = None

        self.train_lr_curve_round = []
        self.valid_lr_curve_round = []
        self.train_lr_curve = []
        self.valid_lr_curve = []

        self._prepare_fit()

    def reset(self):
        self.current_rnd = 0
        self.g_weights = None
        self.g_c = None
        self.g_R = None
        self.keys = None
        self.initial_dict = None

        self._prepare_fit()

    def get_global(self):
        return self.weight_to_state_dict(self.g_weights), self.g_c

    def weight_to_state_dict(self, weights):
        params_dict = zip(self.keys, weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        return state_dict

    def state_dict_to_weight(self, state_dict):
        return [values.cpu().numpy() for _, values in state_dict.items()]

    def _prepare_fit(self):
        net = SVDD(**self.model_params)
        # print('prepare',net)
        self.keys = net.state_dict().keys()
        # print('prepare', net)
        self.initial_dict = self.state_dict_to_weight(net.state_dict())

    def weighted_aggregate(self, results) -> (list, list):
        # calculate total example numbers
        num_examples_total = sum([r['num_train_ex'] for r in results])

        num_clients = self.num_clients
        num_abnormal_clients = self.num_abnormal_clients
        num_normal_clients = self.num_normal_clients

        normal_cls_num = 2
        abnormal_cls_num = normal_cls_num + 1

        # num_abnormal_clients = 5
        # num_normal_clients = 31
        # num_clients = num_abnormal_cli + num_normal_cli

        abnormal_weight = abnormal_cls_num / (abnormal_cls_num + normal_cls_num) * (num_clients / num_abnormal_clients)
        normal_weight = normal_cls_num / (abnormal_cls_num + normal_cls_num) * (num_clients / num_normal_clients)

        is_abnormal_list = [r['is_abnormal'] for r in results]
        weight_list = [abnormal_weight if is_abnormal else normal_weight for is_abnormal in
                       is_abnormal_list]

        # weight_list = np.array(weight_list)

        # calculate weighted weight parameters
        weights = [(r['R'], r['c'], r['state_dict'], r['num_train_ex'], w) for r, w in zip(results, weight_list)]
        weighted_weights = [
            [layer * num_ex * w for layer in weights] for _, _, weights, num_ex, w in weights
        ]
        # calculate weighted center c
        weighted_c = [
            c * num_ex * w for _, c, _, num_ex, w in weights
        ]
        # weighted radius r
        weighted_R = [
            R * num_ex * w for R, _, _, num_ex, w in weights
        ]
        # Avg total weights
        weights_prime = [
            reduce(np.add, layer_updates) / num_examples_total for layer_updates in zip(*weighted_weights)
        ]
        # find global center
        c_prime = [
            reduce(np.add, c_updates) / num_examples_total for c_updates in zip(*weighted_c)
        ]
        # find global r
        r_prime = np.sum(weighted_R) / num_examples_total
        return weights_prime, c_prime, r_prime

    def aggregate(self, results) -> (list, list):
        # calculate total example numbers
        num_examples_total = sum([r['num_train_ex'] for r in results])

        # weight_list = np.array(weight_list)

        # calculate weighted weight parameters
        weights = [(r['R'], r['c'], r['state_dict'], r['num_train_ex']) for r in results]
        weighted_weights = [
            [layer * num_ex for layer in weights] for _, _, weights, num_ex in weights
        ]
        # calculate weighted center c
        weighted_c = [
            c * num_ex for _, c, _, num_ex in weights
        ]
        # weighted radius r
        weighted_R = [
            R * num_ex for R, _, _, num_ex in weights
        ]
        # Avg total weights
        weights_prime = [
            reduce(np.add, layer_updates) / num_examples_total for layer_updates in zip(*weighted_weights)
        ]
        # find global center
        c_prime = [
            reduce(np.add, c_updates) / num_examples_total for c_updates in zip(*weighted_c)
        ]
        r_prime = np.sum(weighted_R) / num_examples_total
        return weights_prime, c_prime, r_prime

    def get_file_name(self, file_name):
        return f"{file_name}_{self.model_name}"

    def get_save_path(self, rnd, file_name):
        self.save_dir_path = self.trainer_params[
            "model_save_path"] if "model_save_path" in self.trainer_params else f"../model_save/{self.save_path}"
        Path(self.save_dir_path).mkdir(exist_ok=True)
        return os.path.join(self.save_dir_path, f"{file_name}_{self.model_name}_{rnd}.pt")
        # f"{self.model_name}_gamma{self.descriptor_params['gamma']}_rep{self.rep_model}_{rnd}.pt")

    def round(self, rnd, trainset, validset, host_list, is_aug):
        global semaphore, return_dict
        start = time.time()
        self.descriptor_params['include_pert'] = False

        manager = mp.Manager()
        # semaphore = manager.list(self.cuda_num_list)
        # return_dict = manager.dict()
        # process_num = len(self.cuda_num_list)

        if self.global_init_steps < rnd:
            semaphore = manager.list(self.cuda_num_list)
            return_dict = manager.dict()
            process_num = len(self.cuda_num_list)
            # process_num = self.num_clients
            ts = trainset
            vs = validset
            hl = host_list
            abnormal_list = self.abnormal_client_list

        else:
            semaphore = manager.list(self.cuda_num_list[:np.sum(self.abnormal_client_list)])
            return_dict = manager.dict()
            process_num = int(np.sum(self.abnormal_client_list))
            trainset = np.array(trainset)
            validset = np.array(validset)
            host_list = np.array(host_list)

            # print(self.abnormal_client_list)
            # print(trainset)
            ts = trainset[self.abnormal_client_list.astype(bool)]
            vs = validset[self.abnormal_client_list.astype(bool)]
            hl = host_list[self.abnormal_client_list.astype(bool)]
            abnormal_list = [1 for _ in range(process_num)]

        p = Pool(processes=process_num)

        if rnd == 0:  # initialize state dict
            state_dict = self.initial_dict
            global_center = None
            global_r = None

        else:  # get from global state dict
            self.loger.info('retrieve global c, R')
            state_dict = self.g_weights
            global_center = self.g_c
            global_r = self.g_R

        if is_aug and rnd > self.descriptor_params['init_steps'] - 1:
            self.descriptor_params['include_pert'] = True

        self.loger.info(f'{global_center}, {global_r}, {len(ts)}, {len(vs)}, {hl}, {abnormal_list}')

        p.map(RHSC_client,
              zip(repeat(global_center),
                  repeat(global_r),
                  ts,
                  vs,
                  hl,
                  repeat(self.trainer_params),
                  repeat(self.model_params),
                  repeat(self.descriptor_params),
                  repeat(rnd),
                  repeat(state_dict),
                  abnormal_list)
              )
        p.close()
        p.join()

        results = list(return_dict.values())

        global_weight, global_c, global_r = self.aggregate(results)

        train_list = [res['train_list'] for res in results]
        valid_list = [res['valid_list'] for res in results]

        c_list = [res['c'] for res in results]

        self.loger.info(f"Round {rnd} time: {time.time() - start} s")
        manager.shutdown()

        return global_weight, np.array(global_c), global_r, train_list, valid_list, c_list

    def fit(self, trainset, validset, testset, host_list):
        self.loger.info('start training..')
        start = time.time()

        is_aug = self.descriptor_params['include_pert']

        best_auc = -9999
        best_acc = -9999
        best_f1 = -9999

        for rnd in range(self.num_rounds):
            g_weights, g_c, g_r, train_list, valid_list, c_list = self.round(self.current_rnd, trainset, validset,
                                                                             host_list,
                                                                             is_aug)

            # local_auc, local_f1, local_acc
            self.g_weights = g_weights
            self.g_c = g_c
            self.g_R = g_r
            self.train_lr_curve.append(train_list)
            self.valid_lr_curve.append(valid_list)
            self.train_lr_curve_round.append([np.mean(tl) for tl in train_list])
            self.valid_lr_curve_round.append([np.mean(vl) for vl in valid_list])

            self.loger.info(f"Global R: {self.g_R}, Global c: {self.g_c}")
            if math.isnan(self.g_R):
                break

            f1_results, auc_results, acc_results = self.eval_multi(validset, host_list, rnd)
            f1_mean, f1_std, auc_mean, auc_std, acc_mean, acc_std = np.mean(f1_results), np.std(f1_results), np.mean(
                auc_results), np.std(auc_results), np.mean(acc_results), np.std(acc_results)

            torch.save(dict(
                rnd=self.current_rnd,
                state_dict=self.g_weights,
                train_lr_curve=self.train_lr_curve,
                valid_lr_curve=self.valid_lr_curve,
                train_lr_curve_round=self.train_lr_curve_round,
                valid_lr_curve_round=self.valid_lr_curve_round,
                c=self.g_c,
                c_list=c_list,
                R=self.g_R,
                trainer_params=self.descriptor_params,
                val_results={"f1_mean": f1_mean, "f1_std": f1_std,
                             "auc_mean": auc_mean, "auc_std": auc_std, "acc_mean": acc_mean, "acc_std": acc_std}
            ), self.get_save_path(rnd, self.file_name))
            best_file = self.get_file_name(self.file_name)

            if best_f1 < f1_mean:
                best_f1 = f1_mean
                shutil.copy(self.get_save_path(rnd, self.file_name), f'../model_save/best_models/{best_file}_bestf1.pt')

            if best_auc < auc_mean:
                best_auc = auc_mean
                shutil.copy(self.get_save_path(rnd, self.file_name),
                            f'../model_save/best_models/{best_file}_bestauc.pt')

            if best_acc < acc_mean:
                best_acc = acc_mean
                shutil.copy(self.get_save_path(rnd, self.file_name),
                            f'../model_save/best_models/{best_file}_bestacc.pt')

            # save_data(results, self.save_dir_path, f"{self.file_name}_rnd{self.current_rnd}_result.pkl")

            self.current_rnd += 1

        self.loger.info(f'total training time: {time.time() - start}')

        # return results

    def resume_round(self, rnd=0, file_name=None):
        if file_name is None:
            file_name = self.file_name
        file_path = self.get_save_path(rnd, file_name)
        res = torch.load(file_path)
        self.current_rnd = rnd
        self.g_weights = res['state_dict']
        self.g_c = res['c']
        self.g_R = res['R']

    def eval(self, testset, host_list, rnd):
        args = (self.trainer_params,
                self.model_params,
                self.descriptor_params,
                self.g_weights,
                testset,
                host_list,
                self.abnormal_client_list,
                self.g_c,
                self.g_R
                )

        num_test, results, test_auc, acc, return_type = evaluation(args)
        return num_test, results, test_auc, acc, return_type

    def eval_multi(self, testset, host_list, rnd): # evaluate with val
        global eval_semaphore, eval_return_dict
        eval_manager = mp.Manager()

        eval_semaphore = eval_manager.list(self.cuda_num_list[:np.sum(self.abnormal_client_list)])
        eval_return_dict = eval_manager.dict()


        process_num = int(np.sum(self.abnormal_client_list))
        testset = np.array(testset)
        host_list = np.array(host_list)

        ts = testset[self.abnormal_client_list.astype(bool)]
        hl = host_list[self.abnormal_client_list.astype(bool)]
        abnormal_list = [1 for _ in range(process_num)]

        # print(process_num,len(ts), len(hl), len(abnormal_list))

        p = Pool(processes=process_num)

        p.map(multi_eval, zip(
            repeat(self.trainer_params),
            repeat(self.model_params),
            repeat(self.descriptor_params),
            repeat(self.g_weights),
            hl,
            abnormal_list,
            repeat(self.g_c),
            repeat(self.g_R),
            ts
        ))
        p.close()
        p.join()

        print(eval_return_dict)

        results = list(eval_return_dict.values())

        f1_results = [val['val_f1'] for val in results]
        auc_results = [val['val_auc'] for val in results]
        acc_results = [val['val_acc'] for val in results]

        # auc_list = [d['auc'] for d in results]
        # acc_list = [d['acc'] for d in results]
        self.loger.info(f"Local results \t| f1: {f1_results}, auc: {auc_results} acc: {acc_results}")
        self.loger.info(
            f"Total results \t| f1: {np.mean(f1_results)} {np.std(f1_results)}, auc: {np.mean(auc_results)} {np.std(auc_results)}, acc: {np.mean(acc_results)} {np.std(acc_results)}")

        eval_manager.shutdown()

        return f1_results, auc_results, acc_results


def multi_eval(args):
    trainer_params, model_params, descriptor_params, g_weights, hostid, is_abnormal, c, r, testset = args[0], args[1], \
                                                                                                     args[2], args[3], \
                                                                                                     args[4], args[5], \
                                                                                                     args[6], args[7], \
                                                                                                     args[8]
    if is_abnormal:
        testl = DataLoader(testset, **trainer_params['testloader'])
        gpu_idx = eval_semaphore.pop()
        device = 'cuda:%d' % gpu_idx if torch.cuda.is_available() else 'cpu'

        net = SVDD(**model_params)
        keys = net.state_dict().keys()
        global_state_dict = weight_to_state_dict(keys, g_weights)  # init global network
        net.load_state_dict(global_state_dict)  # load global state dict

        descriptor_params['device'] = device
        descriptor_params['cid'] = hostid
        descriptor_params['is_abnormal'] = is_abnormal

        trainer = RHSCTrainerV2(**descriptor_params)
        # trainer.c = torch.tensor(c).to(device)

        num_test, obj, test_auc, acc, return_type = trainer.test(testl, net, r=r, c=torch.tensor(c).to(device))  # return type = 1: normal acc

        eval_semaphore.append(gpu_idx)

        eval_return_dict[hostid] = dict(
            val_f1=obj['1']['f1-score'],
            val_auc=test_auc,
            val_acc=acc
        )


def evaluation(args):
    trainer_params, model_params, descriptor_params, g_weights, testset, hostid, is_abnormal, c, r = args[0], args[1], \
                                                                                                     args[2], args[3], \
                                                                                                     args[4], args[5], \
                                                                                                     args[6], args[7], \
                                                                                                     args[8]
    testl = DataLoader(testset, **trainer_params['testloader'])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = SVDD(**model_params)
    keys = net.state_dict().keys()
    global_state_dict = weight_to_state_dict(keys, g_weights)  # init global network
    net.load_state_dict(global_state_dict)  # load global state dict

    descriptor_params['device'] = device
    descriptor_params['cid'] = hostid[0]
    descriptor_params['is_abnormal'] = is_abnormal

    trainer = RHSCTrainerV2(**descriptor_params)
    trainer.c = torch.tensor(c).to(device)

    num_test, results, test_auc, acc, return_type = trainer.test(testl, net, c=c, r=r)  # return type = 1: normal acc

    return num_test, results, test_auc, acc, return_type


def evaluate(g_weights, testset, hostid, device, is_abnormal, c=None, r=None):
    print('Evaluation..')
    # print(self.g_weights)
    testl = DataLoader(testset, **trainer_params['testloader'])

    net = SVDD(**model_params)
    keys = net.state_dict().keys()
    global_state_dict = weight_to_state_dict(keys, g_weights)  # init global network
    net.load_state_dict(global_state_dict)  # load global state dict

    descriptor_params['device'] = device
    descriptor_params['cid'] = hostid
    descriptor_params['is_abnormal'] = is_abnormal

    trainer = RHSCTrainerV2(**descriptor_params)

    if not c is None:
        trainer.c = torch.tensor(c).to(device)

    num_test, obj, test_auc, return_type, acc = trainer.test(testl, net, r=r)  # return type = 1: normal acc

    return num_test, obj, test_auc, return_type


def RHSC_client(args):
    g_c, g_R, trainset, validset, hostid, trainer_params, model_params, descriptor_params, current_rnd, g_weights, is_abnormal = \
        args[0], \
        args[1], \
        args[2], \
        args[3], \
        args[4], \
        args[5], \
        args[6], \
        args[7], \
        args[8], \
        args[9], \
        args[10]

    gpu_idx = semaphore.pop()
    device = 'cuda:%d' % gpu_idx if torch.cuda.is_available() else 'cpu'

    if not is_abnormal:
        print(hostid, 'set epoch 1')
        # descriptor_params['n_epochs'] = 1
    trainer_params['valloader']['batch_size'] = validset.__len__()
    trainl = DataLoader(trainset, **trainer_params['trainloader'])
    validl = DataLoader(validset, **trainer_params['valloader'])

    print(f'hostid : {hostid}, device: {device}')

    net = SVDD(**model_params)
    # print(weights)
    keys = net.state_dict().keys()
    state_dict = weight_to_state_dict(keys, g_weights)

    net.load_state_dict(state_dict)  # load global state dict

    descriptor_params['device'] = device
    descriptor_params['cid'] = hostid
    descriptor_params['is_abnormal'] = is_abnormal
    if current_rnd > 0:
        descriptor_params['init_steps'] = 0

    trainer = RHSCTrainerV2(**descriptor_params)

    # update c with global center
    if not g_c is None:
        trainer.c = torch.tensor(g_c).to(device)
        if descriptor_params['update_c']:
            trainer.c.requires_grad = True

    if not g_R is None:
        trainer.R = torch.tensor(g_R, requires_grad=True, device=device)

    c, R, net, num_train = trainer.train(trainl, validl, net)
    R_hist = trainer.R_hist

    res = dict(
        state_dict=state_dict_to_weight(net.state_dict()),
        c=c.detach().cpu().numpy(),
        R=R,
        R_hist=R_hist,
        train_list=trainer.train_loss_list,
        valid_list=trainer.valid_loss_list,
        num_train_ex=num_train,
        hostid=hostid,
        is_abnormal=is_abnormal
        # report=obj,
        # auc=test_auc,
        # return_type=return_type,
        # num_test_ex=num_test,
    )

    semaphore.append(gpu_idx)
    return_dict[hostid] = res

    # return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Federated Learning Simulation based on Flower")
    parser.add_argument('--seed', type=int, default=999, help='random seed')
    parser.add_argument('--data_portion', '-dp', type=float, default=1.0, help='number of portion')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size for local update')
    parser.add_argument('--num_epochs', '-e', type=int, default=5, help='number of local epochs')
    parser.add_argument('--num_rounds', '-r', type=int, default=40, help='number of required rounds')
    parser.add_argument('--eval_device', '-ed', type=str, default='cuda:0', help='cuda device for evaluation')
    parser.add_argument('--weight_decay', '-w', type=float, default=0., help='weight decay')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--c_learning_rate', '-clr', type=float, default=5e-5, help='c learning rate')
    parser.add_argument('--r_learning_rate', '-rlr', type=float, default=5e-5, help='r learning rate')
    parser.add_argument('--init_steps', '-i', type=int, default=5, help='initial steps before augmentation')
    parser.add_argument('--global_init_steps', '-gi', type=int, default=5, help='initial steps before augmentation')
    parser.add_argument('--patience', '-p', type=int, default=10, help='patience for early stopping (no meaning)')
    parser.add_argument('--pert_steps', '-ps', type=int, default=10, help='augmentation perturbation steps')
    parser.add_argument('--pert_step_size', '-pss', type=float, default=0.001,
                        help='augmentation perturbation step sizes')
    parser.add_argument('--pert_duration', '-pd', type=int, default=2, help='augmentation duration (epochs)')
    parser.add_argument('--gamma', '-g', type=float, default=0.1, help='local gamma weight')
    parser.add_argument('--radius_thresh', '-rt', type=float, default=0.95,
                        help='threshold for local radius calculation')
    parser.add_argument('--test_portion', '-tp', type=float, default=0.8, help='portion of testset')
    parser.add_argument('--rep_model', '-rm', type=str, default="CAE", help='representation model')
    parser.add_argument('--abnormal_in_val', dest='abnormal_in_val', action='store_true')
    parser.add_argument('--include_pert', dest='include_pert', action='store_true')
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--resume_rnd', '-rr', type=int, default=-1, help='resume round')
    parser.add_argument('--eval_rnd', '-er', type=int, default=0, help='evaluation round per')
    parser.add_argument('--additional', '-ad', type=str, default='', help='additional file name')
    parser.add_argument('--update_c', dest='update_c', action='store_true')
    parser.add_argument('--return_best', dest='return_best', action='store_true')
    parser.add_argument('--lamb', '-l', type=float, default=0., help='lambda')
    parser.add_argument('--ood_ratio', '-or', type=float, default=0.025, help='portion of testset')
    parser.add_argument('--train_ratio', '-tr', type=float, default=0.2, help='tr')
    parser.add_argument('--val_set', '-v', type=int, default=0, help='validation setting (0: orig, 1: dist')
    parser.add_argument('--mode', '-m', type=str, default="mnist", help='dataset type')

    # ood_list = [0.1, 0.3]
    # train_ratio = [0.2, 0.3, 0.5]

    # loger = make_logger()

    args = parser.parse_args()
    save_path = f'fl3_{args.mode}'
    model_type = args.mode  # 'RNNEncoder

    if model_type == 'cifar10' or model_type == 'toniot':
        rep_dim = 128
    else:
        rep_dim = 32

    file_name = f'{args.mode}{args.num_epochs}'
    if args.additional != '':
        file_name = f"{args.mode}_{args.additional}_{args.num_epochs}_tr{args.train_ratio}or{args.ood_ratio}val{args.val_set}best{args.return_best}rlr{args.r_learning_rate}_clr{args.c_learning_rate}upc{args.update_c}"
        # file_name = f'{args.additional}{args.mode}{args.num_epochs}_tr{args.train_ratio}val{args.val_set}best{args.return_best}altlr{args.alt_learning_rate}'
        print(file_name)

    loger = make_logger(f'../logs/{file_name}.log')

    set_random_seed(args.seed)

    trainer_params = {
        'trainloader': dict(
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        ),
        'testloader': dict(
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        ),
        'valloader': dict(
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        ),
        'model_save_path': f'/workspace/code/FedHSC/model_save/{save_path}'
    }

    model_params = {
        'rep_model_type': model_type,
        'in_dim': 32,
        'hidden_dims': [rep_dim],  # hidden_dim, rep_dim
        'window_size': 128,
        'num_svdd_layer': 3,
        'num_rep_layer': 3
    }

    descriptor_params = {'optimizer_name': 'Adam', 'lr': args.learning_rate, 'n_epochs': args.num_epochs,
                         'lr_milestones': [20, 40], 'verbose': args.verbose, 'return_best': args.return_best,
                         'weight_decay': args.weight_decay, 'init_steps': args.init_steps, 'patience': args.patience,
                         'pert_steps': args.pert_steps, 'pert_step_size': args.pert_step_size, 'lamb': args.lamb,
                         'pert_duration': args.pert_duration, 'gamma': args.gamma, 'radius_thresh': args.radius_thresh,
                         'include_pert': args.include_pert, 'update_c': args.update_c, 'val_set': args.val_set,
                         'r_lr': args.r_learning_rate, 'c_lr': args.c_learning_rate}


    dataset_params = dict(
        is_shuffle=True,
        id_targets=[0, 2, 3, 4, 5, 6, 7, 8, 9],
        ood_target=1,
        num_id_target_per_client=2,
        train_ratio=0.1,
        val_ratio=0.2,
        ood_ratio_per_client=args.ood_ratio,
        data_portion=args.data_portion,
        random_seed=999
    )

    if model_type == 'mnist':
        datamodule = MNISTDatasetModule(dataset_params=dataset_params)
    elif model_type == 'cifar10':
        datamodule = CIFAR10DatasetModule(dataset_params=dataset_params)

    if model_type == 'mnist' or model_type == 'cifar10':
        datamodule.prepare_data()
        data_dict = datamodule.setup()

        trainset = []
        validset = []

        # data_dict['num_clients']
        num_clients =  data_dict['num_clients'] # 36
        abnormal_clients = data_dict['num_abnormal_clients'] #5
        normal_clients = data_dict['num_normal_clients'] #31

        for client_idx in range(num_clients):
            # client_idx = 0

            dataset_dict = datamodule.convert_dataset(
                data_dict=data_dict,
                client_idx=client_idx
            )

            train_dataset = dataset_dict["train_dataset"]
            val_dataset = dataset_dict["val_dataset"]

            trainset.append(train_dataset)
            validset.append(val_dataset)

        total_test_dataset = datamodule.convert_total_dataset(
            data_dict=data_dict
        )
    else:
        data_file = '/workspace/data/add/data/ton/ton_iot_data_dict_v8.pickle'
        res = load_data(data_file)

        num_clients = res['num_clients']  # 36
        abnormal_clients = res['num_abnormal_clients']  # 5
        normal_clients = res['num_normal_clients']  # 31
        data_dict = res['client_data_dict']

        trainset = []
        validset = []

        for client in range(res['num_clients']):
            train_set = TensorDataset(torch.from_numpy(data_dict[client]['train_X']).float(),
                                      torch.from_numpy(data_dict[client]['train_y']).int())
            valid_set = TensorDataset(torch.from_numpy(data_dict[client]['valid_X']).float(),
                                      torch.from_numpy(data_dict[client]['valid_y']).int())
            trainset.append(train_set)
            validset.append(valid_set)

        total_test_dataset = TensorDataset(torch.from_numpy(res['test_data_list']).float(),
                                      torch.from_numpy(res['test_label_list']).int())

    abnormal_client_list = np.array([0 if i > abnormal_clients - 1 else 1 for i in range(num_clients)])

    fl_trainer = RHSCFedTraining(loger,
                                 num_clients=num_clients,
                                 num_abnormal_clients=abnormal_clients,
                                 num_normal_clients=normal_clients,
                                 save_path=save_path,
                                 num_rounds=args.num_rounds,
                                 eval_rounds=args.eval_rnd,
                                 rep_model=args.rep_model,
                                 abnormal_client_list=abnormal_client_list,
                                 abnormal_in_val=args.abnormal_in_val,
                                 trainer_params=trainer_params,
                                 model_params=model_params,
                                 file_name=file_name,
                                 descriptor_params=descriptor_params,
                                 cuda_num_list=[0, 0, 1, 1, 2, 2, 3, 3],
                                 eval_device=args.eval_device,
                                 global_init_steps=args.global_init_steps)

    if args.resume_rnd > 0:
        fl_trainer.resume_round(args.resume_rnd)

    host_list = [f'client{i + 1}' for i in range(num_clients)]

    fl_trainer.fit(trainset, validset, total_test_dataset, host_list)