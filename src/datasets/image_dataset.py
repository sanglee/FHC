#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/03/03 21:37
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : iamge_v8.py
# @Software  : PyCharm
import copy
from itertools import combinations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, random_split, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from utils import set_random_seed
import math

class ImageDataset(Dataset):

    def __init__(self, data, targets, transform=None):
        super(ImageDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        mode = None

        if len(img.shape) == 2:
            mode = 'L'
            img = Image.fromarray(img.astype(np.uint8), mode=mode)
        elif len(img.shape) == 3:
            img = Image.fromarray((img * 255).astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class ImageDatasetModule(object):

    def __init__(self, dataset_params):
        self.params = dataset_params
        # self.train_id_dataset = None
        # self.train_ood_dataset = None
        # self.val_id_dataset = None
        # self.val_ood_dataset = None
        # self.test_id_dataset = None
        # self.test_ood_dataset = None
        self.data_dict = dict()
        self.client_dict = dict()

        self.normalization_variables = None
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def repr(self):
        total_num = 0
        data_dict = self.data_dict
        client_data_dict = data_dict["client_data_dict"]

        for i, cli_idx in enumerate(client_data_dict.keys()):
            print("===============================")
            print(f"Client {i}")
            client_dict = client_data_dict[cli_idx]
            print("Train ", len(client_dict["train_X"]))
            print("Train - normal :", np.sum([True if target == 0 else False for target in client_dict["train_y"]]))
            print("Train - abnormal :", np.sum([True if target == 1 else False for target in client_dict["train_y"]]))

            print("Valid ", len(client_dict["valid_X"]))
            print("Valid - normal :", np.sum([True if target == 0 else False for target in client_dict["valid_y"]]))
            print("Valid - abnormal :", np.sum([True if target == 1 else False for target in client_dict["valid_y"]]))

            print("Test ", len(client_dict["test_X"]))
            print("Test - normal :", np.sum([True if target == 0 else False for target in client_dict["test_y"]]))
            print("Test - abnormal :", np.sum([True if target == 1 else False for target in client_dict["test_y"]]))

            total_num += client_data_dict[cli_idx]['train_X'].shape[0]
            total_num += client_data_dict[cli_idx]['valid_X'].shape[0]
            total_num += client_data_dict[cli_idx]['test_X'].shape[0]

        print("===============================================")

        print("Total Test ", len(data_dict["test_label_list"]))
        print("Total Test - normal ", np.sum([True if target == 0 else False for target in data_dict["test_label_list"]]))
        print("Total Test - abnormal ", np.sum([True if target == 1 else False for target in data_dict["test_label_list"]]))

        print("Total num : ", total_num)

    @property
    def train_dataloader(self):
        return None

    @property
    def val_dataloader(self):
        return None

    @property
    def test_dataloader(self):
        return None

    @classmethod
    def split_by_client(cls, dataset, num_client):
        num_data_per_client = int(len(dataset) / num_client)
        num_data_last_client = len(dataset) - (num_data_per_client * (num_client - 1))
        num_data_per_client_list = [num_data_per_client for _ in range(num_client - 1)] + [num_data_last_client]

        return random_split(dataset, num_data_per_client_list)

    @classmethod
    def create(cls, dataset_name, dataset_params):
        dataset_module = None

        if dataset_name.lower() == "mnist":
            dataset_module = MNISTDatasetModule(dataset_params=dataset_params)
        if dataset_name.lower() == "cifar10":
            dataset_module = CIFAR10DatasetModule(dataset_params=dataset_params)

        return dataset_module

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, *args, **kwargs):
        pass

    def split_dataset(self, data_list, target_list):
        is_shuffle = self.params["is_shuffle"]
        id_targets = self.params["id_targets"]
        ood_target = self.params["ood_target"]
        num_id_target_per_client = self.params["num_id_target_per_client"]
        train_ratio = self.params["train_ratio"]
        val_ratio = self.params["val_ratio"]
        ood_ratio_per_client = self.params["ood_ratio_per_client"]
        random_seed = self.params["random_seed"]
        data_portion = self.params["data_portion"]
        # num_clients = len(class_comb)

        set_random_seed(random_seed)
        print(f"Set random seed : {random_seed}")

        class_comb = list(combinations(id_targets, num_id_target_per_client))
        num_ood_clients = math.ceil(len(id_targets)/num_id_target_per_client)

        abnormal_client_class_idx_list = list()
        normal_client_class_idx_list = list()

        for comb_idx, idx_list in enumerate(np.array(class_comb)):
            is_able_insert = True

            for idx in idx_list:
                if idx in np.unique(np.array(abnormal_client_class_idx_list)):
                    is_able_insert = False

            if is_able_insert or comb_idx == (len(np.array(class_comb)) - 1):
                abnormal_client_class_idx_list.append(idx_list)
            else:
                normal_client_class_idx_list.append(idx_list)

        num_abnormal_clients = len(abnormal_client_class_idx_list)
        num_normal_clients = len(normal_client_class_idx_list)
        num_clients = num_normal_clients + num_abnormal_clients
        results = abnormal_client_class_idx_list + normal_client_class_idx_list
        results = [class_idx_list.tolist() for class_idx_list in results]

        # # Get class idx per clients
        # class_comb = list(combinations(id_targets, num_id_target_per_client))
        # num_clients = num_clients
        #
        # results = np.random.permutation(class_comb)
        # np.unique(results[:num_clients], return_counts=True)

        # shuffle
        if is_shuffle:
            indexes = np.arange(len(data_list))

            indexes = np.random.permutation(indexes)

            dataset = data_list[indexes]
            labels = target_list[indexes]
        else:
            dataset = data_list
            labels = target_list

        dataset = Subset(dataset, np.arange(round(data_portion * dataset.__len__())))
        labels = Subset(labels, np.arange(round(data_portion * labels.__len__())))

        # dataset = copy.deepcopy(dataset[:round(data_portion * len(dataset))])
        # labels = copy.deepcopy(labels[:round(data_portion * len(labels))])

        # Split train | test
        trainset = dataset[:int(train_ratio * len(dataset))]
        trainlabels = labels[:int(train_ratio * len(dataset))]
        testset = dataset[int(train_ratio * len(dataset)):]
        testlabels = labels[int(train_ratio * len(dataset)):]

        # class_comb = list(combinations(id_targets, num_id_target_per_client))
        # results = np.random.permutation(class_comb)
        results = results[:num_clients]

        classes, num_classes = np.unique(results, return_counts=True)

        # 각 sampling 에 필요한 class 개수만큼씩 trainset 으로부터 나누기
        total_class_data = []
        total_class_label = []

        for cls, num_cls in zip(classes, num_classes):
            cls_data = trainset[trainlabels == cls]
            cls_label = trainlabels[trainlabels == cls]
            cls_datas = []
            cls_labels = []

            for i in range(num_cls):
                if i < num_cls - 1:
                    cls_datas.append(cls_data[i * (len(cls_data) // num_cls):(i + 1) * (len(cls_data) // num_cls)])
                    cls_labels.append(cls_label[i * (len(cls_label) // num_cls):(i + 1) * (len(cls_label) // num_cls)])
                else:
                    cls_datas.append(cls_data[i * (len(cls_data) // num_cls):])
                    cls_labels.append(cls_label[i * (len(cls_label) // num_cls):])

            total_class_data.append(cls_datas)
            total_class_label.append(cls_labels)

        # print(np.sum(total_class_data[0][0]))

        # client 별로 나눠주기
        client_data = {}
        for client_idx, client_cls in enumerate(results):
            client_data[client_idx] = {}

            client_X = []
            client_y = []

            for cl in client_cls:
                idx = np.where(classes == cl)[0][0]
                client_X.append(total_class_data[idx].pop())
                client_y.append(total_class_label[idx].pop())

            client_data[client_idx]['train_X'] = np.concatenate(client_X)
            client_data[client_idx]['train_y'] = np.concatenate(client_y)

        # abnormal 처리
        abnormal_per_client_num = len(trainlabels[trainlabels == 1]) // num_clients

        train_abnormal_data = copy.deepcopy(trainset[trainlabels == 1]).tolist()
        train_abnormal_label = copy.deepcopy(trainlabels[trainlabels == 1]).tolist()

        for i in range(num_clients):
            train_abnormal_size = round(len(client_data[i]['train_X']) * ood_ratio_per_client)

            if train_abnormal_size == 0:
                train_abnormal_size = 1

            # ad = train_abnormal_data[i * abnormal_per_client_num:(i + 1) * abnormal_per_client_num]
            # adl = train_abnormal_label[i * abnormal_per_client_num:(i + 1) * abnormal_per_client_num]

            # ood 포함 클라이언트의 경우, abnormal append
            if i < num_ood_clients:
                cls_train_abnormal_data = train_abnormal_data[:train_abnormal_size]
                cls_train_abnormal_label = train_abnormal_label[:train_abnormal_size]

                client_data[i]['train_X'] = np.concatenate([client_data[i]['train_X'], cls_train_abnormal_data])
                client_data[i]['train_y'] = np.concatenate([client_data[i]['train_y'], cls_train_abnormal_label])

                del train_abnormal_data[:train_abnormal_size]
                del train_abnormal_label[:train_abnormal_size]

            # # 아니면 test에 추가
            # else:
            #     testset = np.concatenate([testset, ad])
            #     testlabels = np.concatenate([testlabels, adl])

        for cli in range(num_clients):
            client_data[cli]['train_y'] = list(
                map(lambda target: 0 if target != ood_target else 1, client_data[cli]['train_y']))

            idx = np.arange(len(client_data[cli]['train_y']))
            idx = np.random.permutation(idx)

            # permuted data
            train_y = np.array(client_data[cli]['train_y'])[idx]
            train_X = client_data[cli]['train_X'][idx]

            val_X = train_X[:round(len(train_X) * val_ratio)]
            val_y = train_y[:round(len(train_y) * val_ratio)]

            client_data[cli]['train_X'] = train_X[round(len(train_X) * val_ratio):]
            client_data[cli]['train_y'] = train_y[round(len(train_y) * val_ratio):]
            client_data[cli]['valid_X'] = val_X
            client_data[cli]['valid_y'] = val_y

            # 만약 valid 에 비정상 없으면 추가
            if np.sum(client_data[cli]['valid_y']) == 0 and cli < num_ood_clients:
                client_data[cli]['valid_X'] = np.concatenate([client_data[cli]['valid_X'], train_abnormal_data[:1]])
                client_data[cli]['valid_y'] = np.concatenate([client_data[cli]['valid_y'], train_abnormal_label[:1]])

                del train_abnormal_data[:1]
                del train_abnormal_label[:1]

        # test 합치고 train valid seperation -> 하나인 것 처럼 보이게 나누기 (?)
        testlabels = list(map(lambda target: 0 if target != 1 else 1, testlabels))

        testlabels = np.array(testlabels)

        normal_test_X = testset[testlabels == 0]
        normal_test_y = testlabels[testlabels == 0]

        abnormal_test_X = testset[testlabels == 1]
        abnormal_test_y = testlabels[testlabels == 1]

        test_abnormal_size = round(len(normal_test_X) * ood_ratio_per_client)

        if test_abnormal_size == 0:
            test_abnormal_size = 1

        abnormal_test_X = abnormal_test_X[:test_abnormal_size]
        abnormal_test_y = abnormal_test_y[:test_abnormal_size]

        total_test_X = np.concatenate([normal_test_X, abnormal_test_X])
        total_test_y = np.concatenate([normal_test_y, abnormal_test_y])

        # Test shuffling
        indexes = np.arange(len(total_test_X))

        indexes = np.random.permutation(indexes)

        total_test_X = total_test_X[indexes]
        total_test_y = total_test_y[indexes]

        test_data_list = copy.deepcopy(total_test_X)
        test_label_list = copy.deepcopy(total_test_y)

        for cli in range(num_clients):
            if cli < num_clients - 1:
                client_data[cli]['test_X'] = total_test_X[
                                             cli * round(len(total_test_X) / num_clients):(cli + 1) * round(
                                                 len(total_test_X) / num_clients)]
                client_data[cli]['test_y'] = total_test_y[
                                             cli * round(len(total_test_y) / num_clients):(cli + 1) * round(
                                                 len(total_test_y) / num_clients)]
            else:
                client_data[cli]['test_X'] = total_test_X[cli * round(len(total_test_X) / num_clients):]
                client_data[cli]['test_y'] = total_test_y[cli * round(len(total_test_y) / num_clients):]

        # Set class information
        print(results)
        for idx, data_dict in client_data.items():
            data_dict["class_index_list"] = results[idx]

        self.data_dict = dict(
            num_clients=num_clients,
            num_abnormal_clients=num_abnormal_clients,
            num_normal_clients=num_normal_clients,
            client_data_dict=client_data,
            test_data_list=test_data_list,
            test_label_list=test_label_list
        )

        return self.data_dict

    def convert_dataset(self, data_dict, client_idx):
        client_data_dict = data_dict["client_data_dict"][client_idx]

        train_X = client_data_dict["train_X"]
        train_y = client_data_dict["train_y"]
        valid_X = client_data_dict["valid_X"]
        valid_y = client_data_dict["valid_y"]
        test_X = client_data_dict["test_X"]
        test_y = client_data_dict["test_y"]

        normalization_variables_dict = self.calculate_normalization_variables(
            data_list=np.concatenate([train_X, valid_X]),
            target_list=np.concatenate([train_y, valid_y])
        )
        self.set_normalization_variables(
            normalization_variables_dict=normalization_variables_dict
        )
        normalization_variables = self.normalization_variables

        # Set transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*normalization_variables)
        ])

        return dict(
            train_dataset=ImageDataset(data=train_X, targets=train_y, transform=transform),
            val_dataset=ImageDataset(data=valid_X, targets=valid_y, transform=transform),
            test_dataset=ImageDataset(data=test_X, targets=test_y, transform=transform),
        )

    def convert_total_dataset(self, data_dict):
        test_X = data_dict["test_data_list"]
        test_y = data_dict["test_label_list"]

        normalization_variables_dict = self.calculate_normalization_variables(
            data_list=test_X,
            target_list=test_y
        )
        self.set_normalization_variables(
            normalization_variables_dict=normalization_variables_dict
        )
        normalization_variables = self.normalization_variables

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*normalization_variables)
        ])

        return ImageDataset(data=test_X, targets=test_y, transform=transform)

    def calculate_normalization_variables(self, data_list, target_list):
        # Default : (n, n, 3) images, plz cascading this method.
        normal_data_list = list()

        for data, target in zip(data_list, target_list):
            if target == 0:
                normal_data_list.append(data)

        data_list = np.array(normal_data_list)

        mean_arr = np.array([np.mean(data, axis=(0, 1)) for data in data_list])
        std_arr = np.array([np.std(data, axis=(0, 1)) for data in data_list])

        mean_list = np.mean(mean_arr, axis=0)
        std_list = np.mean(std_arr, axis=0)

        # If has one channel.
        if not isinstance(mean_list, np.ndarray) or not isinstance(std_list, np.ndarray):
            mean_list = [mean_list]
            std_list = [std_list]

        mean_list = list(map(lambda mean: float(mean), mean_list))
        std_list = list(map(lambda mean: float(mean), std_list))

        return dict(
            means=mean_list,
            stds=std_list,
        )

    def set_normalization_variables(self, normalization_variables_dict=None):
        mean_list = normalization_variables_dict["means"]
        std_list = normalization_variables_dict["stds"]

        print(f"Normalization variables are NOT exist")

        self.normalization_variables = [[mean for mean in mean_list], [std for std in std_list]]

        print(f"Complete to set normalization variables :\n"
              f" > Mean : {mean_list}\n"
              f" > Std : {std_list}")


class MNISTDatasetModule(ImageDatasetModule):
    SAVE_DIR_PATH = "/workspace/code/data/"

    def __init__(self, dataset_params):
        super().__init__(dataset_params=dataset_params)

    def prepare_data(self, *args, **kwargs):
        MNIST(root=MNISTDatasetModule.SAVE_DIR_PATH, train=True, download=True)
        MNIST(root=MNISTDatasetModule.SAVE_DIR_PATH, train=False, download=True)

    def setup(self, client_idx=None, *args, **kwargs):
        """
        Generate Train | Validation | Test
        """
        dataset_dict = dict()

        train_entire = MNIST(
            root=MNISTDatasetModule.SAVE_DIR_PATH,
            train=True,
            transform=self.transform
        )
        test_entire = MNIST(
            root=MNISTDatasetModule.SAVE_DIR_PATH,
            train=False,
            transform=self.transform
        )

        data_list = torch.cat(tensors=[train_entire.data, test_entire.data])
        target_list = torch.cat(tensors=[train_entire.targets, test_entire.targets])

        dataset_dict = self.split_dataset(data_list=data_list, target_list=target_list)

        return dataset_dict

    @property
    def train_dataset(self):
        return ConcatDataset(datasets=[self.train_id_dataset, self.train_ood_dataset])

    @property
    def val_dataset(self):
        return ConcatDataset(datasets=[self.val_id_dataset, self.val_ood_dataset])

    @property
    def test_dataset(self):
        return ConcatDataset(datasets=[self.test_id_dataset, self.test_ood_dataset])


class CIFAR10DatasetModule(ImageDatasetModule):
    SAVE_DIR_PATH = "/workspace/code/data/"

    def __init__(self, dataset_params):
        super().__init__(dataset_params=dataset_params)

    def prepare_data(self, *args, **kwargs):
        CIFAR10(root=CIFAR10DatasetModule.SAVE_DIR_PATH, train=True, download=True)
        CIFAR10(root=CIFAR10DatasetModule.SAVE_DIR_PATH, train=False, download=True)

    def setup(self, *args, **kwargs):
        """
        Generate Train | Validation | Test
        """
        train_entire = CIFAR10(root=CIFAR10DatasetModule.SAVE_DIR_PATH, train=True, transform=self.transform)
        test_entire = CIFAR10(root=CIFAR10DatasetModule.SAVE_DIR_PATH, train=False, transform=self.transform)

        data_list = torch.cat(tensors=[torch.tensor(train_entire.data), torch.tensor(test_entire.data)])
        target_list = torch.cat(tensors=[torch.tensor(train_entire.targets), torch.tensor(test_entire.targets)])

        return self.split_dataset(data_list=data_list, target_list=target_list)

    @property
    def train_dataset(self):
        return ConcatDataset(datasets=[self.train_id_dataset, self.train_ood_dataset])

    @property
    def val_dataset(self):
        return ConcatDataset(datasets=[self.val_id_dataset, self.val_ood_dataset])

    @property
    def test_dataset(self):
        return ConcatDataset(datasets=[self.test_id_dataset, self.test_ood_dataset])
