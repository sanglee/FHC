#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/13 3:32 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : utils.py
# @Software  : PyCharm

import json
import os
import pickle

import numpy as np
import torch
import random
from sklearn.metrics import roc_auc_score
import logging

def make_logger(filename, name=None):
    if os.path.exists(filename):
        os.remove(filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s]: (%(levelname)s) %(name)s - %(message)s")
    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=filename)
    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


# Function for repeatability
def set_random_seed(seed_value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.random.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def not_null(df, col):
    unique_list = df[~df[col].isnull()][col].unique()
    return unique_list


def one_hot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot


def concatenate_columns(cols):
    conc = None

    for k in cols.keys():
        if len(cols[k].shape) < 2:
            conc = concatenate(conc, np.expand_dims(cols[k], 1), axis=1)
        else:
            conc = concatenate(conc, cols[k], axis=1)

    return conc


def not_null_df(df, col):
    return df[~df[col].isnull()]


def read_json(path):
    with open(path) as json_file:
        json_data = json.load(json_file)

    return json_data


# def read_json(file):
#     tweets = []
#     i = 0
#     for line in open(file, 'r'):
#         tweets.append(json.loads(line))
#         i += 1
#     df = pd.DataFrame(tweets)
#
#     return df


def save_data(state, directory, filename='inference_result.pkl'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    with open(filename, 'wb') as f:
        pickle.dump(state, f)


def load_data(file_path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
    return file


def join_features(dataset):  # 모든 feature를 하나로 unify
    all_features = []
    for d in dataset:
        for feat in list(d.keys()):
            if feat not in all_features:
                all_features.append(feat)

    return np.array(all_features)


def select_column_list(df, column_list):
    return df[[c for c in df.columns if c in column_list]]


def concatenate(array1, array2, axis=0):
    assert isinstance(array2, np.ndarray)
    if array1 is not None:
        assert isinstance(array1, np.ndarray)
        return np.concatenate((array1, array2), axis=axis)
    else:
        return array2

def roc_auc(y_true, y_score):
    return 100 * roc_auc_score(y_true=y_true, y_score=y_score)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count