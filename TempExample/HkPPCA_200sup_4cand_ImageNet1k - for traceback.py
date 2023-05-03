#!/usr/bin/env python
# coding: utf-8

### imports
import random
import time
from typing import Union, Any

import winsound

import hdf5storage
import numpy as np
import torch
from scipy.io import loadmat
from scipy.io import savemat

import os

import matplotlib.pyplot as plt
import math
from collections import OrderedDict

from torch import device
from torch.utils.data import DataLoader, Dataset

from sklearn import metrics


### Functions

## Simulated Dataset
class MixtureDataset(Dataset):
    def __init__(self, values, labels, local_id=None):
        self.values = values
        self.labels = labels
        self.len = len(self.labels)
        self.original_id = [i for i in range(self.len)]
        self.local_id = local_id
        self.d = self.values.shape[1]

        self.dict = OrderedDict.fromkeys(set(self.labels))
        for i in range(len(self.values)):
            if self.dict[self.labels[i]] is None:
                self.dict[self.labels[i]] = [self.values[i]]
            else:
                self.dict[self.labels[i]].append(self.values[i])
        for i in self.dict.keys():
            self.dict[i] = torch.stack(self.dict[i])

    def __getitem__(self, index):
        values = self.values[index]
        labels = self.labels[index]
        original_id = self.original_id[index]
        return values, labels, original_id

    def __len__(self):
        return self.len


def IMAGENET_Feature(feature_addresses, cls_range=(0, 1000)):
    folder_label_map = {}
    y_train = []
    y_val = []
    x_train = torch.empty((0,))
    x_val = torch.empty((0,))
    N_tr = 0
    N_val = 0
    for path in feature_addresses:
        for sub_path in os.listdir(path):
            path_datafolder = os.path.join(path, sub_path)
            if sub_path == 'train':
                # build a folder-label map first
                classes_list = os.listdir(path_datafolder)
                for global_class_id, filename in enumerate(classes_list):
                    folder_label_map[filename] = global_class_id

                classid = 0
                local_id_trainset = []
                for local_class_id, filename in enumerate(classes_list[cls_range[0]:cls_range[1]]):
                    if classid % 100 == 0:
                        print(classid)
                    path_1datafile = os.path.join(path_datafolder, filename)
                    data1 = loadmat(path_1datafile)
                    x_train = torch.cat((x_train, torch.tensor(data1['feature'], dtype=torch.float)), dim=0)
                    n, dim = data1['feature'].shape
                    N_tr += n
                    local_id_1cls = [localid for localid in range(n)]
                    local_id_trainset.extend(local_id_1cls)

                    # y_train.extend(list(np.repeat(data1['label'], n))) # for mat has label
                    # file_label_map[file] = data1['label'].item()
                    cls = cls_range[0] + local_class_id
                    y_train.extend(list(np.repeat(cls, n)))  # for mat does not have label
                    classid += 1

            if sub_path == 'val':
                classid = 0
                classes_list = os.listdir(path_datafolder)
                for local_class_id, filename in enumerate(classes_list[:cls_range[1]]):
                    if classid % 200 == 0:
                        print(classid)
                    path_1datafile = os.path.join(path_datafolder, filename)
                    data1 = loadmat(path_1datafile)
                    x_val = torch.cat((x_val, torch.tensor(data1['feature'])), dim=0)
                    n, dim = data1['feature'].shape
                    N_val += n
                    y_val.extend(list(np.repeat(folder_label_map[filename], n)))
                    classid += 1

    train_set = MixtureDataset(x_train, y_train, local_id_trainset)
    val_set = MixtureDataset(x_val, y_val)
    return train_set, val_set, folder_label_map


def CIFAR100_Feature(feature_addresses, cls_range=(0, 100)):
    N_lab = 0
    N_unlab = 0
    start_cls, end_cls = cls_range
    for path in feature_addresses:
        for sub_path in os.listdir(path):
            path_data = os.path.join(path, sub_path)
            if 'train' in sub_path:
                data1 = loadmat(path_data)
                x_train = torch.flatten(torch.from_numpy(data1['feature']), start_dim=1)
                n, dim = x_train.shape
                N_lab += n
                y_train = data1['label'].flatten()
                x_train = x_train[(start_cls <= y_train) * (y_train < end_cls)]
                y_train = y_train[(start_cls <= y_train) * (y_train < end_cls)].tolist()
            if 'val' in sub_path:
                data1 = loadmat(path_data)
                x_val = torch.flatten(torch.from_numpy(data1['feature']), start_dim=1)
                n, dim = x_val.shape
                N_unlab += n
                y_val = data1['label'].flatten()
                x_val = x_val[y_val < end_cls]
                y_val = y_val[y_val < end_cls].tolist()

    train_set = MixtureDataset(x_train, y_train)
    val_set = MixtureDataset(x_val, y_val)
    return train_set, val_set


# MPPCA functions

def logdet_cov(N, S_cls, d, lda):
    k = len(S_cls)
    diag = lda * torch.eye(d)
    diag[:k, :k] = torch.diag(S_cls ** 2 / N + lda)  # commented when k=0
    logdetcov = torch.logdet(diag)
    return logdetcov.clone()


def logdet_cov_D2(D2_cls, d, lda):
    q = len(D2_cls)
    diag = lda * torch.eye(d)
    diag[:q, :q] = torch.diag(D2_cls + lda)  # commented when k=0
    logdetcov = torch.logdet(diag)
    return logdetcov.clone()


def deltaDiag(N, L_cls, S_cls, lda):
    d2 = S_cls ** 2 / N
    diagM = torch.diag(d2 / (lda * (d2 + lda)))  # k x k
    delta = L_cls.t() @ diagM @ L_cls  # d x d
    return delta.clone()


def deltaDiag_D2(L_cls, D2_cls, lda):
    d2 = D2_cls
    diagM = torch.diag(d2 / (lda * (d2 + lda)))  # k x k
    delta = L_cls.t() @ diagM @ L_cls  # d x d
    return delta.clone()


def score(x, mu_cls, delta_cls, lda, t=1):
    # x: n x d
    # mu_cls: d
    # delta_cls: d x d
    xc = x - mu_cls
    Xt = xc.unsqueeze(-2)  # n x 1 x d
    X = xc.unsqueeze(-1)  # n x d x 1
    # score = (Xt @ X)/t  #k=0, n x 1 x 1
    # score = (Xt @ X / lda)/t  #k=0, n x 1 x 1
    score = (Xt @ X / lda - Xt @ delta_cls @ X) / t  # n x 1 x 1
    dist = Xt @ X
    return score.flatten().clone(), dist.flatten().clone()


# def score_temp(x, mu_cls, delta_cls, lda, t=1):
#     # x: n x d
#     # mu_cls: d
#     # delta_cls: d x d
#     xc = x - mu_cls
#     Xt = xc.unsqueeze(-2)  # n x 1 x d
#     X = xc.unsqueeze(-1)  # n x d x 1
#     score = (Xt @ X) / t  # k=0, n x 1 x 1
#     # score = (Xt @ X / lda)/t  #k=0, n x 1 x 1
#     # score = (Xt @ X / lda - Xt @ delta_cls @ X) / t  # n x 1 x 1
#     return score.flatten()


# def score_dist_temp(x, mu_cls, L_cls, S_cls):
#     n = x.shape[0]  # x:nxd
#     xc = x - mu_cls
#     X = xc.unsqueeze(-1)  # nxdx1
#     dists = torch.zeros((20, n), device=device)
#     ones = torch.ones((n, 20, 1), device=device)
#     # (ones-L_cls@X)
#     for col in range(20):
#         v0 = L_cls[col]  # d
#         vt = v0.unsqueeze(-2)  # 1xd
#         # v=v0.unsqueeze(-1) #dx1
#         # vvt=(v@vt).repeat(n,1,1) #nxdxd
#         # dist_vec=(I-vvt/(vt@X))@X
#         ones = torch.ones((n, 1, 1), device=device)
#         dist_vec = (ones - 1 / (vt @ X)) * X
#         dists[col] = torch.linalg.norm(dist_vec, dim=(1, 2))
#     dist = torch.norm(dists, dim=0)
#     return dist.flatten()


def log_posterior_x_i_2(logdet_si_1cls, score_1cls, d):
    # i represent id of one cluster
    # mu_i: 1 x d
    # L_i: k x d
    # S_i: k
    # x: N x d
    # return prob_post: Nx1

    # log_prob_post = (-1 / 2) * score_1cls
    log_prob_post = (-d / 2) * np.log(2 * np.pi) + (-1 / 2) * logdet_si_1cls + (-1 / 2) * \
                    score_1cls  # torch.sum((x - mu_i) @ si_inv_i * (x - mu_i), 1)

    return log_prob_post.squeeze()


def prediction_online(x_val, y_val, nc, num_cls, mu, pi, L, S, lda, t, device):
    d = x_val.shape[1]
    N_val = x_val.shape[0]
    pi_online = pi.clone()
    mu_online = mu.clone()
    L_online = L.clone()
    S_online = S.clone()
    num_batches = 10
    y_pred = torch.empty(0, device=device)

    logdet_si = torch.zeros((num_cls, 1), device=device)
    delta = torch.zeros((num_cls, d, d), device=device)
    for j in range(num_cls):
        logdet_si[j] = logdet_cov(nc[j], S_online[j], d, lda)
        delta[j] = deltaDiag(nc[j], L_online[j], S_online[j], lda)

    size_batch = int(N_val / num_batches)
    valset = MixtureDataset(x_val, y_val)
    val_loader = DataLoader(valset, batch_size=size_batch, shuffle=False, drop_last=False)
    for id_batch, data in enumerate(val_loader):
        val_values, val_labels = data[0], list(data[1].numpy())
        x_batch = val_values.to(device)
        # print(id_batch, x_batch.shape)

        num_all_batch = x_batch.shape[0]
        ### Task: EM for PPCA2 - mix labeled and unlabeled together
        log_ppi_post_x_i = torch.zeros((num_cls, num_all_batch), device=device)
        for j in range(num_cls):
            score_X_j = score(x_batch, mu_online[j], delta[j], lda, t)
            log_ppi_post_x_i[j, :] = torch.log(pi_online[j, :]) + log_posterior_x_i_2(logdet_si[j], score_X_j, d)  # N

        y_pred_batch = torch.argmax(log_ppi_post_x_i, dim=0)
        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

        del log_ppi_post_x_i, y_pred_batch
        torch.cuda.empty_cache()

    return y_pred


def prediction_online_s(x_val, y_val, nc, num_cls, mu, L, S, lda, t, device, num_batches=10):
    d = x_val.shape[1]
    N_val = x_val.shape[0]

    mu_online = mu.clone()
    L_online = L.clone()
    S_online = S.clone()
    # num_batches = 10
    y_pred = torch.empty(0, device=device)
    loss = torch.tensor(0.0, device=device)

    delta = torch.zeros((num_cls, d, d), device=device)
    for j in range(num_cls):
        delta[j] = deltaDiag(nc[j], L_online[j], S_online[j], lda)

    size_batch = int(N_val / num_batches)
    valset = MixtureDataset(x_val, y_val)
    val_loader = DataLoader(valset, batch_size=size_batch, shuffle=False, drop_last=False)
    for id_batch, data in enumerate(val_loader):
        val_values, val_labels = data[0], list(data[1].numpy())
        x_batch = val_values.to(device)
        # print(id_batch, x_batch.shape)

        num_all_batch = x_batch.shape[0]
        ### Task: EM for PPCA2 - mix labeled and unlabeled together
        score_X = torch.zeros((num_cls, num_all_batch), device=device)
        dist_X = torch.zeros((num_cls, num_all_batch), device=device)
        for j in range(num_cls):
            score_X[j, :], dist_X[j, :] = score(x_batch, mu_online[j], delta[j], lda, t)

        y_pred_batch = torch.argmin(score_X, dim=0)
        y_pred_batch.int()

        for i in range(num_all_batch):
            # loss += dist_X[y_pred_batch[i], i]
            loss += score_X[y_pred_batch[i], i]

        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

        del score_X, y_pred_batch, dist_X
        torch.cuda.empty_cache()

    return y_pred.clone(), loss.clone()


def prediction_online_s_train(train_loader, nc, num_cls, mu, L, S, lda, t, device):
    # d = 640
    # d = 768
    # d=1024
    d = mu.shape[1]
    # N_val = x_val.shape[0]

    mu_online = mu.clone()
    L_online = L.clone()
    S_online = S.clone()
    # num_batches = 10
    y_pred = torch.empty(0, device=device)
    loss = torch.tensor(0.0, device=device)

    delta = torch.zeros((num_cls, d, d))
    for j in range(num_cls):
        delta[j] = deltaDiag(nc[j], L_online[j], S_online[j], lda).cpu()

    for id_batch, data in enumerate(train_loader):
        train_values, train_labels = data[0], list(data[1].numpy())
        x_batch = train_values.to(device)
        # print(id_batch, x_batch.shape)

        num_all_batch = x_batch.shape[0]
        ### Task: EM for PPCA2 - mix labeled and unlabeled together
        score_X = torch.zeros((num_cls, num_all_batch), device=device)
        dist_X = torch.zeros((num_cls, num_all_batch), device=device)
        for j in range(num_cls):
            delta_j = delta[j].to(device).clone()
            score_X[j, :], dist_X[j, :] = score(x_batch, mu_online[j], delta_j, lda, t)

        y_pred_batch = torch.argmin(score_X, dim=0)
        y_pred_batch.int()

        for i in range(num_all_batch):
            # loss += dist_X[y_pred_batch[i], i]
            loss += score_X[y_pred_batch[i], i]

        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

        del score_X, y_pred_batch, dist_X
        torch.cuda.empty_cache()

    return y_pred.clone(), loss.clone()


def prediction_online_D2(x_val, y_val, num_cls, mu, L, D2, lda, t, device, num_batches=10):
    d = x_val.shape[1]
    N_val = x_val.shape[0]

    mu_online = mu.clone()
    L_online = L.clone()
    D2_online = D2.clone()
    # num_batches = 10
    y_pred = torch.empty(0, device=device)
    loss = torch.tensor(0.0, device=device)

    delta = torch.zeros((num_cls, d, d), device=device)
    for j in range(num_cls):
        # delta[j] = deltaDiag(nc[j], L_online[j], S_online[j], lda)
        delta[j] = deltaDiag_D2(L_online[j], D2_online[j], lda)

    size_batch = int(N_val / num_batches)
    valset = MixtureDataset(x_val, y_val)
    val_loader = DataLoader(valset, batch_size=size_batch, shuffle=False, drop_last=False)
    for id_batch, data in enumerate(val_loader):
        val_values, val_labels = data[0], list(data[1].numpy())
        x_batch = val_values.to(device)
        # print(id_batch, x_batch.shape)

        num_all_batch = x_batch.shape[0]
        ### Task: EM for PPCA2 - mix labeled and unlabeled together
        score_X = torch.zeros((num_cls, num_all_batch), device=device)
        dist_X = torch.zeros((num_cls, num_all_batch), device=device)
        for j in range(num_cls):
            score_X[j, :], dist_X[j, :] = score(x_batch, mu_online[j], delta[j], lda, t)

        y_pred_batch = torch.argmin(score_X, dim=0)
        y_pred_batch.int()

        for i in range(num_all_batch):
            # loss += dist_X[y_pred_batch[i], i]
            loss += score_X[y_pred_batch[i], i]

        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

        del score_X, y_pred_batch, dist_X
        torch.cuda.empty_cache()

    return y_pred.clone(), loss.clone()


def prediction_online_D2_train(train_loader, num_cls, mu, L, D2, lda, t, device):
    d = mu.shape[1]

    mu_online = mu.clone()
    L_online = L.clone()
    D2_online = D2.clone()
    # num_batches = 10
    y_pred = torch.empty(0, device=device)
    loss = torch.tensor(0.0, device=device)

    delta = torch.zeros((num_cls, d, d))
    for j in range(num_cls):
        # delta[j] = deltaDiag(nc[j], L_online[j], S_online[j], lda).cpu()
        delta[j] = deltaDiag_D2(L_online[j], D2_online[j], lda).cpu()

    for id_batch, data in enumerate(train_loader):
        train_values, train_labels = data[0], list(data[1].numpy())
        x_batch = train_values.to(device)
        # print(id_batch, x_batch.shape)

        num_all_batch = x_batch.shape[0]
        ### Task: EM for PPCA2 - mix labeled and unlabeled together
        score_X = torch.zeros((num_cls, num_all_batch), device=device)
        dist_X = torch.zeros((num_cls, num_all_batch), device=device)
        for j in range(num_cls):
            delta_j = delta[j].to(device).clone()
            score_X[j, :], dist_X[j, :] = score(x_batch, mu_online[j], delta_j, lda, t)

        y_pred_batch = torch.argmin(score_X, dim=0)
        y_pred_batch.int()

        for i in range(num_all_batch):
            # loss += dist_X[y_pred_batch[i], i]
            loss += score_X[y_pred_batch[i], i]

        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

        del score_X, y_pred_batch, dist_X
        torch.cuda.empty_cache()

    return y_pred.clone(), loss.clone()


def EM_loss(log_ppi_prior):
    return -torch.sum(log_ppi_prior, dim=0)


# def Kmeans_loss(x, pred_cls, mu):
#     pass


class RAVE:
    def __init__(self):
        self.mx = None
        self.my = None
        self.mxx = None
        self.mxy = None
        self.myy = None
        self.n = 0

    def add(self, X, y):
        n, p = X.shape
        Sx = torch.sum(X, dim=0)
        Sy = torch.sum(y)
        Sxx = X.t() @ X
        Sxy = X.t() @ y
        Syy = y.t() @ y
        if self.n == 0:
            self.n = n
            self.mx = Sx / n
            self.my = Sy / n
            self.mxx = Sxx / n
            self.mxy = Sxy / n
            self.myy = Syy / n
        else:
            self.mx = self.mx * (self.n / (self.n + n)) + Sx / (self.n + n)
            self.my = self.my * (self.n / (self.n + n)) + Sy / (self.n + n)
            self.mxx = self.mxx * (self.n / (self.n + n)) + Sxx / (self.n + n)
            self.mxy = self.mxy * (self.n / (self.n + n)) + Sxy / (self.n + n)
            self.myy = self.myy * (self.n / (self.n + n)) + Syy / (self.n + n)
            self.n = self.n + n

    def add_Weighted_onlyX(self, X, weights):
        # Gamma is a diagonal matrix with weights as the diagonal elements
        Gamma = torch.diag(weights)
        N = torch.sum(weights)
        # n, p = X.shape
        Sx = X.t() @ weights  # p x 1
        Sxx = X.t() @ Gamma @ X  # p x p
        if self.n == 0:
            self.n = N
            self.mx = Sx / N
            self.mxx = Sxx / N
        else:
            self.mx = self.mx * (self.n / (self.n + N)) + Sx / (self.n + N)
            self.mxx = self.mxx * (self.n / (self.n + N)) + Sxx / (self.n + N)
            self.n = self.n + N

    def add_onlyX(self, X, mxx_cpu=False):
        n, p = X.shape
        Sx = torch.sum(X, dim=0)  # p x 1
        Sxx = (X.t() @ X)  # p x p
        if self.n == 0:
            self.n = n
            self.mx = Sx / n
            self.mxx = Sxx / n
        else:
            self.mx = (self.mx * (self.n / (self.n + n)) + Sx / (self.n + n))
            self.mxx = (self.mxx.cuda() * (self.n / (self.n + n)) + Sxx / (self.n + n))
            self.n = (self.n + n)
        if mxx_cpu:
            self.mxx = self.mxx.cpu()

    def add_score(self, X):
        n = len(X)
        Sx = torch.sum(X)  # 1
        Sxx = X.t() @ X  # 1
        if self.n == 0:
            self.n = n
            self.mx = Sx / n
            self.mxx = Sxx / n
        else:
            self.mx = self.mx * (self.n / (self.n + n)) + Sx / (self.n + n)
            self.mxx = self.mxx * (self.n / (self.n + n)) + Sxx / (self.n + n)
            self.n = self.n + n

    def add_rave(self, rave):
        n = rave.n
        if self.n == 0:
            self.n = rave.n
            self.mx = rave.mx.clone()
            self.my = rave.my.clone()
            self.mxx = rave.mxx.clone()
            self.mxy = rave.mxy.clone()
            self.myy = rave.myy.clone()
        else:
            n0 = self.n / (self.n + n)
            n1 = n / (self.n + n)
            self.mx = self.mx * n0 + rave.mx * n1
            self.my = self.my * n0 + rave.my * n1
            self.mxx = self.mxx * n0 + rave.mxx * n1
            self.mxy = self.mxy * n0 + rave.mxy * n1
            self.myy = self.myy * n0 + rave.myy * n1
            self.n = self.n + n

    def add_cluster(self, mx, XXn, n):
        ## mx is mean of new rave, XXn is covariance of new rave, n is the number of obs in new rave
        ## - Reconstruct E(X^2)
        mx=mx.clone()
        XXn=XXn.clone()
        mxx = XXn + mx.view(-1, 1) @ mx.view(1, -1)
        if self.n == 0:
            self.n = n
            self.mx = mx
            self.mxx = mxx
        else:
            n0 = self.n / (self.n + n)
            n1 = n / (self.n + n)
            self.n = self.n + n
            self.mx = self.mx * n0 + mx * n1
            self.mxx = self.mxx * n0 + mxx * n1

    def standardize_x(self):
        # standardize the raves for x
        var_x = torch.diag(self.mxx) - self.mx ** 2
        std_x = torch.sqrt(var_x)
        Pi = 1 / std_x

        XXn = self.mxx - self.mx.view(-1, 1) @ self.mx.view(1, -1)
        XXn *= Pi.view(1, -1)
        XXn *= Pi.view(-1, 1)

        return XXn, Pi

    def cov_weighted(self):
        self.mxx = self.mxx.cuda()
        self.mx = self.mx.cuda()
        XXn = self.mxx - self.mx.view(-1, 1) @ self.mx.view(1, -1)
        return XXn.clone()

    def cov_score(self):
        XXn = self.mxx - self.mx ** 2
        return XXn

    def standardize(self):
        # standardize the raves
        XXn, Pi = self.standardize_x()

        Temp1 = Pi * self.mxy
        Temp2 = self.my * Pi * self.mx
        XYn = Temp1 - Temp2

        return XXn, XYn, Pi


# Super Classes
class SuperClasses:
    def __init__(self, mu_cls, cov_cls, n_cls, num_supcls, device):
        self.mu_cls = mu_cls  # num_cls x d
        self.cov_cls = cov_cls  # num_cls x d x d
        self.n_cls = n_cls  # num_cls x 1
        self.num_supcls = num_supcls
        self.num_cls = mu_cls.shape[0]
        self.d = mu_cls.shape[1]
        self.device = device


    def initialization(self):
        ## Start from random selection
        self.all_cls_id = [i for i in range(self.num_cls)]
        self.initial_cls_id = sorted(random.sample(all_cls_id, k=self.num_supcls))
        self.mu_supcls = self.mu_cls[initial_cls_id, :]  # dim: num_supcls x d
        self.cov_supcls = self.cov_cls[initial_cls_id, :, :]  # dim: num_supcls x d x d
        self.cov_inv_supcls = torch.linalg.inv(self.cov_supcls)  # TODO: slow, need to improve
        self.not_initial_cls_id = sorted(list(set(all_cls_id).difference(set(initial_cls_id))))
        self.supcls_cls_dict = {i: [initial_cls_id[i]] for i in range(self.num_supcls)}
        self.raves = [RAVE() for sk in range(num_supcls)]
        for i in range(self.num_supcls):
            ini_cls_id_i=self.supcls_cls_dict[i][0]
            self.raves[i].add_cluster(self.mu_cls[ini_cls_id_i],self.cov_cls[ini_cls_id_i],self.n_cls[ini_cls_id_i])

    def mahalanobis(self, x):
        # x: d
        # mu_supcls: num_supcls x d
        # cov_supcls: num_supcls x d x d
        xc = x - self.mu_supcls  # num_supcls x d
        Xt = xc.unsqueeze(-2)  # num_supcls x 1 x d
        X = xc.unsqueeze(-1)  # num_supcls x d x 1
        mah = Xt @ self.cov_inv_supcls @ X  # n x 1 x 1
        dist = Xt @ X  # num_supcls x 1 x 1
        return mah.flatten().clone(), dist.flatten().clone()

    def update_supcls_kMeans(self, iter=10):
        for i in range(iter):
            # Task: calculate mahalanobis distance from class to centroids
            if iter > 0:
                cls_to_assign = self.all_cls_id
            else:
                cls_to_assign = self.not_initial_cls_id
            for cls_id in cls_to_assign:
                mah_dist, eu_dist = self.mahalanobis(self.mu_cls[cls_id, :])  # distance dim: num_supcls x 1
                _, assigned_supcls = torch.min(mah_dist)
                self.supcls_cls_dict[assigned_supcls].append(cls_id)
                self.raves[assigned_supcls].add_cluster(self.mu_cls[cls_id],self.cov_cls[cls_id],self.n_cls[cls_id])
            ## Task: update mu and cov, cov_inv for each supcls
            for supcls_id in self.all_cls_id:
                self.mu_supcls[supcls_id] = self.raves[supcls_id].mx.to(self.device).clone()
                self.cov_supcls[supcls_id] = self.raves[supcls_id].cov_weighted().clone().cpu()



# Utils
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # import random
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def HkPPCAs(num_labeled_per_class, q_ini, q_train, lda, num_iters, dataset_name, self_learner_name, feature_address,
            checkpoint_folder_address, run_id, sessions_list, seeds_list):
    num_sessions = len(sessions_list)  # 1
    for session_id in range(num_sessions):
        seed = seeds_list[session_id]
        setup_seed(seed)
        cls_range = sessions_list[session_id]
        num_prestored_cls = cls_range[0]
        num_new_cls = cls_range[1] - cls_range[0]
        print('This is {}th run, {}th-{}th classes. {} classes in this session, {} classes has been processed'
              .format(run_id + 1, cls_range[0] + 1, cls_range[1], num_new_cls, num_prestored_cls))

        ### Task: hyper-parameters
        ## weights for labeled data during subspace updating
        # alpha = num_labeled / N  # 0.5  # 10/nx #0.01
        # w_lab = alpha / num_labeled
        # w_unlab = (1 - alpha) / num_unlabeled
        # w_unlab = 0 # for all labeled

        # q_ini = 5  # q in initialization
        # threshold=0.95 # threshold for responsibility, if > threshold, resp becomes 1, else 0
        t = 1  # holder for temperature annealing
        # t_list=[1,0.75,0.5,0.25,0.1] # placeholder for temperature annealing schedule

        ### Task: Data Preparation
        ## Features integration
        print('Starting feature integration from all seperate files')
        if 'cifar100' in dataset_name.lower():
            train_set, val_set = CIFAR100_Feature(feature_address, cls_range=cls_range)  # for resnet x4
        if 'imagenet' in dataset_name.lower():
            train_set, val_set, folder_label_map = IMAGENET_Feature(feature_address,
                                                                    cls_range=cls_range)  # for resnet x4

        ## Randomize training set
        train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
        # for it, data in enumerate(train_loader):
        #     train_values, train_labels, train_ori_id = data[0], list(data[1].numpy()), data[2]
        data = next(iter(train_loader))
        train_values, train_labels, train_ori_id = data[0], list(data[1].numpy()), data[2]

        val_values, val_labels = val_set.values, val_set.labels  # for IMAGE_Feature()

        ## save .mat data
        tr_name = 'HkPPCAs_{}_run{}_{}-{}cls_trainset_{}.pth'.format(
            dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        val_name = 'HkPPCAs_{}_run{}_{}-{}cls_valset_{}.pth'.format(
            dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        addr_tr = checkpoint_folder_address + tr_name
        addr_val = checkpoint_folder_address + val_name
        savemat(addr_tr,
                {'feature': train_values.cpu().float().numpy(),
                 'label': train_labels,
                 'original_id': train_ori_id.numpy()})
        savemat(addr_val,
                {'feature': val_values.cpu().float().numpy(),
                 'val_labels': val_labels})

        ## for large dimension data as .pth
        # addr1 = 'D:/FSU/Academic/Research/CSNN-Project/Data/train_CLIP_resnet.pth'
        # addr2 = 'D:/FSU/Academic/Research/CSNN-Project/Data/val_CLIP_resnet.pth'
        # torch.save({'feature': train_values.cpu().float(),'label':train_labels}, addr1)
        # torch.save({'feature': val_values.cpu().float(),'val_labels':val_labels}, addr2)
        # # or save as .mat
        # addr1 = 'D:/FSU/Academic/Research/CSNN-Project/Data/train_CLIP_resnet.mat'
        # addr2 = 'D:/FSU/Academic/Research/CSNN-Project/Data/val_CLIP_resnet.mat'
        # import hdf5storage
        # hdf5storage.savemat(addr1,{'feature': train_values.cpu().float().numpy(),'label':train_labels},
        #                     matlab_compatible=True, compress=False)
        # savemat(addr2, {'feature': val_values.cpu().float().numpy(), 'val_labels': val_labels})
        ### Task//

        ### Task: Load prestored integrated features
        # for small dimension .mat
        tr_name = 'HkPPCAs_{}_run{}_{}-{}cls_trainset_{}.pth'.format(
            dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        val_name = 'HkPPCAs_{}_run{}_{}-{}cls_valset_{}.pth'.format(
            dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        addr_tr = checkpoint_folder_address + tr_name
        addr_val = checkpoint_folder_address + val_name

        train_data = loadmat(addr_tr)
        val_data = loadmat(addr_val)

        # ## for large dimension mat
        # # addr1 = 'D:/FSU/Academic/Research/CSNN-Project/Data/train_CLIP_resnet.mat'
        # # addr2 = 'D:/FSU/Academic/Research/CSNN-Project/Data/val_CLIP_resnet.mat'
        # # train_data = hdf5storage.loadmat(addr1)
        # # val_data = hdf5storage.loadmat(addr2)
        # addr1 = 'D:/FSU/Academic/Research/CSNN-Project/Data/train_CLIP_resnet.pth'
        # addr2 = 'D:/FSU/Academic/Research/CSNN-Project/Data/val_CLIP_resnet.pth'
        # train_data = torch.load(addr1)
        # val_data = torch.load(addr2)
        # train_data['feature']=train_data['feature'].numpy()
        # train_data['label']=[train_data['label']]
        # val_data['feature']=val_data['feature'].numpy()
        # val_data['val_labels'] = [val_data['val_labels']]

        # ## for combined features (MSN+Resnet x4)
        # addr1 = 'D:/FSU/Academic/Research/CSNN-Project/Data/train_MSN.mat'
        # addr2 = 'D:/FSU/Academic/Research/CSNN-Project/Data/val_MSN.mat'
        # addr3 = 'D:/FSU/Academic/Research/CSNN-Project/Data/train_CLIP.mat'
        # addr4 = 'D:/FSU/Academic/Research/CSNN-Project/Data/val_CLIP.mat'
        # train_data1 = loadmat(addr1)
        # train_data2 = loadmat(addr3)
        # val_data1 = loadmat(addr2)
        # val_data2 = loadmat(addr4)
        # train_data={'feature': np.concatenate((train_data1['feature'],train_data2['feature']),axis=1),
        #             'label': train_data1['label']}
        # val_data={'feature': np.concatenate((val_data1['feature'],val_data2['feature']),axis=1),
        #           'val_labels': val_data1['val_labels']}
        # del train_data1, train_data2, val_data1, val_data2
        #
        # # addr_comb_train = 'D:/FSU/Academic/Research/CSNN-Project/Data/train_MSN_CLIP_resnet.pth'
        # # addr_comb_val = 'D:/FSU/Academic/Research/CSNN-Project/Data/val_MSN_CLIP_resnet.pth'
        # # torch.save(train_data, addr_comb_train, pickle_protocol=4)
        # # torch.save(val_data, addr_comb_val, pickle_protocol=4)
        ### Task//

        norm_name = 'HkPPCAs_{}_run{}_Normalization_{}.pth'.format(
            dataset_name, run_id, self_learner_name)
        norm_base_session_address = checkpoint_folder_address + norm_name
        if session_id == 0:
            ## normalization for train and val data
            m_train_values = np.mean(train_data['feature'], axis=0)
            std_train_values = np.std(train_data['feature'], axis=0)
            d = m_train_values.shape[0]
            train_data['feature'] = (train_data['feature'] - m_train_values) / std_train_values / np.sqrt(d)
            m_val_values = np.mean(val_data['feature'], axis=0)
            std_val_values = np.std(val_data['feature'], axis=0)
            val_data['feature'] = (val_data['feature'] - m_val_values) / std_val_values / np.sqrt(d)

            # train_ori_id = train_data['original_id'] # for data augmentation

            torch.save({'m_train_values': m_train_values,
                        'std_train_values': std_train_values,
                        'm_val_values': m_val_values,
                        'std_val_values': std_val_values
                        },
                       norm_base_session_address)
        else:
            norm_basesession_result = torch.load(norm_base_session_address)
            m_val_basesession = norm_basesession_result['m_val_values']
            m_train_basesession = norm_basesession_result['m_train_values']
            std_val_basesession = norm_basesession_result['std_val_values']
            std_train_basesession = norm_basesession_result['std_train_values']
            d = m_train_basesession.shape[0]
            train_data['feature'] = (train_data['feature'] - m_train_basesession) / std_train_basesession / np.sqrt(d)
            val_data['feature'] = (val_data['feature'] - m_val_basesession) / std_val_basesession / np.sqrt(d)

        trainset = MixtureDataset(torch.tensor(train_data['feature'], dtype=torch.float), list(train_data['label'][0]))
        val_values, val_labels = torch.tensor(val_data['feature'], dtype=torch.float), list(val_data['val_labels'][0])
        N = len(trainset)

        # ## label by percentage
        # perc_labeled = 0.01
        # num_labeled = round(N * perc_labeled)
        # num_unlabeled = N - num_labeled

        ## label by a fixed number per class
        num_labeled = num_new_cls * num_labeled_per_class
        # num_unlabeled = N - num_labeled

        ### Task: obtain indices for labeled data
        train_values, train_labels = trainset.values, trainset.labels
        num_batches = 300  # 80
        batch_size = math.ceil(len(trainset) / num_batches)

        # ## obtain indices for labeled data - label by percentage
        # num_labeled_batch = math.ceil(num_labeled / num_batches)
        # labeled_indices = []
        # for i in range(num_batches):
        #     id_start = i * size_batch
        #     id_end = i * size_batch + num_labeled_batch
        #     labeled_indices_batch_global = [i for i in range(id_start, id_end)]
        #     labeled_indices.extend(labeled_indices_batch_global)
        #     # print(len(labeled_indices_batch),id_start,id_end)

        ## obtain indices for labeled data - label by fixed number per class
        labeled_indices = []  # id of all labeled observations
        selectedobs_dict = {}  # dict(labels:id_labeled_obs)
        for i, y in enumerate(train_labels):
            if y not in selectedobs_dict.keys():
                selectedobs_dict[y] = [i]
                labeled_indices.append(i)
            else:
                if len(selectedobs_dict[y]) < num_labeled_per_class:
                    selectedobs_dict[y].append(i)
                    labeled_indices.append(i)
                else:
                    continue
            if len(labeled_indices) == num_labeled:
                break

        ## - Check and summarize selected labeled results
        assert len(labeled_indices) == num_labeled
        labeled_indices = labeled_indices[:num_labeled]
        print('{} labeled per class, {} labeled samples in current session\n'
              .format(num_labeled_per_class, len(labeled_indices)))

        ## - Save labeled data information for augmentation
        # addr='D:/FSU/Academic/Research/Hierarchical K-PPCAs/TempExample/DataAug_LabeledImageIds_afterShuffling.pth'
        # labeled_original_ids=torch.from_numpy(train_ori_id[0,labeled_indices])
        # torch.save({'unshuffled_dataset': train_set,
        #             'original_id_train': train_ori_id,
        #             'labeled_orginal_ids': labeled_original_ids}, addr)

        ## - Select labeled data for initialization
        xl = train_values[labeled_indices, :]  # dim: (num_labeled, d)
        yl = [train_labels[i] for i in labeled_indices]
        xl = xl.to(device)
        print(xl.shape, len(yl))
        num_cls: int = len(set(train_labels)) + num_prestored_cls  # num_prestored_cls=0

        ### Task: initialization
        print('Start initialization')

        ### Task: initialize PPCA parameters (mu,L,D2) for classes
        pi_ini = torch.tensor([1 / num_cls], device=device).repeat(num_cls).unsqueeze(-1)  # dim: (num_cls, 1)
        mu_ini = torch.zeros((num_cls, d), device=device)
        sorted_unique_labels_cur_trainset = sorted(list(set(train_labels)))
        for _, label in enumerate(sorted_unique_labels_cur_trainset):
            mu_ini[label] = torch.mean(xl[torch.tensor(yl) == label, :], dim=0)

        cov_ini = torch.zeros((num_cls, d, d))
        L_ini = torch.zeros((num_cls, q_ini, d), device=device)
        # S_ini = torch.zeros((num_cls, q_ini), device=device)  # svd on x-mu version
        D2_ini = torch.zeros((num_cls, q_ini), device=device)  # svd on cov version
        nc_ini = torch.zeros((num_cls, 1), device=device)  # number of observations in each class
        for k in range(num_prestored_cls, num_cls):  # num_prestored_cls=0
            id_labeled = torch.tensor(yl) == k
            nc_ini[k] = torch.sum(id_labeled)
            cov_ini[k] = torch.from_numpy(np.cov(xl[id_labeled, :].T))
            _, s, v = torch.linalg.svd(xl[id_labeled, :] - mu_ini[k, :])
            L_ini[k, :, :] = v[0:q_ini, :]
            # s_fixedlenth = torch.zeros(q_ini)
            # minl = min(q_ini, len(s))
            # s_fixedlenth[:minl] = s[:minl]
            # S_ini[i, :] = s_fixedlenth  # svd on x-mu version
            D2_fixedlenth = torch.zeros(q_ini)
            minl = min(q_ini, len(s))
            S2 = torch.square(s) / (nc_ini[k] - 1)
            D2_fixedlenth[:minl] = S2[:minl]
            D2_ini[k, :] = D2_fixedlenth  # svd on cov version
            if k % 100 == 0:
                print(k)

        del id_labeled

        # ## - Recall stored classes from previous sessions and save to mu_ini, L_ini, S_ini, and nc
        # ## TODO: not store cov
        # if session_id > 0:
        #     checkpoint_old_session_name = 'HkPPCAs_{}_run{}_first{}cls_{}_PCAq{}_{}shot_q{}lda{:.0e}_iter{}.pth'.format(
        #         dataset_name, run_id, cls_range[0], self_learner_name, q_ini,
        #         num_labeled_per_class, q_train, lda, num_iters)
        #     prestored = checkpoint_folder_address + checkpoint_old_session_name
        #     # {'mu_online': mu_online,
        #     #  'L_online': L_online,
        #     #  'Sx_online': S_online,
        #     #  'D2_online': D2_online,
        #     #  'lda': lda, 't': t, 'N': N,
        #     #  'nc': nc, 'num_cls': num_cls,
        #     #  'trainset': trainset,
        #     #  'val_values': val_values,
        #     #  'val_labels': val_labels
        #     #  }
        #     prestored_result = torch.load(prestored)
        #
        #     nc_prestored = prestored_result['nc']
        #     mu_prestored = prestored_result['mu_online']
        #     L_prestored = prestored_result['L_online']
        #     # S_prestored = prestored_result['Sx_online']
        #     D2_prestored = prestored_result['D2_online']
        #
        #     nc_ini[:num_prestored_cls] = nc_prestored
        #     mu_ini[:num_prestored_cls] = mu_prestored
        #     L_ini[:num_prestored_cls] = L_prestored
        #     # S_ini[:num_prestored_cls] = S_prestored
        #     D2_ini[:num_prestored_cls] = D2_prestored

        ### Task: initialize super-classes by k-Means
        num_supcls = 200
        iter_supcls_ini = 10
        super_classes=SuperClasses(mu_ini, cov_ini, nc_ini, num_supcls, device=device)
        super_classes.initialization()
        super_classes.update_supcls_kMeans(iter=10)

        ## - Save initialization results
        addr_ini = checkpoint_folder_address + \
                   'HkPPCAs_{}_run{}_{}-{}cls_{}_PCAq_ini{}_{}shot_lda{:.0e}_ini.pth'.format(
                       dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name,
                       q_ini, num_labeled_per_class, lda)
        torch.save({'pi_ini': pi_ini,
                    'mu_ini': mu_ini,
                    'L_ini': L_ini,
                    # 'S_ini': S_ini,
                    'D2_ini': D2_ini,
                    'nc_ini': nc_ini,
                    'labeled_indices': labeled_indices,
                    'trainset': trainset},
                   addr_ini)

        print('initialization done\n')
        ## Task: Initialization End//

        addr_ini = checkpoint_folder_address + \
                   'HkPPCAs_{}_run{}_{}-{}cls_{}_PCAq_ini{}_{}shot_lda{:.0e}_ini.pth'.format(
                       dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name,
                       q_ini, num_labeled_per_class, lda)

        ini_result = torch.load(addr_ini)
        # pi_ini = ini_result['pi_ini']
        mu_ini = ini_result['mu_ini']
        L_ini = ini_result['L_ini']
        # S_ini = ini_result['S_ini']
        D2_ini = ini_result['D2_ini']
        nc_ini = ini_result['nc_ini']

        torch.cuda.empty_cache()

        ### Task: Generate indices of labeled data in each batch
        labeled_local_indices_per_batch = [[] for _ in range(num_batches)]
        for obs1_global_id in labeled_indices:
            obs1_local_id = obs1_global_id % batch_size
            batch_id = math.floor(obs1_global_id / batch_size)
            labeled_local_indices_per_batch[batch_id].append(obs1_local_id)

        ### Task: Prepare dataloader and PPCA parameters for training
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=False)

        # pi_online = pi_ini.clone()
        mu_online = mu_ini.clone()
        L_online = L_ini.clone()
        # S_online = S_ini.clone()
        D2_online = D2_ini.clone()
        nc = nc_ini.clone()
        del pi_ini, mu_ini, L_ini, D2_ini, ini_result
        torch.cuda.empty_cache()

        ### Task: Initial Accuracy for test set
        ## TODO: need to update prediction function to hierarchical version
        # y_pred, loss = prediction_online_s(val_values, val_labels, nc, num_cls,
        #                                    mu_online, L_online, S_online,
        #                                    lda, t, device)
        y_pred, loss = prediction_online_D2(val_values, val_labels, num_cls,
                                            mu_online, L_online, D2_online,
                                            lda, t, device)
        y_true = torch.tensor(val_labels, dtype=y_pred.dtype).to(device)
        cls_err = torch.sum(y_pred != y_true) / (len(val_labels))
        print('Initial Accuracy rate - val: {:.4f}'.format(1 - cls_err.item()))
        print('Initial Loss - val:', loss.item())
        ### Task//

        ## Start training
        q = q_train  # first q PCs in MPPCA
        # lda = 1e-2
        # plt_axis1 = []
        # plt_acc_val = []
        # plt_loss = []
        torch.cuda.empty_cache()
        print('there are {} classes'.format(num_cls))
        # num_iters = 10
        for i in range(num_iters):
            ### Task: online learning
            # t1 = time.time()
            raves = [RAVE() for k in range(num_cls)]

            # logdet_sigma = torch.zeros((num_cls, 1), device=device)
            delta = torch.zeros((num_cls, d, d))  # store at cpu
            for k in range(num_cls):
                # logdet_sigma[j] = logdet_cov(nc[j], S_online[j], d, lda)
                # logdet_sigma[j] = logdet_cov_D2(D2_online[j], d, lda)
                # delta[j] = deltaDiag(nc[j], L_online[j], S_online[j], lda).cpu()
                delta[k] = deltaDiag_D2(L_online[k], D2_online[k], lda).cpu()

            for batch_id, data in enumerate(train_loader):
                if batch_id % 10 == 0:
                    print('Current #batch: {}/{}'.format(batch_id, num_batches))
                # t10 = time.time()
                train_values, train_labels = data[0], list(data[1].numpy())
                x_batch = train_values.to(device)
                num_one_batch = x_batch.shape[0]

                labeled_local_indices_1batch = labeled_local_indices_per_batch[batch_id]
                num_labeled_batch = len(labeled_local_indices_1batch)

                ### Task: KMeans for PPCAs - labeled and unlabeled data mixed
                score_X = torch.zeros((num_cls, num_one_batch), device=device)
                for k in range(num_cls):
                    delta_k = delta[k].to(device)
                    score_X[k], dist = score(x_batch, mu_online[k], delta_k, lda, t)
                    # score_X[j] = score_dist_temp(x_batch, mu_online[j], L_online[j])

                # if num_labeled_batch > 0:
                #     score_X_labeled = torch.ones((num_cls, num_labeled_batch),
                #                                   device=device)  # num_cls x N
                #     for j in range(num_labeled_batch):
                #         score_X_labeled[train_labels[labeled_local_indices_1batch[j]], j] = 0
                #     # print(responsibility_labeled)
                #     score_X[:, labeled_local_indices_1batch] = score_X_labeled

                ## - Create labeled observations' mask for each class
                score_X_labeled = torch.zeros((num_cls, num_labeled_batch), dtype=torch.bool,
                                              device=device)  # num_cls x num_labeled in current batch
                if num_labeled_batch > 0:
                    for j in range(num_labeled_batch):
                        score_X_labeled[train_labels[labeled_local_indices_1batch[j]], j] = True
                    # score_X_labelmask[:,labeled_indices_1batch]=score_X_labeled

                ## - Generate filtered id to get minimum score mask for each observation
                score_X_filterid = torch.zeros((num_cls, num_one_batch), dtype=torch.bool, device=device)
                score_minval, score_label = torch.min(score_X, dim=0)  # vector, length=num_one_batch
                score_X_labeled_mask = score_X == score_minval

                ## - Create large score filtering mask
                for k in range(num_prestored_cls, num_cls):
                    # score_k = score_X[k, score_X_labeled_mask[k]]
                    if i == 0:  # no filter for iter 1
                        score_X_k_OODmask = torch.ones(num_one_batch, dtype=torch.bool, device=device)
                    else:
                        ## Score mean of all obs assigned to class k
                        score_mu_currentbat = torch.mean(score_X[k, score_X_labeled_mask[k]])
                        ## Score std of all obs assigned to class k
                        score_std_currentbat = torch.std(score_X[k, score_X_labeled_mask[k]])
                        score_X_k_OODmask = score_X[k] < (score_mu_currentbat + 2.56 * score_std_currentbat)
                    ## Mask to have obs assign to k th class and within the distribution
                    score_X_filterid[k] = score_X_k_OODmask & score_X_labeled_mask[k]
                    ## Combine all above obs and all labeled obs
                    score_X_filterid[k, labeled_local_indices_1batch] = score_X_filterid[k,
                                                                                         labeled_local_indices_1batch] \
                                                                        | score_X_labeled[k]

                    ## Apply mask to current batch, and move selected observations to corresponding RAVE
                    # raves[j].add_onlyX(x_batch[score_X_filterid[j], :].clone())
                    raves[k].add_onlyX(x_batch[score_X_filterid[k], :].clone(), mxx_cpu=True)

                # del score_k
                del x_batch, score_X, dist
                del score_X_filterid, score_label, score_X_labeled, score_X_labeled_mask, score_minval
                torch.cuda.empty_cache()

            # del logdet_sigma
            del delta
            if i > 0:
                del score_X_k_OODmask, score_mu_currentbat, score_std_currentbat
            torch.cuda.empty_cache()

            ## Move raves mxx mx to cpu()
            for k in range(num_prestored_cls, num_cls):
                raves[k].mxx = raves[k].mxx.cpu()
                raves[k].mx = raves[k].mx.cpu()

            ## - Update PPCAs parameters
            if session_id == 0:
                nc = torch.zeros((num_cls, 1), device=device)
                mu_online = torch.zeros((num_cls, d), device=device)
            else:
                nc = nc.clone()
                mu_online = mu_online.clone()
            cov_online = torch.zeros((num_cls, d, d))
            ## Update for new classes
            for k in range(num_prestored_cls, num_cls):
                nc[k] = raves[k].n
                mu_online[k] = raves[k].mx.to(device).clone()
                cov_online[k] = raves[k].cov_weighted().clone().cpu()
            mu_online = mu_online.cpu()  # save GPU memory

            del raves
            torch.cuda.empty_cache()

            if session_id == 0:
                L_online = torch.zeros((num_cls, q, d), device=device)
                # S_online = torch.zeros((num_cls, q), device=device)
                D2_online = torch.zeros((num_cls, q), device=device)
            else:  # Freeze parameters for old classes
                L_online = L_online.clone()
                # S_online = S_online.clone()
                D2_online = D2_online.clone()
            ## Only update new classes in the following loop
            for k in range(num_prestored_cls, num_cls):
                cov_online_cls = cov_online[k].to(device)
                vT, S2_online, v_online = torch.linalg.svd(cov_online_cls)
                L_online[k] = v_online[0:q, :].clone()
                # S_online[stp] = torch.sqrt(S2_online[0:q] * nc[stp]).clone()
                D2_online[k] = S2_online[0:q].clone()
                if k % 100 == 0:
                    print('svd step:{} cls'.format(k))

            # # dynamic q for each class according to PCs accumulative percentage
            # eigenvalues_sum = torch.sum(s_online, dim=1, keepdim=True)
            # percent_acc = torch.cumsum(s_online / eigenvalues_sum, dim=1).cpu()
            # q_class=torch.sum(percent_acc<q_filter_perc,dim=1)
            # q_max=torch.max(q_class).item()
            # L_online = torch.zeros((num_cls, q_max, d), device=device)
            # S_online = torch.zeros((num_cls, q_max), device=device)
            # for j in range(num_cls):
            #     L_online[j,0:q_class[j]]=v_online[j,0:k_class[j],:]
            #     S_online[j,0:q_class[j]]=torch.sqrt(s_online[j, 0:q_class[j]] * nc[j])

            mu_online = mu_online.to(device)
            print('Parameters update for PPCAs finished in this iteration')

            ## - Save parameters for following incremental sessions
            if (i + 1) % 5 == 0:
                checkpoint_iter_name = 'HkPPCAs_{}_run{}_first{}cls_{}_PCAq{}_{}shot_q{}lda{:.0e}_iter{}.pth'.format(
                    dataset_name, run_id, cls_range[1], self_learner_name, q_ini, num_labeled_per_class, q, lda, i + 1)
                torch.save({'mu_online': mu_online,
                            'L_online': L_online,
                            # 'Sx_online': S_online,
                            'D2_online': D2_online,
                            'lda': lda,
                            't': t,
                            'N': N,
                            'nc': nc,
                            'num_cls': num_cls,
                            'trainset': trainset,
                            'val_values': val_values,
                            'val_labels': val_labels
                            },
                           checkpoint_folder_address + checkpoint_iter_name)

            # del D2_online
            del v_online, vT
            del cov_online
            torch.cuda.empty_cache()

            # t2 = time.time()

            ## - Evaluation: Accuracy for val
            print('Evaluating test set from {}th class to {}th class'.format(min(val_labels), max(val_labels)))
            # y_pred_val, loss = prediction_online_s(val_values, val_labels, nc, num_cls, mu_online, L_online,
            #                                           S_online,
            #                                           lda, t,
            #                                           device)
            y_pred_val, loss = prediction_online_D2(val_values, val_labels, num_cls,
                                                    mu_online, L_online, D2_online,
                                                    lda, t, device)
            y_true_val = torch.tensor(val_labels, dtype=y_pred_val.dtype).to(device)
            cls_err_val = torch.sum(y_pred_val != y_true_val) / (len(val_labels))
            acc_val = 1 - cls_err_val.item()
            print('#{} Accuracy rate - val: {:.4f}'.format(i, acc_val))
            print('#{} loss - on: {}'.format(i, loss.item()))

            ## Calculate KMeans loss
            ## Loss calculation is replaced by prediction_online_s function
            # loss = Kmeans_loss(val_values, y_pred_val_on, mu_online)

            if (i + 1) == num_iters:
                trend_name = 'HkPPCAs_{}_run{}_first{}cls_acc_val_trend.pth'.format(
                    dataset_name, run_id, cls_range[1])
                torch.save(
                    {'acc_val': acc_val},
                    checkpoint_folder_address + trend_name)

            del y_pred_val, y_true_val
            torch.cuda.empty_cache()
            print('End of one iteration')

            # ### Task: plotting
            # plt_axis1.append(i + 1)
            # plt_acc_val.append(acc_val)
            # plt_loss.append(loss.item())
            # plt.clf()
            # x_major_locator = plt.MultipleLocator(1)
            # plt.figure(num=1, figsize=(15, 10))
            # graphic1 = plt.subplot(2, 1, 1)
            # graphic1.set_title('Accuracy')
            # plt.ylim((0, 1))
            # plt.plot(plt_axis1, plt_acc_val, label='val')
            # plt.legend()
            #
            # graphic2 = plt.subplot(2, 1, 2)
            # graphic2.set_title('Loss')
            # plt.plot(plt_axis1, plt_loss, label='online')
            # plt.pause(0.5)
            # ### Task//

            if (i + 1) == num_iters:
                ### Task: train classification
                # y_pred, loss = prediction_online_s_train(train_loader, nc, num_cls, mu_online,  # pi_online,
                #                                          L_online, S_online, lda, t,
                #                                          device)
                y_pred, loss = prediction_online_D2_train(train_loader, num_cls, mu_online,  # pi_online,
                                                          L_online, D2_online, lda, t,
                                                          device)
                y_true = torch.tensor(trainset.labels, dtype=y_pred.dtype).to(device)
                cls_err = torch.sum(y_pred != y_true) / (len(trainset.labels))  # (len(val_labels))
                print('Accuracy rate - train: {:.4f}'.format(1 - cls_err.item()))
                print('Loss - train:', loss.item())

                # plt.show()
                print('Done')

                # ## Accuracy for each class in Validation set
                # N_val=val_labels.__len__()
                # N_iter=int(N_val/50)
                # val_acc_cls=[]
                # for i in range(N_iter):
                #     id_cls_start=i*50
                #     id_cls_end=(i+1)*50
                #     y_pred_i=prediction_online(val_values[id_cls_start:id_cls_end],val_labels[id_cls_start:id_cls_end],
                #                          N, num_cls, mu_online, pi_online, L_online, S_online, lda, t,
                #                          device)
                #     cls_err_i=torch.sum(y_pred_i!=i)/50
                #     print('{}th class accuracy - val: {}'.format(i,1-cls_err_i.item()))
                #     val_acc_cls.append(1-cls_err_i.item())
                # plt.hist(val_acc_cls, bins=[i / 10 for i in range(0, 11, 1)], )


if __name__ == '__main__':
    ### device, seed, and torch print settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    # torch.set_printoptions(
    #     precision=4,
    #     threshold=1000,
    #     edgeitems=3,
    #     # linewidth=150,  # 每行最多显示的字符数，默认80，超过则换行显示
    #     profile=None,
    #     sci_mode=False  # 用科学技术法显示数据，默认True
    # )

    runs_seeds_list = [[5],
                       ]
    sessions_list = [[0, 1000]]
    num_runs = len(runs_seeds_list)
    num_iters = 5

    feature_address = [os.getcwd() + '\\..\\Benchmarks\\ImageNet_CLIP\\']
    checkpoint_folder_addr = os.getcwd() + '\\..\\Checkpoints\\'
    dataset_name = 'ImageNet_288x288'
    self_learner_name = 'CLIP_resnetx4'

    num_labeled_per_class = 2
    q_ini = 2
    q_HkPPCAs = 2
    lda = 1e-2

    for run_id in range(num_runs):
        seeds_list = runs_seeds_list[run_id]
        HkPPCAs(num_labeled_per_class,
                q_ini,
                q_HkPPCAs,
                lda,
                num_iters,
                dataset_name,
                self_learner_name,
                feature_address,
                checkpoint_folder_addr,
                run_id,
                sessions_list,
                seeds_list)

    winsound.Beep(frequency=800, duration=1000)
