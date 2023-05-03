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
import scipy.stats
import os

import matplotlib.pyplot as plt
import math
from collections import OrderedDict

from torch import device
from torch.utils.data import DataLoader, Dataset
import torch.distributions.multivariate_normal as MVNormal

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


def cov_PPCA(L_cls, D2_cls, lda):
    d = L_cls.shape[-1]
    D2_diag = torch.diag(D2_cls)  # q x q
    lda_diag = lda * torch.eye(d).to(L_cls.device)
    cov_PPCA = L_cls.t() @ D2_diag @ L_cls + lda_diag  # d x d
    return cov_PPCA.clone()


def score(x, mu_cls, delta_cls, lda, t=1):
    '''
    Calculate score(Mahalanobis distance) from many observation to 1 class
    :param x: n x d
    :param mu_cls: d
    :param delta_cls: d x d
    :param lda:
    :param t:
    :return: Score (Mahalanobis distance): (num_obs, )
    '''
    xc = x - mu_cls
    Xt = xc.unsqueeze(-2)  # n x 1 x d
    X = xc.unsqueeze(-1)  # n x d x 1
    # score = (Xt @ X)/t  #k=0, n x 1 x 1
    # score = (Xt @ X / lda)/t  #k=0, n x 1 x 1
    score = (Xt @ X / lda - Xt @ delta_cls @ X) / t  # n x 1 x 1
    dist = Xt @ X
    return score.flatten().clone(), dist.flatten().clone()


def score_1obs(x, mu_cls, delta_cls, lda, t=1):
    '''
    Calculate score(Mahalanobis distance) from 1 observation to many classes
    :param x: (d, )
    :param mu_cls: (num_cls, d)
    :param delta_cls: (num_cls, d, d)
    :param lda: scale of probabilistic term
    :param t: temperature (not used)
    :return: Score (Mahalanobis distance): (num_cls, )
    '''
    xc = x - mu_cls  # (num_cls, d)
    Xt = xc.unsqueeze(-2)  # (num_cls, 1, d)
    X = xc.unsqueeze(-1)  # (num_cls, 1, d)
    score = (Xt @ X / lda - Xt @ delta_cls @ X) / t  # (num_cls, 1, 1)
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


def prediction_D2_HkPPCAs(x_val, y_val, num_cls, mu, L, D2, super_classes, num_candid_supcls, lda, t, device,
                          num_batches=10):
    # TODO: debug
    d = x_val.shape[1]
    N_val = x_val.shape[0]
    num_supcls = super_classes.num_supcls

    mu_online = mu.clone()
    L_online = L.clone()
    D2_online = D2.clone()
    # num_batches = 10
    y_pred = torch.empty(0, device=device)
    loss = torch.tensor(0.0, device=device)

    delta = torch.zeros((num_cls, d, d), device=device)
    for k in range(num_cls):
        # delta[j] = deltaDiag(nc[j], L_online[j], S_online[j], lda)
        delta[k] = deltaDiag_D2(L_online[k], D2_online[k], lda)

    size_batch = int(N_val / num_batches)
    valset = MixtureDataset(x_val, y_val)
    val_loader = DataLoader(valset, batch_size=size_batch, shuffle=False, drop_last=False)
    for id_batch, data in enumerate(val_loader):
        val_values, val_labels = data[0], list(data[1].numpy())
        x_batch = val_values.to(device)
        # print(id_batch, x_batch.shape)

        n_one_batch = x_batch.shape[0]
        ### Task: KMeans for PPCAs - unlabeled only
        score_1batch = torch.zeros((n_one_batch), device=device)
        # dist_1batch = torch.zeros((num_cls, n_one_batch), device=device)
        y_pred_batch = torch.empty((n_one_batch), device=device)
        mah_dist_to_sup_cls_X = torch.zeros((num_supcls, n_one_batch), device=device)
        for supcls1_id in range(super_classes.num_supcls):
            ## Task: find closest 4 super classes for each obs in this batch
            mah_dist, eu_dist = super_classes.mahalanobis_1supcls(supcls1_id, x_batch)  # distance dim: n_1batch
            mah_dist_to_sup_cls_X[supcls1_id] = mah_dist
        # candid__supcls_id_X: (num_candid_supcls, n_one_batch)
        _, candid_supcls_id_X = torch.topk(mah_dist_to_sup_cls_X, dim=0, k=num_candid_supcls, largest=False)
        ## Task: find closest class from 4 candidate super classes for each observation
        for obs1_id in range(n_one_batch):
            ## Task: if unlabeled, select closest from candidate classes (formed by candidate sup-classes)
            candid_cls_id = []
            for candid_1supcls_id in candid_supcls_id_X[:, obs1_id].tolist():
                candid_cls_id.extend(super_classes.supcls_cls_dict[candid_1supcls_id])
            num_candid_cls = len(candid_cls_id)
            score_obs1 = torch.zeros(num_candid_cls)
            mu_candid_cls = mu_online[candid_cls_id]  # (num_candid_cls, d)
            delta_candid_cls = delta[candid_cls_id].to(device)  # (num_candid_cls, d, d)
            try:
                score_obs1, _ = score_1obs(x_batch[obs1_id], mu_candid_cls, delta_candid_cls, lda, t)
            except:
                print('ckpt')
            score_obs1_minval, score_obs1_minid = torch.min(score_obs1, dim=0)
            score_obs1_mincls = candid_cls_id[score_obs1_minid]
            score_1batch[obs1_id] = score_obs1_minval
            y_pred_batch[obs1_id] = score_obs1_mincls

        loss += torch.sum(score_1batch)
        y_pred_batch.int()
        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

        del score_1batch, y_pred_batch  # , dist_1batch
        torch.cuda.empty_cache()

    return y_pred.clone(), loss.clone()


def prediction_D2_train_HkPPCAs(train_loader, num_cls, mu, L, D2, super_classes, num_candid_supcls, lda, t, device):
    # TODO: debug
    d = mu.shape[1]
    num_supcls = super_classes.num_supcls

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

        n_one_batch = x_batch.shape[0]
        ### Task: KMeans for PPCAs - labeled and unlabeled mixed
        score_1batch = torch.zeros((n_one_batch), device=device)
        # dist_1batch = torch.zeros((num_cls, n_one_batch), device=device)
        y_pred_batch = torch.empty((n_one_batch), device=device)
        mah_dist_to_sup_cls_X = torch.zeros((num_supcls, n_one_batch), device=device)
        for sup_cls1 in range(num_supcls):
            ## Task: find closest 4 super classes for each obs in this batch
            mah_dist, eu_dist = super_classes.mahalanobis_1supcls(sup_cls1, x_batch)  # distance dim: n_1batch
            mah_dist_to_sup_cls_X[sup_cls1] = mah_dist
        # candid__supcls_id_X: (num_candid_supcls, n_one_batch)
        _, candid_supcls_id_X = torch.topk(mah_dist_to_sup_cls_X, dim=0, k=num_candid_supcls, largest=False)
        ## Task: find closest class from 4 candidate super classes for each observation
        for obs1_id in range(n_one_batch):
            ## Task: if unlabeled, select closest from candidate classes (formed by candidate sup-classes)
            candid_cls_id = []
            for candid_1supcls_id in candid_supcls_id_X[:, obs1_id].tolist():
                candid_cls_id.extend(super_classes.supcls_cls_dict[candid_1supcls_id])
            num_candid_cls = len(candid_cls_id)
            score_obs1 = torch.zeros(num_candid_cls)
            mu_candid_cls = mu_online[candid_cls_id]  # (num_candid_cls, d)
            delta_candid_cls = delta[candid_cls_id].to(device)#.clone()  # (num_candid_cls, d, d)
            try:
                score_obs1, _ = score_1obs(x_batch[obs1_id], mu_candid_cls, delta_candid_cls, lda, t)
            except:
                print('ckpt')
            score_obs1_minval, score_obs1_minid = torch.min(score_obs1, dim=0)
            score_obs1_mincls = candid_cls_id[score_obs1_minid]
            score_1batch[obs1_id] = score_obs1_minval
            y_pred_batch[obs1_id] = score_obs1_mincls

        loss += torch.sum(score_1batch)
        y_pred_batch.int()
        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

        del score_1batch, y_pred_batch  # , dist_1batch
        torch.cuda.empty_cache()

    return y_pred.clone(), loss.clone()


def EM_loss(log_ppi_prior):
    return -torch.sum(log_ppi_prior, dim=0)


# def Kmeans_loss(x, pred_cls, mu):
#     pass


def KLDivergence_emp(p_log, all_q_log):
    '''
    This is emperical KL divergence based on observations in the dataset
    :param p_log: log(P(x)), P(x) is the pdf of class. (n_obs, )
    :param all_q_log: log(Q(x)), Q(x)s are the pdf of all sup-classes. (num_supcls, n_obs)
    :return: emperical KL divergence (num_supcls, )
    '''
    p = p_log.exp()
    KL = (p_log - all_q_log) @ p  # num_supcls x 1
    return KL


def KLDivergence_exp(mean_p, cov_p, mean_q, cov_q, inv_cov_q):
    '''
    This is expected KL divergence based on modeled Gaussian distribution
    :param mean_p: mean of Gaussian for 1 class: (d, )
    :param cov_p: covariance matrix of Gaussian for 1 class: (d, d)
    :param mean_q: mean of Gaussian for all sup-classes: (num_supcls, d)
    :param cov_q: covariance matrix of Gaussian for all sup-classes: (num_supcls, d, d)
    :return: expected KL divergenc: (num_supcls, num_cls)
    '''
    # num_cls = mean_p.shape[0]
    num_supcls = mean_q.shape[0]
    mean_q_unsq = mean_q.detach()  # .unsqueeze(1)  # (num_supcls, 1, d)
    cov_q_unsq = cov_q.detach()  # .unsqueeze(1)  # (num_supcls, 1, d, d)
    inv_cov_q_unsq = inv_cov_q.detach()  # .unsqueeze(1)  # (num_supcls,1, d, d)
    term1 = torch.logdet(cov_p) - torch.logdet(cov_q_unsq)  # (num_supcls, )
    term2 = -mean_p.shape[-1]
    mp_mq = (mean_p - mean_q_unsq).unsqueeze(-2)  # (num_sup_cls, num_cls, d)
    term3 = (mp_mq @ inv_cov_q_unsq @ torch.transpose(mp_mq, 1, 2)).squeeze()  # (num_supcls, )
    inv_cov_q_by_cov_q_set = inv_cov_q_unsq @ cov_p  # (num_supcls, d, d)
    term4 = inv_cov_q_by_cov_q_set.diagonal(0, -2, -1).sum(-1)  # trace of product of two covariances: (num_supcls, )

    # # the 4th term tr(inv_cov_q @ cov_p) generates large matrix during calculation (num_supcls, num_cls, d, d)
    # # So split the calculation by sets of classes
    # num_classes_of_set = 100
    # num_sets = math.ceil(num_cls / num_classes_of_set)
    # term4 = torch.zeros((num_supcls, num_cls), device=cov_p.device)
    # print('Calculating trace term in KL divergence')
    # for set_i in range(num_sets):
    #     start_cls_id = set_i * num_classes_of_set
    #     end_cls_id = (set_i + 1) * num_classes_of_set
    #     if end_cls_id>num_cls:
    #         end_cls_id=num_cls
    #     inv_cov_q_by_cov_q_set =inv_cov_q_unsq @ cov_p[start_cls_id:end_cls_id] # (num_supcls, num_cls_set, d, d)
    #     term4_1set=inv_cov_q_by_cov_q_set.diagonal(0,-2,-1).sum(-1) # trace of product of two covariances: (num_supcls, num_cls_set)
    #     term4[:,start_cls_id:end_cls_id]=term4_1set
    #     if set_i%500==0:
    #         print('{} classes per set, processed {}/{} sets'.format(num_classes_of_set, set_i,num_sets))

    KL = 1 / 2 * (term1 + term2 + term3 + term4)  # (num_supcls, num_cls)
    return KL


class RAVE:
    def __init__(self):
        self.mx = None
        self.my = None
        self.mxx = None
        self.mxy = None
        self.myy = None
        self.n = 0
        self.score_list = []
        self.raw_data=None
        self.raw_resp='not_full'

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

    # def add_score(self, X):
    #     n = len(X)
    #     Sx = torch.sum(X)  # 1
    #     Sxx = X.t() @ X  # 1
    #     if self.n == 0:
    #         self.n = n
    #         self.mx = Sx / n
    #         self.mxx = Sxx / n
    #     else:
    #         self.mx = self.mx * (self.n / (self.n + n)) + Sx / (self.n + n)
    #         self.mxx = self.mxx * (self.n / (self.n + n)) + Sxx / (self.n + n)
    #         self.n = self.n + n

    def add_score(self, score):
        self.score_list.append(score.item())

    def add_scores(self, scores):
        self.score_list.extend(scores.tolist())

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
        mx = mx.clone()
        XXn = XXn.clone()
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

    def calc_score_criteria_normal(self, sig_level=0.975):
        if len(self.score_list) == 0:
            score_criteria = torch.tensor(float('Inf'))
        else:
            score_tensor = torch.tensor(self.score_list)
            self.score_mean = torch.mean(score_tensor)
            self.score_std = torch.std(score_tensor)
            critical_val=scipy.stats.norm.ppf(sig_level)
            score_criteria = self.score_mean + critical_val * self.score_std
        return score_criteria

    def calc_score_criteria_chi(self, sig_level=0.975):
        df=self.mx.shape[-1] # chi-square with df=d
        if len(self.score_list) == 0:
            score_criteria = torch.tensor(float('Inf'))
        else:
            score_criteria=torch.tensor(scipy.stats.chi2.ppf(sig_level, df))
        return score_criteria

    def add_raw_data(self, x, y, m=20):
        '''select m observations to form raw data for superclass updating'''
        if len(y)>=m:
            ids=torch.randperm(len(y))[:m]
            x_raw=x[ids]
            y_raw=y[ids]
        else:
            x_raw=x
            y_raw=y
        # if raw_data is empty
        if self.raw_data is None:
            self.raw_data=[x_raw,y_raw]
        else: # append and trim
            self.raw_data[0]=torch.cat((self.raw_data[0],x_raw),0)
            self.raw_data[1] = torch.cat((self.raw_data[1], y_raw), 0)
        assert self.raw_data[0].shape[0]==len(self.raw_data[1])
        if len(self.raw_data[1])>=m:
            self.raw_data=[self.raw_data[0][:m],self.raw_data[1][:m]]
            self.raw_resp='full'
        return self.raw_resp


# Super Classes
class SuperClasses:
    def __init__(self, mu_cls, n_cls, num_supcls, q_supcls, device, PPCA=False, cov_cls=None, L_cls=None, D2_cls=None):
        self.device = device
        self.mu_cls = mu_cls  # num_cls x d
        self.n_cls = n_cls  # n_cls x 1
        self.num_supcls = num_supcls
        self.num_cls = mu_cls.shape[0]
        self.d = mu_cls.shape[1]
        self.lda = 1e-6
        self.small_jittering = self.lda * torch.eye(self.d, device=self.device)
        self.q_supcls = q_supcls
        self.PPCA = PPCA
        if PPCA is False:
            assert cov_cls is not None
            self.cov_cls = cov_cls.to(self.device)  # num_cls x d x d
        else:
            assert L_cls is not None
            assert D2_cls is not None
            self.L_cls = L_cls.to(self.device) # num_cls x q x d
            self.D2_cls = D2_cls.to(self.device) # num_cls x q

    def initialize(self):
        ## Start from random selection
        self.all_cls_id = [i for i in range(self.num_cls)]
        # self.all_supcls_id = [i for i in range(self.num_supcls)]

        self.initial_cls_id = sorted(
            random.sample(self.all_cls_id, k=self.num_supcls))  # select some classes as initial super class
        self.mu_supcls = self.mu_cls[self.initial_cls_id, :]  # dim: num_supcls x d

        if self.PPCA is True:
            self.cov_cls = torch.zeros((self.num_cls, self.d, self.d))
            # TODO: use L, D2 to reconstruct cov_cls, then use svd to cov_supcls and rebuild cov_supcls and cov_inv_supcls
            for cls_id in range(self.num_cls):
                self.cov_cls[cls_id] = cov_PPCA(self.L_cls[cls_id], self.D2_cls[cls_id], self.lda).cpu()
                if cls_id % 500 == 0:
                    print('SuperClasses::cov & cov_inv reconstruction: {}/{} classes'.format(cls_id, self.num_cls))
        else:
            self.cov_cls = self.cov_cls + self.small_jittering

        # self.cov_inv_cls = torch.linalg.inv(self.cov_cls + self.small_jittering)  # TODO: slow, need to improve
        self.cov_supcls = self.cov_cls[self.initial_cls_id, :, :]  # dim: num_supcls x d x d

        if self.PPCA is True:
            self.cov_inv_supcls = torch.zeros((self.num_supcls, self.d, self.d))
            self.L_supcls = torch.zeros((self.num_supcls, self.q_supcls, self.d), device=self.device)
            self.D2_supcls = torch.zeros((self.num_supcls, self.q_supcls), device=self.device)
            for supcls_id in range(self.num_supcls):
                cov_1supcls = self.cov_supcls[supcls_id].to(self.device)
                vT, S2_1supcls, v_1supcls = torch.linalg.svd(cov_1supcls)
                self.L_supcls[supcls_id] = v_1supcls[0:self.q_supcls, :].clone()
                # S_online[stp] = torch.sqrt(S2_online[0:q] * nc[stp]).clone()
                self.D2_supcls[supcls_id] = S2_1supcls[0:self.q_supcls].clone()
                if supcls_id % 100 == 0:
                    print('SupClasses::cov_supcls svd: {}/{} sup-classes'.format(supcls_id, self.num_supcls))
                self.cov_inv_supcls[supcls_id] = torch.eye(self.d) / self.lda - deltaDiag_D2(self.L_supcls[supcls_id],
                                                                                             self.D2_supcls[supcls_id],
                                                                                             self.lda).cpu()
        else:
            self.cov_inv_supcls = torch.linalg.inv(
                self.cov_supcls + self.small_jittering)  # TODO: slow, need to improve

        self.not_initial_cls_id = sorted(list(set(self.all_cls_id).difference(set(self.initial_cls_id))))
        self.supcls_cls_dict = {i: [self.initial_cls_id[i]] for i in range(self.num_supcls)}
        self.raves = [RAVE() for sc in range(self.num_supcls)]
        for i in range(self.num_supcls):
            ini_cls_id_i = self.supcls_cls_dict[i][0]
            self.raves[i].add_cluster(self.mu_cls[ini_cls_id_i], self.cov_cls[ini_cls_id_i].to(self.device), self.n_cls[ini_cls_id_i])

    def mahalanobis_1obs(self, x):
        '''
        calculate Mahalanobis distance from one (or several) obeservation to all sup-classes
        :param x:
        :return:
        '''
        # x: d or n_obs x d
        # mu_supcls: num_supcls x d
        # cov_supcls: num_supcls x d x d
        if len(x.shape)>1:
            xc = x.unsqueeze(-2)-self.mu_supcls # n_obs x num_supcls x d
        else:
            xc = x - self.mu_supcls  # num_supcls x d
        assert xc.shape[-2] == self.num_supcls
        Xt = xc.unsqueeze(-2)  # (n_obs) x num_supcls x 1 x d
        X = xc.unsqueeze(-1)  # (n_obs) x num_supcls x d x 1
        mah = Xt @ self.cov_inv_supcls.to(x.device) @ X  # (n_obs) x num_supcls x 1 x 1
        dist = Xt @ X  # (n_obs) x num_supcls x 1 x 1
        return mah.flatten(start_dim=-3).clone(), dist.flatten(start_dim=-3).clone()

    def eu_1obs(self, x):
        '''
        calculate Euclidean distance from one (or several) obeservation to all sup-classes
        :param x:
        :return:
        '''
        # x: d
        if len(x.shape)>1:
            xc = x.unsqueeze(-2)-self.mu_supcls # n_obs x num_supcls x d
        else:
            xc = x - self.mu_supcls  # num_supcls x d
        assert xc.shape[-2] == self.num_supcls
        Xt = xc.unsqueeze(-2)  # (n_obs) x num_supcls x 1 x d
        X = xc.unsqueeze(-1)  # (n_obs) x num_supcls x d x 1
        dist = Xt @ X  # (n_obs) x num_supcls x 1 x 1
        return dist.flatten(start_dim=-3).clone()

    def mahalanobis_1supcls(self, sup_cls_id, x):
        '''
        Calculate Mahalanobis distance from one selected sup-class to all observations
        :param sup_cls_id: sup-class id
        :param x: all observations: (n_obs, d)
        :return: Mahalanobis distance: (n_obs, 1)
        '''
        mu_supcls = self.mu_supcls[sup_cls_id]  # (d,)
        cov_inv_supcls = self.cov_inv_supcls[sup_cls_id].to(self.device)
        xc = x - mu_supcls  # (n_obs, d)
        Xt = xc.unsqueeze(-2)  # (n_obs, 1, d)
        X = xc.unsqueeze(-1)  # (n_obs, 1, d)
        mah = Xt @ cov_inv_supcls @ X  # n_obs x 1 x 1
        dist = Xt @ X  # n_obs x 1 x 1
        return mah.flatten().clone(), dist.flatten().clone()

    def update_supcls_kMeans_gaussian(self, measure='Mah', iter=10):
        for i in range(iter):
            # Task: Assign class to closest superclass
            if i == 0:
                cls_to_assign = self.not_initial_cls_id
            else:
                cls_to_assign = self.all_cls_id
            for cls_id in cls_to_assign:
                mah_dist, eu_dist = self.mahalanobis_1obs(self.mu_cls[cls_id, :])  # distance dim: num_supcls x 1
                if measure=='Mah':
                    _, assigned_supcls = torch.min(mah_dist, dim=0)
                if measure=='Eu':
                    _, assigned_supcls = torch.min(eu_dist, dim=0)
                else:
                    raise Exception("Measurement should be Mah(alanobis) or Eu(clidean)")
                self.supcls_cls_dict[assigned_supcls.item()].append(cls_id)
                self.raves[assigned_supcls].add_cluster(self.mu_cls[cls_id], self.cov_cls[cls_id], self.n_cls[cls_id])
            # ## Task: update mu and cov, cov_inv for each supcls
            # for supcls_id in range(self.num_supcls):
            #     try:
            #         self.supcls_cls_dict[supcls_id] = sorted(list(set(self.supcls_cls_dict[supcls_id])))
            #         self.mu_supcls[supcls_id] = self.raves[supcls_id].mx.to(self.device).clone()
            #         self.cov_supcls[supcls_id] = self.raves[supcls_id].cov_weighted().clone().cpu()
            #     except:
            #         print('Modules:SuperClass:ckpt1')
            #
            # self.cov_inv_supcls = torch.linalg.inv(self.cov_supcls + self.small_jittering)
            ## Task: update mu and cov, cov_inv, L, D2 for each supcls
            for supcls_id in range(self.num_supcls):
                self.supcls_cls_dict[supcls_id] = sorted(list(set(self.supcls_cls_dict[supcls_id])))
                self.mu_supcls[supcls_id] = self.raves[supcls_id].mx.to(self.device).clone()
                cov_1supcls = self.raves[supcls_id].cov_weighted().clone()
                self.cov_supcls[supcls_id] = cov_1supcls.cpu()
                if self.PPCA is True:
                    ## Task: apply SVD to cov_supcls
                    vT, S2_1supcls, v_1supcls = torch.linalg.svd(cov_1supcls)
                    self.L_supcls[supcls_id] = v_1supcls[0:self.q_supcls, :].clone()
                    # S_online[stp] = torch.sqrt(S2_online[0:q] * nc[stp]).clone()
                    self.D2_supcls[supcls_id] = S2_1supcls[0:self.q_supcls].clone()
                    if supcls_id % 100 == 0:
                        print('SupClasses::Update::cov_supcls svd: {}/{} sup-classes'.format(supcls_id, self.num_supcls))
                    self.cov_inv_supcls[supcls_id] = torch.eye(self.d) / self.lda - deltaDiag_D2(
                        self.L_supcls[supcls_id],
                        self.D2_supcls[supcls_id],
                        self.lda).cpu()
                else:
                    self.cov_inv_supcls[supcls_id] = torch.linalg.inv(cov_1supcls + self.small_jittering)

    def update_supcls_kMeans_raw(self, raw_data, measure='Mah', iter=10):
        '''
        Update super-class by applying k-Means on labeled raw data sampled from each class
        :param raw_data: labeled raw data [tensor:xl (num_cls * lab_per_cls, d), tensor:yl (num_cls * lab_per_cls)]
        :param measure: measurement for distance, can be 'Mah' for Mahalanobis distance or 'Eu' for Euclidean distance
        :param iter: number of iterations to update super-class
        :return:
        '''
        time0=time.time()
        for i in range(iter):
            # Task: Assign class to closest superclass
            if i == 0:
                cls_to_assign = self.not_initial_cls_id
            else:
                cls_to_assign = self.all_cls_id
            for cls_id in cls_to_assign:
                id_xl_cls = raw_data[1]==cls_id
                xl_cls=raw_data[0][id_xl_cls,:] # (lab_per_cls, d)
                if measure=='Mah':
                    mah_dist, _ = self.mahalanobis_1obs(xl_cls)  # distance dim: n_obs x num_supcls x 1
                    _, assigned_supcls = torch.min(mah_dist, dim=1) # (lab_per_cls,)
                elif measure=='Eu':
                    eu_dist = self.eu_1obs(xl_cls)  # distance dim: num_supcls x 1
                    _, assigned_supcls = torch.min(eu_dist, dim=1) # (lab_per_cls,)
                else:
                    raise Exception("Measurement should be Mah(alanobis) or Eu(clidean)")
                for local_xl_id, assigned_supcls_1obs in enumerate(assigned_supcls):
                    if cls_id not in self.supcls_cls_dict[assigned_supcls_1obs.item()]:
                        self.supcls_cls_dict[assigned_supcls_1obs.item()].append(cls_id)
                    self.raves[assigned_supcls_1obs].add_onlyX(xl_cls[local_xl_id].unsqueeze(-2))
            ## Task: update mu and cov, cov_inv, L, D2 for each supcls
            for supcls_id in range(self.num_supcls):
                self.supcls_cls_dict[supcls_id] = sorted(list(set(self.supcls_cls_dict[supcls_id])))
                self.mu_supcls[supcls_id] = self.raves[supcls_id].mx.to(self.device).clone()
                cov_1supcls = self.raves[supcls_id].cov_weighted().clone()
                self.cov_supcls[supcls_id] = cov_1supcls.cpu()
                if self.PPCA is True:
                    ## Task: apply SVD to cov_supcls
                    vT, S2_1supcls, v_1supcls = torch.linalg.svd(cov_1supcls)
                    self.L_supcls[supcls_id] = v_1supcls[0:self.q_supcls, :].clone()
                    # S_online[stp] = torch.sqrt(S2_online[0:q] * nc[stp]).clone()
                    self.D2_supcls[supcls_id] = S2_1supcls[0:self.q_supcls].clone()
                    if supcls_id % 100 == 0:
                        print('SupClasses::Update::cov_supcls svd: {}/{} sup-classes'.format(supcls_id, self.num_supcls))
                    self.cov_inv_supcls[supcls_id] = torch.eye(self.d) / self.lda - deltaDiag_D2(
                        self.L_supcls[supcls_id],
                        self.D2_supcls[supcls_id],
                        self.lda).cpu()
                else:
                    self.cov_inv_supcls[supcls_id] = torch.linalg.inv(cov_1supcls + self.small_jittering)
        print('time cost - raw data:',time.time()-time0)

    def update_supcls_KLdiv_emp(self, X, cls_obsid_map, iter=10):
        # TODO: cls_obsid_map need to be build in Main function
        '''
        Update Sup-classes by empirical KL divergence. P is the class distribution. Q is the sup=class distribution.
        :param X: Observations to calculate pdf's. Dimension: n x d
        :param cls_obsid_map: dictionary {class: observations belongs to this class or not (bool tensor)}
        :param iter: number of iterations for sup-class updating
        :return: None
        '''
        for i in range(iter):
            print('Updating sup-classes: {}/{} iters'.format(i, iter))
            # Task: Assign class to closest superclass
            if i == 0:
                cls_to_assign = self.not_initial_cls_id
            else:
                cls_to_assign = self.all_cls_id
            kl_div_all_cls = torch.zeros((self.num_cls))
            for cls_id in cls_to_assign:
                n_cls = cls_obsid_map[cls_id].sum()
                X_cls = X[cls_obsid_map[cls_id]]  # dim: n_cls x d
                P_dist = MVNormal.MultivariateNormal(self.mu_cls[cls_id], self.cov_cls[cls_id])
                p_log = P_dist.log_prob(X_cls)  # dim: n_cls
                all_q_log = torch.zeros((self.num_supcls, n_cls), device=self.device)  # dim: n_cls x num_supcls

                for supcls_id in range(self.num_supcls):
                    # TODO: here may lead to slow calculation, not vectorized, O(num_cls*num_supcls ~10^6)
                    Q_dist = MVNormal.MultivariateNormal(self.mu_supcls[supcls_id],
                                                         self.cov_supcls[supcls_id])
                    q_log = Q_dist.log_prob(X_cls)  # dim: n_cls
                    all_q_log[supcls_id] = q_log
                    # if supcls_id%50==0:
                    #     print("Processing log q(x) for all superclasses: {}/{}".format(supcls_id,self.num_supcls))
                kl_div = KLDivergence_emp(p_log, all_q_log)  # dim: num_supcls x 1
                kl_div_assigned, assigned_supcls = torch.min(kl_div, dim=0)
                kl_div_all_cls[cls_id] = kl_div_assigned
                self.supcls_cls_dict[assigned_supcls.item()].append(cls_id)
                self.raves[assigned_supcls].add_cluster(self.mu_cls[cls_id], self.cov_cls[cls_id], self.n_cls[cls_id])
                if cls_id % 500 == 0:
                    print('SuperClasses::Update::Assign Classes:{}/{} Classes'.format(cls_id, self.num_cls))
            ## Task: update mu and cov, cov_inv for each supcls
            for supcls_id in range(self.num_supcls):
                self.supcls_cls_dict[supcls_id] = sorted(list(set(self.supcls_cls_dict[supcls_id])))
                self.mu_supcls[supcls_id] = self.raves[supcls_id].mx.to(self.device).clone()
                cov_1supcls = self.raves[supcls_id].cov_weighted().clone()
                self.cov_supcls[supcls_id] = cov_1supcls.cpu()
                if self.PPCA is True:
                    ## Task: apply SVD to cov_supcls
                    vT, S2_1supcls, v_1supcls = torch.linalg.svd(cov_1supcls)
                    self.L_supcls[supcls_id] = v_1supcls[0:self.q_supcls, :].clone()
                    # S_online[stp] = torch.sqrt(S2_online[0:q] * nc[stp]).clone()
                    self.D2_supcls[supcls_id] = S2_1supcls[0:self.q_supcls].clone()
                    if supcls_id % 100 == 0:
                        print('SupClasses::Update::cov_supcls svd: {}/{} sup-classes'.format(supcls_id, self.num_supcls))
                    self.cov_inv_supcls[supcls_id] = torch.eye(self.d) / self.lda - deltaDiag_D2(
                        self.L_supcls[supcls_id],
                        self.D2_supcls[supcls_id],
                        self.lda).cpu()
                else:
                    self.cov_inv_supcls[supcls_id] = torch.linalg.inv(cov_1supcls + self.small_jittering)
            print('Average KL-Divergence over all classes: {}'.format(torch.mean(kl_div_all_cls)))

            self.cov_inv_supcls = torch.linalg.inv(self.cov_supcls + self.small_jittering)

    def update_supcls_KLdiv_exp(self, iter=10):
        '''
        Update Sup-classes by expected KL divergence. P is the class distribution. Q is the sup=class distribution.
        :param iter:
        :return:
        '''
        time0 = time.time()
        for i in range(iter):
            print('Updating sup-classes: {}/{} iters'.format(i, iter))
            # Task: Assign class to closest superclass
            if i == 0:
                cls_to_assign = self.not_initial_cls_id
            else:
                cls_to_assign = self.all_cls_id

            # kl_div = KLDivergence_exp(self.mu_cls,self.cov_cls,
            #                           self.mu_supcls, self.cov_supcls, self.cov_inv_supcls) # (num_supcls, num_cls)
            # _, assigned_supcls = torch.min(kl_div, dim=0) # (num_cls, )
            kl_div_all_cls = torch.zeros((self.num_cls))
            for cls_id in cls_to_assign:
                kl_div = KLDivergence_exp(self.mu_cls[cls_id], self.cov_cls[cls_id].to(self.device),
                                          self.mu_supcls,
                                          self.cov_supcls.to(self.device),
                                          self.cov_inv_supcls.to(self.device))  # (num_supcls, num_cls)
                kl_div_assigned, assigned_supcls = torch.min(kl_div, dim=0)  # (num_cls, )
                kl_div_all_cls[cls_id] = kl_div_assigned
                self.supcls_cls_dict[assigned_supcls.item()].append(cls_id)
                self.raves[assigned_supcls].add_cluster(self.mu_cls[cls_id], self.cov_cls[cls_id].to(self.device), self.n_cls[cls_id])
                if cls_id % 500 == 0:
                    print('SuperClasses::Update::Assign Classes: {}/{} Classes'.format(cls_id, self.num_cls))
            ## Task: update mu and cov, cov_inv, L, D2 for each supcls
            for supcls_id in range(self.num_supcls):
                self.supcls_cls_dict[supcls_id] = sorted(list(set(self.supcls_cls_dict[supcls_id])))
                self.mu_supcls[supcls_id] = self.raves[supcls_id].mx.to(self.device).clone()
                cov_1supcls = self.raves[supcls_id].cov_weighted().clone()
                self.cov_supcls[supcls_id] = cov_1supcls.cpu()
                if self.PPCA is True:
                    ## Task: apply SVD to cov_supcls
                    vT, S2_1supcls, v_1supcls = torch.linalg.svd(cov_1supcls)
                    self.L_supcls[supcls_id] = v_1supcls[0:self.q_supcls, :].clone()
                    # S_online[stp] = torch.sqrt(S2_online[0:q] * nc[stp]).clone()
                    self.D2_supcls[supcls_id] = S2_1supcls[0:self.q_supcls].clone()
                    if supcls_id % 100 == 0:
                        print('SupClasses::Update::cov_supcls svd: {}/{} sup-classes'.format(supcls_id, self.num_supcls))
                    self.cov_inv_supcls[supcls_id] = torch.eye(self.d) / self.lda - deltaDiag_D2(
                        self.L_supcls[supcls_id],
                        self.D2_supcls[supcls_id],
                        self.lda).cpu()
                else:
                    self.cov_inv_supcls[supcls_id] = torch.linalg.inv(cov_1supcls + self.small_jittering)
            print('Average KL-Divergence over all classes: {}'.format(torch.mean(kl_div_all_cls)))
        print('time cost - raw data:', time.time() - time0)

# Utils
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # import random
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def speed_up(K, S, T, A):
    return K/(S+T*A)

def speed_up_TA(K, S, TA):
    return K/(S+TA)

def speed_up_1obs(K, S, C):
    return K/(S+C)


setup_seed(seed=0)