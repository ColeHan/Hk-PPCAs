import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim

import scipy.io as io
from scipy.io import loadmat
from scipy.io import savemat

import os

import matplotlib.pyplot as plt
import math


# Utils
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # import random
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def DataGenerator2D(n=1000, dim=2,
                    mean=torch.tensor([[13., 50], [20, 51]]),
                    cov=torch.tensor([[[1., 0],
                                       [0, 2]],
                                      [[6., -3],
                                       [-3, 8]]]),
                    # pi=torch.tensor([0.5, 0.5]),
                    lda=0.01
                    ):
    num_cls = len(mean)
    clusters_dict = {}
    main0 = np.random.multivariate_normal(mean[0], cov[0], size=(n,), check_valid='raise')
    lda_diag0 = np.random.multivariate_normal(np.zeros(dim), lda * np.eye(dim), size=(n,), check_valid='raise')
    sample0 = main0 + lda_diag0
    clusters_dict[0] = sample0

    main1 = np.random.multivariate_normal(mean[1], cov[1], size=(n,), check_valid='raise')
    lda_diag1 = np.random.multivariate_normal(np.zeros(dim), lda * np.eye(dim), size=(n,), check_valid='raise')
    sample1 = main1 + lda_diag1
    clusters_dict[1] = sample1

    # pi = np.array(pi)
    # syn_index = [np.random.choice([i for i in range(num_cls)], p=pi.ravel()) for j in range(n)]
    # syn_values = torch.tensor([clusters_dict[syn_index[j]][j] for j in range(n)], dtype=torch.float)
    return clusters_dict


# def DataPlot2D(dataset, figure_id=1):
#     plt.figure(figure_id)
#     plt.title("Scatterplot")
#     plt.scatter(dataset[0][0], dataset[0][1], c=np.repeat(0,len(dataset[0])))
#     plt.scatter(dataset[1][0], dataset[1][1], c=np.repeat(1,len(dataset[1])))


def score(x, mean_cls, cov_cls):
    # x: n x d
    # mu_cls: d
    xc = x - mean_cls
    Xt = np.expand_dims(xc, -2)  # n x 1 x d
    X = np.expand_dims(xc, -1)  # n x d x 1
    score = Xt @ np.linalg.inv(cov_cls) @ X  # n x 1 x 1
    EuD = Xt @ X
    print(EuD)
    return score




if __name__ == '__main__':
    setup_seed(1)
    torch.set_printoptions(
        precision=1,  # 精度，保留小数点后几位，默认4
        threshold=1000,
        edgeitems=10,
        linewidth=150,  # 每行最多显示的字符数，默认80，超过则换行显示
        profile=None,
        sci_mode=False  # 用科学技术法显示数据，默认True
    )

    n = 50  # number of each class

    dataset = DataGenerator2D(n=50)
    all_dataset = np.concatenate(np.array([dataset[0], dataset[1]]), axis=0)

    dataset_sess1 = DataGenerator2D(n=50, dim=2,
                                    mean=torch.tensor([[14., 43], [23, 30]]),
                                    cov=torch.tensor([[[7, -1.2],
                                                       [-1.2, 3.2]],
                                                      [[2.7, 3],
                                                       [3, 9.3]]]),
                                    lda=0.01)
    all_dataset_sess1 = np.concatenate(np.array([dataset_sess1[0], dataset_sess1[1]]), axis=0)


    plt.figure(0)
    plt.xticks([])
    plt.yticks([])
    # plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c='red', alpha=0.5)
    # plt.scatter(dataset[1][:, 0], dataset[1][:, 1], c='steelblue', alpha=0.5)
    plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset[1][:, 0], dataset[1][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='grey', alpha=0.2)
    # plt.show()

    lab_id = [8, 3]

    plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')
    plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    plt.show()

    plt.figure(1)
    plt.xticks([])
    plt.yticks([])
    # plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c='red',alpha=0.5)
    # plt.scatter(dataset[1][:, 0], dataset[1][:, 1], c='steelblue',alpha=0.5)
    plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset[1][:, 0], dataset[1][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='grey', alpha=0.2)
    # plt.show()

    # lab_id = [8, 9, 10, 3, 4]
    lab_id = [8, 3]

    plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')
    plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    # plt.show()

    # Draw mean point from labeled obs
    ini_mean_cls1 = np.mean(dataset[0][lab_id], axis=0)
    ini_mean_cls2 = np.mean(dataset[1][lab_id], axis=0)
    plt.scatter(ini_mean_cls1[0], ini_mean_cls1[1], s=150, marker='x', c='red')
    plt.scatter(ini_mean_cls2[0], ini_mean_cls2[1], s=150, marker='x', c='blue')

    ini_mean_cls3 = np.mean(dataset_sess1[0][lab_id], axis=0)
    ini_mean_cls4 = np.mean(dataset_sess1[1][lab_id], axis=0)
    plt.scatter(ini_mean_cls3[0], ini_mean_cls3[1], s=150, marker='x', c='magenta')
    plt.scatter(ini_mean_cls4[0], ini_mean_cls4[1], s=150, marker='x', c='darkorange')
    plt.show()

    plt.figure(2)
    plt.xticks([])
    plt.yticks([])
    # plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c='red',alpha=0.5)
    # plt.scatter(dataset[1][:, 0], dataset[1][:, 1], c='steelblue',alpha=0.5)
    plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset[1][:, 0], dataset[1][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='grey', alpha=0.2)
    # plt.show()

    plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')
    plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    # plt.show()

    # Draw mean point from labeled obs
    ini_mean_cls1 = np.mean(dataset[0][lab_id], axis=0)
    ini_mean_cls2 = np.mean(dataset[1][lab_id], axis=0)
    plt.scatter(ini_mean_cls1[0], ini_mean_cls1[1], s=150, marker='x', c='red')
    plt.scatter(ini_mean_cls2[0], ini_mean_cls2[1], s=150, marker='x', c='blue')

    ini_mean_cls3 = np.mean(dataset_sess1[0][lab_id], axis=0)
    ini_mean_cls4 = np.mean(dataset_sess1[1][lab_id], axis=0)
    plt.scatter(ini_mean_cls3[0], ini_mean_cls3[1], s=150, marker='x', c='magenta')
    plt.scatter(ini_mean_cls4[0], ini_mean_cls4[1], s=150, marker='x', c='darkorange')


    # Draw normalized graph
    ini_cov_cls1 = np.cov(dataset[0][lab_id].T)
    ini_cov_cls2 = np.cov(dataset[1][lab_id].T)
    u1, s1, v1 = np.linalg.svd(ini_cov_cls1)
    u2, s2, v2 = np.linalg.svd(ini_cov_cls2)
    L1 = v1[0:1, :]
    L2 = v2[0:1, :]
    S1 = np.sqrt(s1[0:1] * n)
    S2 = np.sqrt(s2[0:1] * n)
    cov_cls1_reconst = L1.T @ L1 * (s1[0]) + 0.01 * np.eye(2)
    cov_cls2_reconst = L2.T @ L2 * (s2[0]) + 0.01 * np.eye(2)

    ## plot the first principal direction on the original data
    TP1 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 0], L1[0, 0]]) + ini_mean_cls1[0]
    TP2 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 1], L1[0, 1]]) + ini_mean_cls1[1]
    plt.plot(TP1, TP2, '-', color='red')
    TP1 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 0], L2[0, 0]]) + ini_mean_cls2[0]
    TP2 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 1], L2[0, 1]]) + ini_mean_cls2[1]
    plt.plot(TP1, TP2, '-', color='blue')

    # Draw normalized graph
    ini_cov_cls3 = np.cov(dataset_sess1[0][lab_id].T)
    ini_cov_cls4 = np.cov(dataset_sess1[1][lab_id].T)

    ## plot the first principal direction on the original data
    u3, s3, v3 = np.linalg.svd(ini_cov_cls3)
    u4, s4, v4 = np.linalg.svd(ini_cov_cls4)
    L3 = v3[0:1, :]
    L4 = v4[0:1, :]
    S3 = np.sqrt(s3[0:1] * n)
    S4 = np.sqrt(s4[0:1] * n)
    cov_cls3_reconst = L3.T @ L3 * (s3[0]) + 0.01 * np.eye(2)
    cov_cls4_reconst = L4.T @ L4 * (s4[0]) + 0.01 * np.eye(2)

    ## plot the first principal direction on the original data
    TP3 = 2 * S3[0] / (n ** 0.5) * np.array([-L3[0, 0], L3[0, 0]]) + ini_mean_cls3[0]
    TP4 = 2 * S3[0] / (n ** 0.5) * np.array([-L3[0, 1], L3[0, 1]]) + ini_mean_cls3[1]
    plt.plot(TP3, TP4, '-', color='magenta')
    TP3 = 2 * S4[0] / (n ** 0.5) * np.array([-L4[0, 0], L4[0, 0]]) + ini_mean_cls4[0]
    TP4 = 2 * S4[0] / (n ** 0.5) * np.array([-L4[0, 1], L4[0, 1]]) + ini_mean_cls4[1]
    plt.plot(TP3, TP4, '-', color='darkorange')
    plt.show()


    # plt true result
    plt.figure(5)
    plt.xticks([])
    plt.yticks([])
    scores = np.empty([n * 2, 2])
    score_data1_cls1 = score(dataset[0], ini_mean_cls1, cov_cls1_reconst).squeeze()
    scores[:50, 0] = score_data1_cls1
    score_data2_cls1 = score(dataset[1], ini_mean_cls1, cov_cls1_reconst).squeeze()
    scores[50:, 0] = score_data2_cls1
    score_data1_cls2 = score(dataset[0], ini_mean_cls2, cov_cls2_reconst).squeeze()
    scores[:50, 1] = score_data1_cls2
    score_data2_cls2 = score(dataset[1], ini_mean_cls2, cov_cls2_reconst).squeeze()
    scores[50:, 1] = score_data2_cls2
    assigned_labels = np.argmin(scores, axis=1)
    assigned_labels[lab_id] = np.repeat(0, 2)
    assigned_labels[[id + 50 for id in lab_id]] = np.repeat(1, 2)

    new_dataset = {0: [], 1: []}
    for i in range(2 * n):
        if assigned_labels[i] == 0:
            new_dataset[0].append(all_dataset[i])
        else:
            new_dataset[1].append(all_dataset[i])
    new_dataset[0] = np.array(new_dataset[0])
    new_dataset[1] = np.array(new_dataset[1])
    # print(new_dataset[0])

    # plt.scatter(new_dataset[0][:, 0], new_dataset[0][:, 1], c='red')
    # # plt.scatter(dataset[1][:, 0], dataset[1][:, 1], c=np.repeat(20, len(dataset[1])))
    # plt.scatter(new_dataset[1][:, 0], new_dataset[1][:, 1], c='blue')

    plt.scatter(new_dataset[0][:, 0], new_dataset[0][:, 1], c='red', alpha=0.5)
    plt.scatter(new_dataset[1][:, 0], new_dataset[1][:, 1], c='steelblue', alpha=0.5)
    plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')

    # plt true label
    plt.scatter(new_dataset[0][:, 0], new_dataset[0][:, 1], c='red', alpha=0.5)
    plt.scatter(new_dataset[1][:, 0], new_dataset[1][:, 1], c='steelblue', alpha=0.5)
    plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')
    plt.scatter(ini_mean_cls1[0], ini_mean_cls1[1], s=150, marker='x', c='red')
    plt.scatter(ini_mean_cls2[0], ini_mean_cls2[1], s=150, marker='x', c='blue')
    ## plot the first principal direction on the original data
    TP1 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 0], L1[0, 0]]) + ini_mean_cls1[0]
    TP2 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 1], L1[0, 1]]) + ini_mean_cls1[1]
    plt.plot(TP1, TP2, 'r-')
    TP1 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 0], L2[0, 0]]) + ini_mean_cls2[0]
    TP2 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 1], L2[0, 1]]) + ini_mean_cls2[1]
    plt.plot(TP1, TP2, 'g-')

    #############################

    plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='darkmagenta', alpha=0.2)
    plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='orange', alpha=0.2)

    # Draw mean point from labeled obs
    ini_mean_cls3 = np.mean(dataset_sess1[0][lab_id], axis=0)
    ini_mean_cls4 = np.mean(dataset_sess1[1][lab_id], axis=0)
    plt.scatter(ini_mean_cls3[0], ini_mean_cls3[1], s=150, marker='x', c='magenta')
    plt.scatter(ini_mean_cls4[0], ini_mean_cls4[1], s=150, marker='x', c='darkorange')

    # Draw normalized graph
    ini_cov_cls3 = np.cov(dataset_sess1[0][lab_id].T)
    ini_cov_cls4 = np.cov(dataset_sess1[1][lab_id].T)

    ## plot the first principal direction on the original data
    u3, s3, v3 = np.linalg.svd(ini_cov_cls3)
    u4, s4, v4 = np.linalg.svd(ini_cov_cls4)
    L3 = v3[0:1, :]
    L4 = v4[0:1, :]
    S3 = np.sqrt(s3[0:1] * n)
    S4 = np.sqrt(s4[0:1] * n)
    cov_cls3_reconst = L3.T @ L3 * (s3[0]) + 0.01 * np.eye(2)
    cov_cls4_reconst = L4.T @ L4 * (s4[0]) + 0.01 * np.eye(2)

    ## plot the first principal direction on the original data
    TP3 = 2 * S3[0] / (n ** 0.5) * np.array([-L3[0, 0], L3[0, 0]]) + ini_mean_cls3[0]
    TP4 = 2 * S3[0] / (n ** 0.5) * np.array([-L3[0, 1], L3[0, 1]]) + ini_mean_cls3[1]
    plt.plot(TP3, TP4, '-', color='magenta')
    TP3 = 2 * S4[0] / (n ** 0.5) * np.array([-L4[0, 0], L4[0, 0]]) + ini_mean_cls4[0]
    TP4 = 2 * S4[0] / (n ** 0.5) * np.array([-L4[0, 1], L4[0, 1]]) + ini_mean_cls4[1]
    plt.plot(TP3, TP4, '-', color='darkorange')

    # plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    # plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')

    plt.show()

    # plt true result
    plt.figure(5)
    plt.xticks([])
    plt.yticks([])
    scores = np.empty([n * 2, 2])
    score_data1_cls1 = score(dataset[0], ini_mean_cls1, cov_cls1_reconst).squeeze()
    scores[:50, 0] = score_data1_cls1
    score_data2_cls1 = score(dataset[1], ini_mean_cls1, cov_cls1_reconst).squeeze()
    scores[50:, 0] = score_data2_cls1
    score_data1_cls2 = score(dataset[0], ini_mean_cls2, cov_cls2_reconst).squeeze()
    scores[:50, 1] = score_data1_cls2
    score_data2_cls2 = score(dataset[1], ini_mean_cls2, cov_cls2_reconst).squeeze()
    scores[50:, 1] = score_data2_cls2
    assigned_labels = np.argmin(scores, axis=1)
    assigned_labels[lab_id] = np.repeat(0, 2)
    assigned_labels[[id + 50 for id in lab_id]] = np.repeat(1, 2)

    new_dataset = {0: [], 1: []}
    for i in range(2 * n):
        if assigned_labels[i] == 0:
            new_dataset[0].append(all_dataset[i])
        else:
            new_dataset[1].append(all_dataset[i])
    new_dataset[0] = np.array(new_dataset[0])
    new_dataset[1] = np.array(new_dataset[1])
    # print(new_dataset[0])

    # plt.scatter(new_dataset[0][:, 0], new_dataset[0][:, 1], c='red')
    # # plt.scatter(dataset[1][:, 0], dataset[1][:, 1], c=np.repeat(20, len(dataset[1])))
    # plt.scatter(new_dataset[1][:, 0], new_dataset[1][:, 1], c='blue')

    plt.scatter(new_dataset[0][:, 0], new_dataset[0][:, 1], c='red', alpha=0.5)
    plt.scatter(new_dataset[1][:, 0], new_dataset[1][:, 1], c='steelblue', alpha=0.5)
    plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')

    # plt true label
    plt.scatter(new_dataset[0][:, 0], new_dataset[0][:, 1], c='red', alpha=0.5)
    plt.scatter(new_dataset[1][:, 0], new_dataset[1][:, 1], c='steelblue', alpha=0.5)
    plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')
    # plt.scatter(ini_mean_cls1[0], ini_mean_cls1[1], s=150, marker='x', c='red')
    # plt.scatter(ini_mean_cls2[0], ini_mean_cls2[1], s=150, marker='x', c='blue')
    # ## plot the first principal direction on the original data
    # TP1 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 0], L1[0, 0]]) + ini_mean_cls1[0]
    # TP2 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 1], L1[0, 1]]) + ini_mean_cls1[1]
    # plt.plot(TP1, TP2, 'r-')
    # TP1 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 0], L2[0, 0]]) + ini_mean_cls2[0]
    # TP2 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 1], L2[0, 1]]) + ini_mean_cls2[1]
    # plt.plot(TP1, TP2, 'g-')

    #############################

    plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='darkmagenta', alpha=0.2)
    plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='orange', alpha=0.2)

    # Draw mean point from labeled obs
    ini_mean_cls3 = np.mean(dataset_sess1[0][lab_id], axis=0)
    ini_mean_cls4 = np.mean(dataset_sess1[1][lab_id], axis=0)
    # plt.scatter(ini_mean_cls3[0], ini_mean_cls3[1], s=150, marker='x', c='magenta')
    # plt.scatter(ini_mean_cls4[0], ini_mean_cls4[1], s=150, marker='x', c='darkorange')

    # Draw normalized graph
    ini_cov_cls3 = np.cov(dataset_sess1[0][lab_id].T)
    ini_cov_cls4 = np.cov(dataset_sess1[1][lab_id].T)

    ## plot the first principal direction on the original data
    u3, s3, v3 = np.linalg.svd(ini_cov_cls3)
    u4, s4, v4 = np.linalg.svd(ini_cov_cls4)
    L3 = v3[0:1, :]
    L4 = v4[0:1, :]
    S3 = np.sqrt(s3[0:1] * n)
    S4 = np.sqrt(s4[0:1] * n)
    cov_cls3_reconst = L3.T @ L3 * (s3[0]) + 0.01 * np.eye(2)
    cov_cls4_reconst = L4.T @ L4 * (s4[0]) + 0.01 * np.eye(2)

    # ## plot the first principal direction on the original data
    # TP3 = 2 * S3[0] / (n ** 0.5) * np.array([-L3[0, 0], L3[0, 0]]) + ini_mean_cls3[0]
    # TP4 = 2 * S3[0] / (n ** 0.5) * np.array([-L3[0, 1], L3[0, 1]]) + ini_mean_cls3[1]
    # plt.plot(TP3, TP4, '-', color='magenta')
    # TP3 = 2 * S4[0] / (n ** 0.5) * np.array([-L4[0, 0], L4[0, 0]]) + ini_mean_cls4[0]
    # TP4 = 2 * S4[0] / (n ** 0.5) * np.array([-L4[0, 1], L4[0, 1]]) + ini_mean_cls4[1]
    # plt.plot(TP3, TP4, '-', color='darkorange')

    # plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    # plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')

    plt.show()

    plt.figure(6)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(new_dataset[0][:, 0], new_dataset[0][:, 1], c='red', alpha=0.5)
    plt.scatter(new_dataset[1][:, 0], new_dataset[1][:, 1], c='steelblue', alpha=0.5)
    plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')

    # Draw mean point from labeled obs
    mean_cls1 = np.mean(new_dataset[0], axis=0)
    mean_cls2 = np.mean(new_dataset[1], axis=0)
    plt.scatter(mean_cls1[0], mean_cls1[1], s=150, marker='x', c='red')
    plt.scatter(mean_cls2[0], mean_cls2[1], s=150, marker='x', c='blue')

    # Draw normalized graph
    cov_cls1 = np.cov(new_dataset[0].T)
    cov_cls2 = np.cov(new_dataset[1].T)
    u1, s1, v1 = np.linalg.svd(cov_cls1)
    u2, s2, v2 = np.linalg.svd(cov_cls2)
    L1 = v1[0:1, :]
    L2 = v2[0:1, :]
    S1 = np.sqrt(s1[0:1] * n)
    S2 = np.sqrt(s2[0:1] * n)
    cov_cls1_reconst = L1.T @ L1 * (s1[0]) + 0.01 * np.eye(2)
    cov_cls2_reconst = L2.T @ L2 * (s2[0]) + 0.01 * np.eye(2)

    ## plot the first principal direction on the original data
    TP1 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 0], L1[0, 0]]) + mean_cls1[0]
    TP2 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 1], L1[0, 1]]) + mean_cls1[1]
    plt.plot(TP1, TP2, '-', color='red')
    TP1 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 0], L2[0, 0]]) + mean_cls2[0]
    TP2 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 1], L2[0, 1]]) + mean_cls2[1]
    plt.plot(TP1, TP2, '-', color='blue')


    plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='darkmagenta', alpha=0.2)
    plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='orange', alpha=0.2)
    plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')

    # Draw mean point from labeled obs
    mean_cls3 = np.mean(dataset_sess1[0], axis=0)
    mean_cls4 = np.mean(dataset_sess1[1], axis=0)
    plt.scatter(mean_cls3[0], mean_cls3[1], s=150, marker='x', c='magenta')
    plt.scatter(mean_cls4[0], mean_cls4[1], s=150, marker='x', c='darkorange')

    # Draw normalized graph
    cov_cls3 = np.cov(dataset_sess1[0].T)
    cov_cls4 = np.cov(dataset_sess1[1].T)
    u3, s3, v3 = np.linalg.svd(cov_cls3)
    u4, s4, v4 = np.linalg.svd(cov_cls4)
    L3 = v3[0:1, :]
    L4 = v4[0:1, :]
    S3 = np.sqrt(s3[0:1] * n)
    S4 = np.sqrt(s4[0:1] * n)

    ## plot the first principal direction on the original data
    TP3 = 2 * S3[0] / (n ** 0.5) * np.array([-L3[0, 0], L3[0, 0]]) + mean_cls3[0]
    TP4 = 2 * S3[0] / (n ** 0.5) * np.array([-L3[0, 1], L3[0, 1]]) + mean_cls3[1]
    plt.plot(TP3, TP4, '-', color='magenta')
    TP3 = 2 * S4[0] / (n ** 0.5) * np.array([-L4[0, 0], L4[0, 0]]) + mean_cls4[0]
    TP4 = 2 * S4[0] / (n ** 0.5) * np.array([-L4[0, 1], L4[0, 1]]) + mean_cls4[1]
    plt.plot(TP3, TP4, '-', color='darkorange')
    plt.show()


###############################################################################    #################################################################################
    dataset_sess1 = DataGenerator2D(n=50, dim=2,
                                    mean=torch.tensor([[14., 43], [23, 30]]),
                                    cov=torch.tensor([[[7, -1.2],
                                                       [-1.2, 3.2]],
                                                      [[2.7, 3],
                                                       [3, 9.3]]]),
                                    lda=0.01)
    all_dataset_sess1 = np.concatenate(np.array([dataset_sess1[0], dataset_sess1[1]]), axis=0)

    plt.figure(0)
    plt.xticks([])
    plt.yticks([])
    # plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c='red', alpha=0.5)
    # plt.scatter(dataset[1][:, 0], dataset[1][:, 1], c='steelblue', alpha=0.5)
    plt.scatter(new_dataset[0][:, 0], new_dataset[0][:, 1], c='red', alpha=0.5)
    plt.scatter(new_dataset[1][:, 0], new_dataset[1][:, 1], c='steelblue', alpha=0.5)
    plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')
    plt.scatter(mean_cls1[0], mean_cls1[1], s=150, marker='x', c='red')
    plt.scatter(mean_cls2[0], mean_cls2[1], s=150, marker='x', c='blue')
    ## plot the first principal direction on the original data
    TP1 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 0], L1[0, 0]]) + mean_cls1[0]
    TP2 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 1], L1[0, 1]]) + mean_cls1[1]
    plt.plot(TP1, TP2, 'r-', )
    TP1 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 0], L2[0, 0]]) + mean_cls2[0]
    TP2 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 1], L2[0, 1]]) + mean_cls2[1]
    plt.plot(TP1, TP2, 'g-')

    plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='grey', alpha=0.2)
    # plt.show()

    lab_id = [8, 9, 10, 3, 4]
    plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    plt.show()

    plt.figure(1)
    plt.xticks([])
    plt.yticks([])
    # plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c='red',alpha=0.5)
    # plt.scatter(dataset[1][:, 0], dataset[1][:, 1], c='steelblue',alpha=0.5)
    plt.scatter(new_dataset[0][:, 0], new_dataset[0][:, 1], c='red', alpha=0.5)
    plt.scatter(new_dataset[1][:, 0], new_dataset[1][:, 1], c='steelblue', alpha=0.5)
    plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')
    plt.scatter(mean_cls1[0], mean_cls1[1], s=150, marker='x', c='red')
    plt.scatter(mean_cls2[0], mean_cls2[1], s=150, marker='x', c='blue')
    ## plot the first principal direction on the original data
    TP1 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 0], L1[0, 0]]) + mean_cls1[0]
    TP2 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 1], L1[0, 1]]) + mean_cls1[1]
    plt.plot(TP1, TP2, 'r-')
    TP1 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 0], L2[0, 0]]) + mean_cls2[0]
    TP2 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 1], L2[0, 1]]) + mean_cls2[1]
    plt.plot(TP1, TP2, 'g-')

    plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='grey', alpha=0.2)
    # plt.show()

    plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    # plt.show()

    # Draw mean point from labeled obs
    ini_mean_cls3 = np.mean(dataset_sess1[0][lab_id], axis=0)
    ini_mean_cls4 = np.mean(dataset_sess1[1][lab_id], axis=0)
    plt.scatter(ini_mean_cls3[0], ini_mean_cls3[1], s=150, marker='x', c='magenta')
    plt.scatter(ini_mean_cls4[0], ini_mean_cls4[1], s=150, marker='x', c='darkorange')
    plt.show()

    plt.figure(2)
    plt.xticks([])
    plt.yticks([])

    plt.scatter(new_dataset[0][:, 0], new_dataset[0][:, 1], c='red', alpha=0.5)
    plt.scatter(new_dataset[1][:, 0], new_dataset[1][:, 1], c='steelblue', alpha=0.5)
    plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')
    plt.scatter(mean_cls1[0], mean_cls1[1], s=150, marker='x', c='red')
    plt.scatter(mean_cls2[0], mean_cls2[1], s=150, marker='x', c='blue')
    ## plot the first principal direction on the original data
    TP1 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 0], L1[0, 0]]) + mean_cls1[0]
    TP2 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 1], L1[0, 1]]) + mean_cls1[1]
    plt.plot(TP1, TP2, 'r-')
    TP1 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 0], L2[0, 0]]) + mean_cls2[0]
    TP2 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 1], L2[0, 1]]) + mean_cls2[1]
    plt.plot(TP1, TP2, 'g-')

    plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='grey', alpha=0.2)
    # plt.show()

    plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    # plt.show()

    # Draw mean point from labeled obs
    ini_mean_cls3 = np.mean(dataset_sess1[0][lab_id], axis=0)
    ini_mean_cls4 = np.mean(dataset_sess1[1][lab_id], axis=0)
    plt.scatter(ini_mean_cls3[0], ini_mean_cls3[1], s=150, marker='x', c='magenta')
    plt.scatter(ini_mean_cls4[0], ini_mean_cls4[1], s=150, marker='x', c='darkorange')

    ####################
    plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='grey', alpha=0.2)
    # plt.show()

    plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    # plt.show()

    # Draw mean point from labeled obs
    ini_mean_cls3 = np.mean(dataset_sess1[0][lab_id], axis=0)
    ini_mean_cls4 = np.mean(dataset_sess1[1][lab_id], axis=0)
    plt.scatter(ini_mean_cls3[0], ini_mean_cls3[1], s=150, marker='x', c='magenta')
    plt.scatter(ini_mean_cls4[0], ini_mean_cls4[1], s=150, marker='x', c='darkorange')

    # Draw normalized graph
    ini_cov_cls3 = np.cov(dataset_sess1[0][lab_id].T)
    ini_cov_cls4 = np.cov(dataset_sess1[1][lab_id].T)

    ## plot the first principal direction on the original data
    u3, s3, v3 = np.linalg.svd(ini_cov_cls3)
    u4, s4, v4 = np.linalg.svd(ini_cov_cls4)
    L3 = v3[0:1, :]
    L4 = v4[0:1, :]
    S3 = np.sqrt(s3[0:1] * n)
    S4 = np.sqrt(s4[0:1] * n)
    cov_cls3_reconst = L3.T @ L3 * (s3[0]) + 0.01 * np.eye(2)
    cov_cls4_reconst = L4.T @ L4 * (s4[0]) + 0.01 * np.eye(2)

    ## plot the first principal direction on the original data
    TP3 = 2 * S3[0] / (n ** 0.5) * np.array([-L3[0, 0], L3[0, 0]]) + ini_mean_cls3[0]
    TP4 = 2 * S3[0] / (n ** 0.5) * np.array([-L3[0, 1], L3[0, 1]]) + ini_mean_cls3[1]
    plt.plot(TP3, TP4, '-', color='magenta')
    TP3 = 2 * S4[0] / (n ** 0.5) * np.array([-L4[0, 0], L4[0, 0]]) + ini_mean_cls4[0]
    TP4 = 2 * S4[0] / (n ** 0.5) * np.array([-L4[0, 1], L4[0, 1]]) + ini_mean_cls4[1]
    plt.plot(TP3, TP4, '-', color='darkorange')
    plt.show()

    # normalize with class1 's cov
    sqrt_cov_cls3_reconst = L3.T @ L3 * (np.sqrt(s3[0])) + 0.01 * np.eye(2)
    sqrt_cov_cls4_reconst = L4.T @ L4 * (np.sqrt(s4[0])) + 0.01 * np.eye(2)

    plt.figure(3)
    plt.xticks([])
    plt.yticks([])

    norm_data3_cls1 = (dataset_sess1[0] - ini_mean_cls1) @ np.linalg.inv(sqrt_cov_cls1_reconst)
    # norm_data2=(dataset[1] - ini_mean_cls2) / np.diag(cov_cls2_reconst) / np.sqrt(2)
    norm_data4_cls1 = (dataset_sess1[1] - ini_mean_cls1) @ np.linalg.inv(sqrt_cov_cls1_reconst)
    # norm_ini_mean_cls1 = (ini_mean_cls1- ini_mean_cls1) @ np.linalg.inv(sqrt_cov_cls1_reconst)

    plt.scatter(norm_data3_cls1[:, 0], norm_data3_cls1[:, 1], c='grey', alpha=0.2)
    plt.scatter(norm_data4_cls1[:, 0], norm_data4_cls1[:, 1], c='grey', alpha=0.2)

    plt.scatter(norm_data3_cls1[lab_id, 0], norm_data3_cls1[lab_id, 1], c='magenta')
    plt.scatter(norm_data4_cls1[lab_id, 0], norm_data4_cls1[lab_id, 1], c='darkorange')

    plt.scatter(norm_ini_mean_cls1[0], norm_ini_mean_cls1[1], s=150, marker='x', c='red')

    plt.show()

    plt.figure(4)
    plt.xticks([])
    plt.yticks([])

    norm_data3_cls2 = (dataset_sess1[0] - ini_mean_cls2) @ np.linalg.inv(sqrt_cov_cls2_reconst)
    # norm_data2=(dataset[1] - ini_mean_cls2) / np.diag(cov_cls2_reconst) / np.sqrt(2)
    norm_data4_cls2 = (dataset_sess1[1] - ini_mean_cls2) @ np.linalg.inv(sqrt_cov_cls2_reconst)
    # norm_ini_mean_cls1 = (ini_mean_cls1- ini_mean_cls1) @ np.linalg.inv(sqrt_cov_cls1_reconst)

    plt.scatter(norm_data3_cls2[:, 0], norm_data3_cls2[:, 1], c='grey', alpha=0.2)
    plt.scatter(norm_data4_cls2[:, 0], norm_data4_cls2[:, 1], c='grey', alpha=0.2)

    plt.scatter(norm_data3_cls2[lab_id, 0], norm_data3_cls2[lab_id, 1], c='magenta')
    plt.scatter(norm_data4_cls2[lab_id, 0], norm_data4_cls2[lab_id, 1], c='darkorange')

    plt.scatter(norm_ini_mean_cls2[0], norm_ini_mean_cls2[1], s=150, marker='x', c='blue')

    plt.show()

    plt.figure(5)
    plt.xticks([])
    plt.yticks([])
    norm_data3_cls3 = (dataset_sess1[0] - ini_mean_cls3) @ np.linalg.inv(sqrt_cov_cls3_reconst)
    # norm_data2=(dataset[1] - ini_mean_cls2) / np.diag(cov_cls2_reconst) / np.sqrt(2)
    norm_data4_cls3 = (dataset_sess1[1] - ini_mean_cls3) @ np.linalg.inv(sqrt_cov_cls3_reconst)
    norm_ini_mean_cls3 = (ini_mean_cls3 - ini_mean_cls3) @ np.linalg.inv(sqrt_cov_cls3_reconst)

    plt.scatter(norm_data3_cls3[:, 0], norm_data3_cls3[:, 1], c='grey', alpha=0.2)
    plt.scatter(norm_data4_cls3[:, 0], norm_data4_cls3[:, 1], c='grey', alpha=0.2)

    plt.scatter(norm_data3_cls3[lab_id, 0], norm_data3_cls3[lab_id, 1], c='magenta')
    plt.scatter(norm_data4_cls3[lab_id, 0], norm_data4_cls3[lab_id, 1], c='darkorange')

    plt.scatter(norm_ini_mean_cls3[0], norm_ini_mean_cls3[1], s=150, marker='x', c='magenta')
    plt.show()

    plt.figure(6)
    plt.xticks([])
    plt.yticks([])
    norm_data3_cls4 = (dataset_sess1[0] - ini_mean_cls4) @ np.linalg.inv(sqrt_cov_cls4_reconst)
    norm_data4_cls4 = (dataset_sess1[1] - ini_mean_cls4) @ np.linalg.inv(sqrt_cov_cls4_reconst)
    norm_ini_mean_cls4 = (ini_mean_cls4 - ini_mean_cls4) @ np.linalg.inv(sqrt_cov_cls4_reconst)

    plt.scatter(norm_data3_cls4[:, 0], norm_data3_cls4[:, 1], c='grey', alpha=0.2)
    plt.scatter(norm_data4_cls4[:, 0], norm_data4_cls4[:, 1], c='grey', alpha=0.2)

    plt.scatter(norm_data3_cls4[lab_id, 0], norm_data3_cls4[lab_id, 1], c='magenta')
    plt.scatter(norm_data4_cls4[lab_id, 0], norm_data4_cls4[lab_id, 1], c='darkorange')

    plt.scatter(norm_ini_mean_cls4[0], norm_ini_mean_cls4[1], s=150, marker='x', c='darkorange')
    plt.show()

    # plt result
    # plt.figure(7)
    # plt.xticks([])
    # plt.yticks([])
    #
    # plt.scatter(new_dataset[0][:, 0], new_dataset[0][:, 1], c='red', alpha=0.5)
    # plt.scatter(new_dataset[1][:, 0], new_dataset[1][:, 1], c='steelblue', alpha=0.5)
    # plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    # plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')
    # plt.scatter(mean_cls1[0], mean_cls1[1], s=150, marker='x', c='red')
    # plt.scatter(mean_cls2[0], mean_cls2[1], s=150, marker='x', c='blue')
    # ## plot the first principal direction on the original data
    # TP1 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 0], L1[0, 0]]) + mean_cls1[0]
    # TP2 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 1], L1[0, 1]]) + mean_cls1[1]
    # plt.plot(TP1, TP2, 'r-')
    # TP1 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 0], L2[0, 0]]) + mean_cls2[0]
    # TP2 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 1], L2[0, 1]]) + mean_cls2[1]
    # plt.plot(TP1, TP2, 'g-')
    #
    # plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='grey', alpha=0.2)
    # plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='grey', alpha=0.2)
    # # plt.show()
    #
    # plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    # plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    # # plt.show()
    #
    # # Draw mean point from labeled obs
    # ini_mean_cls3 = np.mean(dataset_sess1[0][lab_id], axis=0)
    # ini_mean_cls4 = np.mean(dataset_sess1[1][lab_id], axis=0)
    # plt.scatter(ini_mean_cls3[0], ini_mean_cls3[1], s=150, marker='x', c='magenta')
    # plt.scatter(ini_mean_cls4[0], ini_mean_cls4[1], s=150, marker='x', c='darkorange')
    #
    #
    # ######################
    # scores_sess1=np.empty([n*2,4])
    # score_data3_cls1=score(dataset_sess1[0],ini_mean_cls1,cov_cls1_reconst).squeeze()
    # scores_sess1[:50,0]=score_data3_cls1
    # score_data4_cls1=score(dataset_sess1[1],ini_mean_cls1,cov_cls1_reconst).squeeze()
    # scores_sess1[50:,0]=score_data4_cls1
    # score_data3_cls2=score(dataset_sess1[0],ini_mean_cls2,cov_cls2_reconst).squeeze()
    # scores_sess1[:50,1]=score_data3_cls2
    # score_data4_cls2=score(dataset_sess1[1],ini_mean_cls2,cov_cls2_reconst).squeeze()
    # scores_sess1[50:,1]=score_data4_cls2
    # score_data3_cls3=score(dataset_sess1[0],ini_mean_cls3,cov_cls3_reconst).squeeze()
    # scores_sess1[:50,2]=score_data3_cls3
    # score_data4_cls3=score(dataset_sess1[1],ini_mean_cls3,cov_cls3_reconst).squeeze()
    # scores_sess1[50:,2]=score_data4_cls3
    # score_data3_cls4=score(dataset_sess1[0],ini_mean_cls4,cov_cls4_reconst).squeeze()
    # scores_sess1[:50,3]=score_data3_cls4
    # score_data4_cls4=score(dataset_sess1[1],ini_mean_cls4,cov_cls4_reconst).squeeze()
    # scores_sess1[50:,3]=score_data4_cls4
    # assigned_labels_sess1=np.argmin(scores_sess1,axis=1)
    # assigned_labels_sess1[lab_id]=np.repeat(2,5)
    # assigned_labels_sess1[[id+50 for id in lab_id]] =np.repeat(3,5)
    #
    # new_dataset_sess1={0:[],1:[],2:[],3:[]}
    # for i in range(2*n):
    #     if assigned_labels_sess1[i]==0:
    #         new_dataset_sess1[0].append(all_dataset_sess1[i])
    #     elif assigned_labels_sess1[i]==1:
    #         new_dataset_sess1[1].append(all_dataset_sess1[i])
    #     elif assigned_labels_sess1[i]==2:
    #         new_dataset_sess1[2].append(all_dataset_sess1[i])
    #     else:
    #         new_dataset_sess1[3].append(all_dataset_sess1[i])
    # new_dataset_sess1[0] = np.array(new_dataset_sess1[0])
    # new_dataset_sess1[1] = np.array(new_dataset_sess1[1])
    # new_dataset_sess1[2]=np.array(new_dataset_sess1[2])
    # new_dataset_sess1[3]=np.array(new_dataset_sess1[3])
    # # print(new_dataset[0])
    #
    # plt.scatter(new_dataset_sess1[2][:, 0], new_dataset_sess1[2][:, 1], c='darkmagenta',alpha=0.2)
    # plt.scatter(new_dataset_sess1[3][:, 0], new_dataset_sess1[3][:, 1], c='orange',alpha=0.2)
    # plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    # plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    # plt.show()
    #
    #
    # plt.figure(8)
    # plt.xticks([])
    # plt.yticks([])
    #
    #
    # plt.scatter(new_dataset[0][:, 0], new_dataset[0][:, 1], c='red', alpha=0.5)
    # plt.scatter(new_dataset[1][:, 0], new_dataset[1][:, 1], c='steelblue', alpha=0.5)
    # plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    # plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')
    # plt.scatter(mean_cls1[0], mean_cls1[1], s=150, marker='x', c='red')
    # plt.scatter(mean_cls2[0], mean_cls2[1], s=150, marker='x', c='blue')
    # ## plot the first principal direction on the original data
    # TP1 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 0], L1[0, 0]]) + mean_cls1[0]
    # TP2 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 1], L1[0, 1]]) + mean_cls1[1]
    # plt.plot(TP1, TP2, 'r-')
    # TP1 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 0], L2[0, 0]]) + mean_cls2[0]
    # TP2 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 1], L2[0, 1]]) + mean_cls2[1]
    # plt.plot(TP1, TP2, 'g-')
    #
    # plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='grey', alpha=0.2)
    # plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='grey', alpha=0.2)
    # # plt.show()
    #
    # plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    # plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    # # plt.show()
    #
    # # Draw mean point from labeled obs
    # ini_mean_cls3 = np.mean(dataset_sess1[0][lab_id], axis=0)
    # ini_mean_cls4 = np.mean(dataset_sess1[1][lab_id], axis=0)
    # plt.scatter(ini_mean_cls3[0], ini_mean_cls3[1], s=150, marker='x', c='magenta')
    # plt.scatter(ini_mean_cls4[0], ini_mean_cls4[1], s=150, marker='x', c='darkorange')
    #
    #
    # #############################
    # plt.scatter(new_dataset_sess1[2][:, 0], new_dataset_sess1[2][:, 1], c='darkmagenta', alpha=0.2)
    # plt.scatter(new_dataset_sess1[3][:, 0], new_dataset_sess1[3][:, 1], c='orange', alpha=0.2)
    # plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    # plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    #
    # # Draw mean point from labeled obs
    # mean_cls3=np.mean(new_dataset_sess1[2],axis=0)
    # mean_cls4=np.mean(new_dataset_sess1[3],axis=0)
    # plt.scatter(mean_cls3[0], mean_cls3[1], s=150, marker='x', c='magenta')
    # plt.scatter(mean_cls4[0], mean_cls4[1], s=150, marker='x', c='darkorange')
    #
    # # Draw normalized graph
    # cov_cls3=np.cov(new_dataset_sess1[2].T)
    # cov_cls4=np.cov(new_dataset_sess1[3].T)
    # u3, s3, v3 = np.linalg.svd(cov_cls3)
    # u4, s4, v4 = np.linalg.svd(cov_cls4)
    # L3 = v3[0:1, :]
    # L4 = v4[0:1, :]
    # S3 = np.sqrt(s3[0:1] * n)
    # S4 = np.sqrt(s4[0:1] * n)
    # cov_cls3_reconst=L3.T@L3*(s3[0])+0.01*np.eye(2)
    # cov_cls4_reconst = L4.T @ L4 * (s4[0]) + 0.01 * np.eye(2)
    #
    # ## plot the first principal direction on the original data
    # TP3 = 2 * S3[0] / (n ** 0.5) * np.array([-L3[0, 0], L3[0, 0]]) + mean_cls3[0]
    # TP4 = 2 * S3[0] / (n ** 0.5) * np.array([-L3[0, 1], L3[0, 1]]) + mean_cls3[1]
    # plt.plot(TP3, TP4, '-',color='magenta')
    # TP3 = 2 * S4[0] / (n ** 0.5) * np.array([-L4[0, 0], L4[0, 0]]) + mean_cls4[0]
    # TP4 = 2 * S4[0] / (n ** 0.5) * np.array([-L4[0, 1], L4[0, 1]]) + mean_cls4[1]
    # plt.plot(TP3, TP4, '-',color='darkorange')
    # plt.show()

    # plt true label
    plt.figure(9)
    plt.xticks([])
    plt.yticks([])

    plt.scatter(new_dataset[0][:, 0], new_dataset[0][:, 1], c='red', alpha=0.5)
    plt.scatter(new_dataset[1][:, 0], new_dataset[1][:, 1], c='steelblue', alpha=0.5)
    plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')
    plt.scatter(mean_cls1[0], mean_cls1[1], s=150, marker='x', c='red')
    plt.scatter(mean_cls2[0], mean_cls2[1], s=150, marker='x', c='blue')
    ## plot the first principal direction on the original data
    TP1 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 0], L1[0, 0]]) + mean_cls1[0]
    TP2 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 1], L1[0, 1]]) + mean_cls1[1]
    plt.plot(TP1, TP2, 'r-')
    TP1 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 0], L2[0, 0]]) + mean_cls2[0]
    TP2 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 1], L2[0, 1]]) + mean_cls2[1]
    plt.plot(TP1, TP2, 'g-')

    plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='grey', alpha=0.2)
    # plt.show()

    plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    # plt.show()

    # Draw mean point from labeled obs
    ini_mean_cls3 = np.mean(dataset_sess1[0][lab_id], axis=0)
    ini_mean_cls4 = np.mean(dataset_sess1[1][lab_id], axis=0)

    #############################
    plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='darkmagenta', alpha=0.2)
    plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='orange', alpha=0.2)
    plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    plt.show()

    plt.figure(10)
    plt.xticks([])
    plt.yticks([])

    plt.scatter(new_dataset[0][:, 0], new_dataset[0][:, 1], c='red', alpha=0.5)
    plt.scatter(new_dataset[1][:, 0], new_dataset[1][:, 1], c='steelblue', alpha=0.5)
    plt.scatter(dataset[0][lab_id, 0], dataset[0][lab_id, 1], c='red')
    plt.scatter(dataset[1][lab_id, 0], dataset[1][lab_id, 1], c='blue')
    plt.scatter(mean_cls1[0], mean_cls1[1], s=150, marker='x', c='red')
    plt.scatter(mean_cls2[0], mean_cls2[1], s=150, marker='x', c='blue')
    ## plot the first principal direction on the original data
    TP1 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 0], L1[0, 0]]) + mean_cls1[0]
    TP2 = 2 * S1[0] / (n ** 0.5) * np.array([-L1[0, 1], L1[0, 1]]) + mean_cls1[1]
    plt.plot(TP1, TP2, 'r-')
    TP1 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 0], L2[0, 0]]) + mean_cls2[0]
    TP2 = 2 * S2[0] / (n ** 0.5) * np.array([-L2[0, 1], L2[0, 1]]) + mean_cls2[1]
    plt.plot(TP1, TP2, 'g-')

    plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='grey', alpha=0.2)
    plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='grey', alpha=0.2)
    # plt.show()

    plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')
    # plt.show()

    # Draw mean point from labeled obs
    ini_mean_cls3 = np.mean(dataset_sess1[0][lab_id], axis=0)
    ini_mean_cls4 = np.mean(dataset_sess1[1][lab_id], axis=0)

    #############################
    plt.scatter(dataset_sess1[0][:, 0], dataset_sess1[0][:, 1], c='darkmagenta', alpha=0.2)
    plt.scatter(dataset_sess1[1][:, 0], dataset_sess1[1][:, 1], c='orange', alpha=0.2)
    plt.scatter(dataset_sess1[0][lab_id, 0], dataset_sess1[0][lab_id, 1], c='magenta')
    plt.scatter(dataset_sess1[1][lab_id, 0], dataset_sess1[1][lab_id, 1], c='darkorange')

    # Draw mean point from labeled obs
    mean_cls3 = np.mean(dataset_sess1[0], axis=0)
    mean_cls4 = np.mean(dataset_sess1[1], axis=0)
    plt.scatter(mean_cls3[0], mean_cls3[1], s=150, marker='x', c='magenta')
    plt.scatter(mean_cls4[0], mean_cls4[1], s=150, marker='x', c='darkorange')

    # Draw normalized graph
    cov_cls3 = np.cov(dataset_sess1[0].T)
    cov_cls4 = np.cov(dataset_sess1[1].T)
    u3, s3, v3 = np.linalg.svd(cov_cls3)
    u4, s4, v4 = np.linalg.svd(cov_cls4)
    L3 = v3[0:1, :]
    L4 = v4[0:1, :]
    S3 = np.sqrt(s3[0:1] * n)
    S4 = np.sqrt(s4[0:1] * n)

    ## plot the first principal direction on the original data
    TP3 = 2 * S3[0] / (n ** 0.5) * np.array([-L3[0, 0], L3[0, 0]]) + mean_cls3[0]
    TP4 = 2 * S3[0] / (n ** 0.5) * np.array([-L3[0, 1], L3[0, 1]]) + mean_cls3[1]
    plt.plot(TP3, TP4, '-', color='magenta')
    TP3 = 2 * S4[0] / (n ** 0.5) * np.array([-L4[0, 0], L4[0, 0]]) + mean_cls4[0]
    TP4 = 2 * S4[0] / (n ** 0.5) * np.array([-L4[0, 1], L4[0, 1]]) + mean_cls4[1]
    plt.plot(TP3, TP4, '-', color='darkorange')
    plt.show()

    ####################################################
    # plt.figure(2)
    # # plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=np.repeat(0, len(dataset[0])))
    # plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c='red')
    # # plt.scatter(dataset[1][:, 0], dataset[1][:, 1], c=np.repeat(20, len(dataset[1])))
    # plt.scatter(dataset[1][:, 0], dataset[1][:, 1], c='steelblue')
    # plt.show()
