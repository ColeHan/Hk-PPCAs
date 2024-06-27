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
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import matplotlib.pyplot as plt
import math
from collections import OrderedDict

from torch import device
from torch.utils.data import DataLoader, Dataset

from sklearn import metrics

import Modules.Modules as modules


def HkPPCAs(num_labeled_per_class, q_ini, q_train, lda, num_iters, dataset_name, self_learner_name, feature_address,
            checkpoint_folder_address, run_id, sessions_list, seeds_list):
    num_sessions = len(sessions_list)  # 1
    for session_id in range(num_sessions):
        seed = seeds_list[session_id]
        modules.setup_seed(seed)
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

        #### Task: Data Preparation

        ######################################################################
        #################### Task: Features integration
        ######################################################################
        # print('Starting feature integration from all seperate files')
        # if 'cifar100' in dataset_name.lower():
        #     train_set, val_set = modules.CIFAR100_Feature(feature_address, cls_range=cls_range)  # for resnet x4
        # if 'imagenet' in dataset_name.lower():
        #     train_set, val_set, folder_label_map = modules.IMAGENET_Feature(feature_address,
        #                                                             cls_range=cls_range)  # for resnet x4
        #
        # ## Randomize training set
        # train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
        # # for it, data in enumerate(train_loader):
        # #     train_values, train_labels, train_ori_id = data[0], list(data[1].numpy()), data[2]
        # data = next(iter(train_loader))
        # train_values, train_labels, train_ori_id = data[0], list(data[1].numpy()), data[2]
        #
        # val_values, val_labels = val_set.values, val_set.labels  # for IMAGE_Feature()
        #
        # ## save .mat data
        # tr_name = 'HkPPCAs_{}_run{}_{}-{}cls_trainset_{}.pth'.format(
        #     dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        # val_name = 'HkPPCAs_{}_run{}_{}-{}cls_valset_{}.pth'.format(
        #     dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        # addr_tr = checkpoint_folder_address + tr_name
        # addr_val = checkpoint_folder_address + val_name
        # # savemat(addr_tr,
        # #         {'feature': train_values.cpu().float().numpy(),
        # #          'label': train_labels,
        # #          'original_id': train_ori_id.numpy()})
        # # savemat(addr_val,
        # #         {'feature': val_values.cpu().float().numpy(),
        # #          'val_labels': val_labels})
        # torch.save({'feature': train_values.cpu().to(torch.float),
        #             'label':train_labels,
        #             'original_id':train_ori_id}, addr_tr)
        # torch.save({'feature': val_values.cpu().to(torch.float),
        #             'label':val_labels}, addr_val)
        ## Task//
        ######################################################################

        ## for large dimension data as .pth
        # addr_tr = checkpoint_folder_address+tr_name
        # addr_val = checkpoint_folder_address+val_name
        # torch.save({'feature': train_values.cpu().float(),
        #             'label':train_labels,
        #             'original_id':train_ori_id}, addr_tr)
        # torch.save({'feature': val_values.cpu().float(),
        #             'val_labels':val_labels}, addr_val)
        # # or save as .mat
        # addr1 = 'D:/FSU/Academic/Research/CSNN-Project/Data/train_CLIP_resnet.mat'
        # addr2 = 'D:/FSU/Academic/Research/CSNN-Project/Data/val_CLIP_resnet.mat'
        # import hdf5storage
        # hdf5storage.savemat(addr1,{'feature': train_values.cpu().float().numpy(),'label':train_labels},
        #                     matlab_compatible=True, compress=False)
        # savemat(addr2, {'feature': val_values.cpu().float().numpy(), 'val_labels': val_labels})
        ######################################################################



        ######################################################################
        #### Task: integrate too many classes into several integrated features
        ######################################################################
        # alltr_feature_address = [os.getcwd() + '\\Benchmarks\\ImageNet10k_alltrain_640_resnetx4\\']
        # # allval_feature_address = [os.getcwd() + '\\..\\..\\TestExamples\\ImageNet10k_allval_640_resnetx4\\']
        # ##train set is too large for memory, divide it into two parts.
        # trainset_first500 = modules.combine_ImageNet10k_by_part(alltr_feature_address, n_features=640,first=True) # for resnet x4
        # train_values=trainset_first500.values
        # train_labels=trainset_first500.labels # list
        # # addr_aug = 'D:/FSU/Academic/Research/CSNN-Project/TestExamples/' \
        # #            'Test_DataAug_AugmentedImageNet_80ratio_x10perimg_trainset_CLIP_resnetx4_first500.mat'
        # # savemat(addr_aug,{'feature': train_aug_values.cpu().float().numpy(), 'label': train_aug_labels})
        # # import hdf5storage
        # # hdf5storage.savemat(addr_aug, {'feature': train_aug_values.cpu().to(torch.float).numpy(), 'label': train_aug_labels},
        # #                     matlab_compatible=True, compress=False)
        # ## for dimension data as .pth
        # addr_half = 'D:/FSU/Academic/Research/Hierarchical K-PPCAs/TempExample/' \
        #            'trainset_ImageNet10k_CLIP_resnetx4_first.pth'
        # torch.save({'feature': train_values.cpu().to(torch.float),
        #             'label':train_labels}, addr_half)
        #
        #
        #
        # trainset_second500 = modules.combine_ImageNet10k_by_part(alltr_feature_address, n_features=640, first=False)  # for resnet x4
        # train_values = trainset_second500.values
        # train_labels = trainset_second500.labels  # list
        #
        # # for dimension data as .pth
        # addr_half = 'D:/FSU/Academic/Research/Hierarchical K-PPCAs/TempExample/' \
        #            'trainset_ImageNet10k_CLIP_resnetx4_second.pth'
        # torch.save({'feature': train_values.cpu().to(torch.float),
        #             'label':train_labels}, addr_half)



        ######################################################################



        ######################################################################
        ########### Task: Load prestored integrated features
        ######################################################################
        # for small dimension .mat
        # tr_name = 'HkPPCAs_{}_run{}_{}-{}cls_trainset_{}.pth'.format(
        #     dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        # val_name = 'HkPPCAs_{}_run{}_{}-{}cls_valset_{}.pth'.format(
        #     dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        # addr_tr = checkpoint_folder_address + tr_name
        # addr_val = checkpoint_folder_address + val_name
        #
        # # train_data = loadmat(addr_tr)
        # # val_data = loadmat(addr_val)
        # train_data = torch.load(addr_tr)
        # val_data = torch.load(addr_val)
        #

        # # load all ImageNet10k features
        # # Train
        # addr_1half = 'D:/FSU/Academic/Research/Hierarchical K-PPCAs/TempExample/' \
        #        'Test_DataAug_AugmentedImageNet_80ratio_x10perimg_trainset_CLIP_resnetx4_first500.pth'
        #
        # addr_2half = 'D:/FSU/Academic/Research/Hierarchical K-PPCAs/TempExample/' \
        #            'Test_DataAug_AugmentedImageNet_80ratio_x10perimg_trainset_CLIP_resnetx4_second500.pth'
        #
        # # train_aug_data_half = loadmat(addr_aug)
        # # train_aug_data_half = mat73.loadmat(addr_aug)
        # train_data_1half = torch.load(addr_1half)
        # train_data_2half = torch.load(addr_2half)
        #
        # x_all=torch.cat((train_data_1half['feature'],train_data_2half['feature']),dim=0)
        # y_all = train_data_1half['label']
        # y_all.extend(train_data_2half['label'])
        # train_data=train_data_1half
        # del train_data_1half,train_data_2half
        #
        # train_set = MixtureDataset(x_all, y_all)
        #
        # ## Randomize training set
        # train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
        # # for it, data in enumerate(train_loader):
        # #     train_values, train_labels, train_ori_id = data[0], list(data[1].numpy()), data[2]
        # data = next(iter(train_loader))
        # train_values, train_labels, train_ori_id = data[0], list(data[1].numpy()), data[2]
        #
        # val_values, val_labels = val_set.values, val_set.labels  # for IMAGE_Feature()
        #
        # ## save .mat data
        # tr_name = 'HkPPCAs_{}_run{}_{}-{}cls_trainset_{}.pth'.format(
        #     dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        # val_name = 'HkPPCAs_{}_run{}_{}-{}cls_valset_{}.pth'.format(
        #     dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        # addr_tr = checkpoint_folder_address + tr_name
        # addr_val = checkpoint_folder_address + val_name
        # savemat(addr_tr,
        #         {'feature': train_values.cpu().float().numpy(),
        #          'label': train_labels,
        #          'original_id': train_ori_id.numpy()})
        # savemat(addr_val,
        #         {'feature': val_values.cpu().float().numpy(),
        #          'val_labels': val_labels})
        #
        # # val
        # val_name = 'HkPPCAs_{}_run{}_{}-{}cls_valset_{}.pth'.format(
        #     dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        # addr_val = checkpoint_folder_address + val_name
        #
        # val_data = loadmat(addr_val)


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
        ## Task//
        ######################################################################

        ######################################################################
        ##################   Task: Feature Normalization
        ######################################################################
        # norm_name = 'HkPPCAs_{}_run{}_Normalization_{}.pth'.format(
        #     dataset_name, run_id, self_learner_name)
        # norm_base_session_address = checkpoint_folder_address + norm_name
        # if session_id == 0:
        #     ## normalization for train and val data
        #     m_train_values = np.mean(train_data['feature'].numpy(), axis=0)
        #     std_train_values = np.std(train_data['feature'].numpy(), axis=0)
        #     d = m_train_values.shape[0]
        #     train_data['feature'] = (train_data['feature'].numpy() - m_train_values) / std_train_values / np.sqrt(d)
        #     m_val_values = np.mean(val_data['feature'].numpy(), axis=0)
        #     std_val_values = np.std(val_data['feature'].numpy(), axis=0)
        #     val_data['feature'] = (val_data['feature'].numpy() - m_val_values) / std_val_values / np.sqrt(d)
        #
        #     # train_ori_id = train_data['original_id'] # for data augmentation
        #
        #     torch.save({'m_train_values': m_train_values,
        #                 'std_train_values': std_train_values,
        #                 'm_val_values': m_val_values,
        #                 'std_val_values': std_val_values
        #                 },
        #                norm_base_session_address)
        # else:
        #     norm_basesession_result = torch.load(norm_base_session_address)
        #     m_val_basesession = norm_basesession_result['m_val_values']
        #     m_train_basesession = norm_basesession_result['m_train_values']
        #     std_val_basesession = norm_basesession_result['std_val_values']
        #     std_train_basesession = norm_basesession_result['std_train_values']
        #     d = m_train_basesession.shape[0]
        #     train_data['feature'] = (train_data['feature'].numpy() - m_train_basesession) / std_train_basesession / np.sqrt(d)
        #     val_data['feature'] = (val_data['feature'].numpy() - m_val_basesession) / std_val_basesession / np.sqrt(d)
        # ##//Task
        ######################################################################

        tr_name = 'HkPPCAs_{}_run{}_{}-{}cls_trainset_{}_afterNorm.pth'.format(
            dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        val_name = 'HkPPCAs_{}_run{}_{}-{}cls_valset_{}_afterNorm.pth'.format(
            dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        addr_tr = checkpoint_folder_address + tr_name
        addr_val = checkpoint_folder_address + val_name

        # torch.save({'feature': torch.tensor(train_data['feature'], dtype=torch.float),
        #             'label':train_data['label'],
        #             'original_id':train_data['original_id']}, addr_tr)
        # torch.save({'feature': torch.tensor(val_data['feature'], dtype=torch.float),
        #             'label':val_data['label'],}, addr_val)

        train_data = torch.load(addr_tr)
        val_data = torch.load(addr_val)



        trainset = modules.MixtureDataset(train_data['feature'],
                                          list(train_data['label']))
        val_values, val_labels = val_data['feature'], list(val_data['label'])
        N = len(trainset)

        d=trainset.values.shape[1]

        ######################################################################
        ################# Task: Picking obs as labeled obs
        ######################################################################
        # ## label by percentage
        # perc_labeled = 0.01
        # num_labeled = round(N * perc_labeled)
        # num_unlabeled = N - num_labeled

        ## label by a fixed number per class
        num_labeled = num_new_cls * num_labeled_per_class
        # num_unlabeled = N - num_labeled

        ### Task: obtain indices for labeled data
        train_values, train_labels = trainset.values, trainset.labels
        num_batches = 120#150 #300  # 80
        batch_size = math.ceil(len(trainset) / num_batches)
        ######################################################################
        # ## obtain indices for labeled data - label by percentage
        # num_labeled_batch = math.ceil(num_labeled / num_batches)
        # labeled_indices = []
        # for i in range(num_batches):
        #     id_start = i * size_batch
        #     id_end = i * size_batch + num_labeled_batch
        #     labeled_indices_batch_global = [i for i in range(id_start, id_end)]
        #     labeled_indices.extend(labeled_indices_batch_global)
        #     # print(len(labeled_indices_batch),id_start,id_end)
        ######################################################################

        ######################################################################
        ## obtain indices for labeled data - label by fixed number per class
        labeled_indices = []  # id of all labeled observations in trainset
        selectedobs_dict = {}  # dict(labels:id_labeled_obs)
        for label_id, y in enumerate(train_labels):
            if y not in selectedobs_dict.keys():
                selectedobs_dict[y] = [label_id]
                labeled_indices.append(label_id)
            else:
                if len(selectedobs_dict[y]) < num_labeled_per_class:
                    selectedobs_dict[y].append(label_id)
                    labeled_indices.append(label_id)
                else:
                    continue
            if len(labeled_indices) == num_labeled:
                break
        #\\Task
        ######################################################################


        ## - Check and summarize selected labeled results
        assert len(labeled_indices) == num_labeled
        # labeled_indices = labeled_indices[:num_labeled]
        print('Require {} labeled per class. {} classes and {} labeled samples in current session\n'
              .format(num_labeled_per_class, num_new_cls, len(labeled_indices)))






        ######################################################################
        #################### Task: Initialization
        ######################################################################
        print('Starting initialization')
        # all classes can be assigned for current session: num_cls
        num_cls: int = len(set(train_labels)) + num_prestored_cls  # num_prestored_cls=0
        num_supcls =  1
        # # # num_supcls =  200
        # # # num_supcls = 50 # 200
        # # # iter_supcls = 5
        
        ## - Select labeled data for initialization
        xl = train_values[labeled_indices, :]  # dim: (num_labeled, d)
        yl = [train_labels[i] for i in labeled_indices]
        xl = xl.to(device)
        print('Dimension of labeled data in current session:', xl.shape,
              '\nLength of list of labels:', len(yl))



        # ### Task: initialize PPCA parameters (mu,L,D2) for classes
        # pi_ini = torch.tensor([1 / num_cls], device=device).repeat(num_cls).unsqueeze(-1)  # dim: (num_cls, 1)
        # mu_ini = torch.zeros((num_cls, d), device=device)
        # sorted_unique_labels_cur_trainset = sorted(list(set(train_labels)))
        # for _, label in enumerate(sorted_unique_labels_cur_trainset):
        #     mu_ini[label] = torch.mean(xl[torch.tensor(yl) == label, :], dim=0)
        #
        # cov_ini = torch.zeros((num_cls, d, d))
        # L_ini = torch.zeros((num_cls, q_ini, d), device=device)
        # # S_ini = torch.zeros((num_cls, q_ini), device=device)  # svd on x-mu version
        # D2_ini = torch.zeros((num_cls, q_ini), device=device)  # svd on cov version
        # nc_ini = torch.zeros((num_cls, 1), device=device)  # number of observations in each class
        # newcls_labeled_obsid_dict = dict.fromkeys([newcls_id for newcls_id in range(num_prestored_cls, num_cls)])
        # for k in range(num_prestored_cls, num_cls):  # num_prestored_cls=0
        #     id_labeled_k = torch.tensor(yl) == k
        #     newcls_labeled_obsid_dict[k] = id_labeled_k
        #     nc_ini[k] = torch.sum(id_labeled_k)
        #     cov_ini[k] = torch.from_numpy(np.cov(xl[id_labeled_k, :].T.cpu()))
        #     _, s, v = torch.linalg.svd(xl[id_labeled_k, :] - mu_ini[k, :])
        #     L_ini[k, :, :] = v[0:q_ini, :]
        #     # s_fixedlenth = torch.zeros(q_ini)
        #     # minl = min(q_ini, len(s))
        #     # s_fixedlenth[:minl] = s[:minl]
        #     # S_ini[i, :] = s_fixedlenth  # svd on x-mu version
        #     D2_fixedlenth_k = torch.zeros(q_ini)
        #     minl = min(q_ini, len(s))
        #     S2 = torch.square(s) / (nc_ini[k] - 1)
        #     D2_fixedlenth_k[:minl] = S2[:minl]
        #     D2_ini[k, :] = D2_fixedlenth_k  # svd on cov version
        #     if k % 200 == 0:
        #         print(k)
        #
        # # ## - Recall stored classes from previous sessions and save to mu_ini, L_ini, S_ini, and nc
        # # ## TODO: not store cov
        # # if session_id > 0:
        # #     checkpoint_old_session_name = 'HkPPCAs_{}_run{}_first{}cls_{}_PCAq{}_{}shot_q{}lda{:.0e}_iter{}.pth'.format(
        # #         dataset_name, run_id, cls_range[0], self_learner_name, q_ini,
        # #         num_labeled_per_class, q_train, lda, num_iters)
        # #     prestored = checkpoint_folder_address + checkpoint_old_session_name
        # #     # {'mu_online': mu_online,
        # #     #  'L_online': L_online,
        # #     #  'Sx_online': S_online,
        # #     #  'D2_online': D2_online,
        # #     #  'lda': lda, 't': t, 'N': N,
        # #     #  'nc': nc, 'num_cls': num_cls,
        # #     #  'trainset': trainset,
        # #     #  'val_values': val_values,
        # #     #  'val_labels': val_labels
        # #     #  }
        # #     prestored_result = torch.load(prestored)
        # #
        # #     nc_prestored = prestored_result['nc']
        # #     mu_prestored = prestored_result['mu_online']
        # #     L_prestored = prestored_result['L_online']
        # #     # S_prestored = prestored_result['Sx_online']
        # #     D2_prestored = prestored_result['D2_online']
        # #
        # #     nc_ini[:num_prestored_cls] = nc_prestored
        # #     mu_ini[:num_prestored_cls] = mu_prestored
        # #     L_ini[:num_prestored_cls] = L_prestored
        # #     # S_ini[:num_prestored_cls] = S_prestored
        # #     D2_ini[:num_prestored_cls] = D2_prestored
        #
        #


        # # ### Task: initialize super-classes by k-Means
        # # # super_classes = modules.SuperClasses(mu_ini, nc_ini, num_supcls, q_supcls=q_ini, device=device,
        # # #                                      PPCA=False,
        # # #                                      cov_cls=cov_ini)
        # # super_classes = modules.SuperClasses(mu_ini, nc_ini, num_supcls, q_supcls=q_ini, device=device,
        # #                                      PPCA=True,
        # #                                      L_cls=L_ini,
        # #                                      D2_cls=D2_ini)
        # # super_classes.initialize()
        # # # super_classes.update_supcls_kMeans_gaussian(iter=iter_supcls)
        # # rawdata=[xl, torch.tensor(yl)]
        # # # super_classes.update_supcls_kMeans_raw(rawdata, iter=iter_supcls)
        # # super_classes.update_supcls_KLdiv_raw(rawdata, iter=iter_supcls)  # raw data
        # # # super_classes.update_supcls_KLdiv_emp(xl, newcls_labeled_obsid_dict, iter=iter_supcls)
        # # # super_classes.update_supcls_KLdiv_exp(iter=iter_supcls)
        #
        # ## - Save initialization results
        # # addr_ini = checkpoint_folder_address + \
        # #            'HkPPCAs_{}_run{}_{}supclss_{}-{}cls_{}_PCAq_ini{}_{}shot_lda{:.0e}_ini_KMsupPE.pth'.format(
        # #                dataset_name, run_id, num_supcls, cls_range[0] + 1, cls_range[1], self_learner_name,
        # #                q_ini, num_labeled_per_class, lda)
        # addr_ini = checkpoint_folder_address + \
        #            'HkPPCAs_{}_run{}_{}supclss_{}-{}cls_{}_PCAq_ini{}_{}shot_lda{:.0e}_ini_flat.pth'.format(
        #                dataset_name, run_id, num_supcls, cls_range[0] + 1, cls_range[1], self_learner_name,
        #                q_ini, num_labeled_per_class, lda)
        # torch.save({'pi_ini': pi_ini,
        #             'mu_ini': mu_ini,
        #             'L_ini': L_ini,
        #             # 'S_ini': S_ini,
        #             'D2_ini': D2_ini,
        #             'nc_ini': nc_ini,
        #             'labeled_indices': labeled_indices,
        #             'trainset': trainset,
        #             # 'super_classes': super_classes
        #             },
        #            addr_ini)
        #
        # print('initialization done\n')
        # ## Task: Initialization End//



        # addr_ini = checkpoint_folder_address + \
        #            'HkPPCAs_{}_run{}_{}supclss_{}-{}cls_{}_PCAq_ini{}_{}shot_lda{:.0e}_ini.pth'.format(
        #                dataset_name, run_id, num_supcls, cls_range[0] + 1, cls_range[1], self_learner_name,
        #                q_ini, num_labeled_per_class, lda)
        addr_ini = checkpoint_folder_address + \
                   'HkPPCAs_{}_run{}_{}supclss_{}-{}cls_{}_PCAq_ini{}_{}shot_lda{:.0e}_ini_flat.pth'.format(
                       dataset_name, run_id, num_supcls, cls_range[0] + 1, cls_range[1], self_learner_name,
                       q_ini, num_labeled_per_class, lda)

        ini_result = torch.load(addr_ini)
        # pi_ini = ini_result['pi_ini']
        mu_ini = ini_result['mu_ini']
        L_ini = ini_result['L_ini']
        # S_ini = ini_result['S_ini']
        D2_ini = ini_result['D2_ini']
        nc_ini = ini_result['nc_ini']
        # super_classes = ini_result['super_classes']

        torch.cuda.empty_cache()

        # ### Task: Preload from checkpoints
        # q=2
        # iter_id=0
        # print('Loading from iter{} ckpt...'.format(iter_id+1))
        # checkpoint_iter_name = 'HkPPCAs_{}_run{}_first{}cls_{}_PCAq{}_{}shot_q{}lda{:.0e}_iter{}.pth'.format(
        #     dataset_name, run_id, cls_range[1], self_learner_name, q_ini, num_labeled_per_class, q, lda,
        #     iter_id + 1)
        # # {'mu_online': mu_online,
        # #  'L_online': L_online,
        # #  # 'Sx_online': S_online,
        # #  'D2_online': D2_online,
        # #  'lda': lda,
        # #  't': t,
        # #  'q': q,
        # #  'N': N,
        # #  'nc': nc,
        # #  'num_cls': num_cls,
        # #  'trainset': trainset,
        # #  'val_values': val_values,
        # #  'val_labels': val_labels,
        # #  'sup_cls': super_classes,
        # #  'num_supcls': num_supcls,
        # #  'num_candid_supcls': num_candid_supcls,
        # # 'score_criteria': score_criteria
        # #  }
        # ini_result=torch.load(checkpoint_folder_address + checkpoint_iter_name)
        #
        # mu_ini = ini_result['mu_online']
        # L_ini = ini_result['L_online']
        # # S_ini = ini_result['S_ini']
        # D2_ini = ini_result['D2_online']
        # nc_ini = ini_result['nc']
        #
        # super_classes = ini_result['sup_cls']
        #
        # torch.cuda.empty_cache()
        # print('ckpt preloading completed')


        ### Task: Generate local indices of labeled data in each batch
        labeled_local_indices_per_batch = [[] for _ in range(num_batches)]
        for obs1_global_id in labeled_indices:
            obs1_local_id = obs1_global_id % batch_size
            batch_id = math.floor(obs1_global_id / batch_size)
            labeled_local_indices_per_batch[batch_id].append(obs1_local_id)

        ### Task: Prepare dataloader and PPCA parameters for training
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=False)

        # pi_online = pi_ini.clone()
        mu_online = mu_ini.clone().detach()
        L_online = L_ini.clone().detach()
        # S_online = S_ini.clone()
        D2_online = D2_ini.clone().detach()
        nc = nc_ini.clone().detach()
        # del pi_ini

        del mu_ini, L_ini, D2_ini, ini_result
        torch.cuda.empty_cache()

        ## Task: Initial Accuracy for test set
        num_candid_supcls = 4


        ## - Evaluation: Accuracy for val
        print('Evaluating test set from {}th class to {}th class'.format(min(val_labels), max(val_labels)))
        # y_pred_val, loss = modules.prediction_flat_D2_HkPPCAs(val_values, val_labels, num_cls,
        #                                                   mu_online, L_online, D2_online,
        #                                                   lda, t, device,
        #                                                   num_batches=100, max_cls_batch_size=1300)
        y_pred_val, loss = modules.prediction_flat_D2_HkPPCAs(val_values, val_labels, num_cls,
                                                          mu_online, L_online, D2_online,
                                                          lda, t, device,
                                                          num_batches=100, max_cls_batch_size=1300,dist='Eu') # Tag: k-means
        # y_pred_val, loss = modules.prediction_D2_HkPPCAs(val_values, val_labels, num_cls,
        #                                                  mu_online, L_online, D2_online,
        #                                                  super_classes, num_candid_supcls,
        #                                                  lda, t, device,
        #                                                  num_batches=100, max_cls_batch_size=1300)
        y_true_val = torch.tensor(val_labels, dtype=y_pred_val.dtype).to(device)
        cls_err_val = torch.sum(y_pred_val != y_true_val) / (len(val_labels))
        acc_val = 1 - cls_err_val.item()
        print('#{} Accuracy rate - val: {:.4f}'.format('init', acc_val))
        print('#{} loss - on: {}'.format('init', loss.item()))
        # Task//

        ### Task: Start training
        q = q_train  # first q PCs in MPPCA
        # lda = 1e-2
        # plt_axis1 = []
        # plt_acc_val = []
        # plt_loss = []
        torch.cuda.empty_cache()
        print('there are {} classes'.format(num_cls))
        # num_iters = 10
        score_criteria = [torch.tensor(float('Inf')) for k in range(num_cls)]
        for iter_id in range(num_iters):
            # t1 = time.time()
            raves = [modules.RAVE() for k in range(num_cls)]
            num_candid_cls_all_obs=[]
            speedup_all_obs=[]

            # not_full_raw_data_RAVEs_id=[k for k in range(num_prestored_cls, num_cls)]

            for batch_id, data in enumerate(train_loader):
                time0=time.time()
                if batch_id % 10 == 0:
                    print('Current #batch: {}/{}'.format(batch_id+1, num_batches))
                # t10 = time.time()
                train_values, train_labels = data[0], list(data[1].numpy())
                x_batch = train_values.to(device)
                assigned_y_batch=-torch.ones(data[1].shape)
                n_one_batch = x_batch.shape[0]

                assigned_cls_map_obs_id= {ci:[] for ci in range(num_cls)}
                scores_batch=-torch.ones(n_one_batch)

                labeled_local_indices_1batch = labeled_local_indices_per_batch[batch_id]
                num_labeled_batch = len(labeled_local_indices_1batch)

                ### Task: KMeans for PPCAs - labeled and unlabeled data mixed

                time00=time.time()
                for obs1_id in range(n_one_batch):
                    time00_obs1=time.time()
                    ## Task: if labeled, assign to its label
                    if obs1_id in labeled_local_indices_1batch:
                        labeled_cls = train_labels[obs1_id]
                        # ## Task: update RAVE by 1 obs
                        # raves[labeled_cls].add_onlyX(x_batch[obs1_id, :].unsqueeze(0).clone(), mxx_cpu=True)
                        ## Task: update RAVE by 1 batch of observations
                        assigned_cls_map_obs_id[labeled_cls].append(obs1_id)
                        assigned_y_batch[obs1_id]=labeled_cls
                        continue


                    ## Task: if unlabeled, select closest from candidate classes (formed by candidate sup-classes)
                    # time00_unlab=time.time()
                    # candid_cls_id = []
                    # # for candid_1supcls_id in candid_supcls_id_X[:, obs1_id].tolist():
                    # #     candid_cls_id.extend(super_classes.supcls_cls_dict[candid_1supcls_id])
                    candid_cls_id = [i for i in range(num_cls)]
                     # time01_unlab_candidclss=time.time()

                    ## Task: Speed-ups
                    num_candid_cls = len(candid_cls_id)
                    speedup_1obs=modules.speed_up_1obs(num_cls, num_supcls, num_candid_cls)
                    num_candid_cls_all_obs.append(num_candid_cls)
                    speedup_all_obs.append(speedup_1obs)
                    ##Task/


                    max_cls_batch_size=2000
                    torch.cuda.synchronize()
                    time01_unlab_assign=time.time()

                    mu_candid_cls = mu_online[candid_cls_id]  # (num_candid_cls, d)
                    L_candid_cls = L_online[candid_cls_id]  # .to(device)  # (num_candid_cls, k, d)
                    D2_candid_cls = D2_online[candid_cls_id]  # .to(device)  # (num_candid_cls, k, k)

                    if num_candid_cls <= max_cls_batch_size:
                        torch.cuda.synchronize()
                        # t00 = time.time()
                        t1 = time.time()
                        # score_obs1, t3 = modules.score_1obs_L_D2(x_batch[obs1_id], mu_candid_cls,
                        #                                          L_candid_cls, D2_candid_cls, lda, t)
                        score_obs1, t3 = modules.score_1obs_L_D2(x_batch[obs1_id], mu_candid_cls,
                                                                 L_candid_cls, D2_candid_cls, lda, t,dist='Eu') # Tag: k-Means
                        torch.cuda.synchronize()
                        t2 = time.time()
                        # print(delta_candid_cls.shape, 'score calculation', t2 - t1, t2-t3)
                        # print('score calculation', t2 - t1, t2-t3)
                        # del mu_candid_cls
                        # del L_candid_cls, D2_candid_cls
                        # del delta_candid_cls
                    else:
                        score_obs1 = torch.zeros(num_candid_cls,device=device)
                        ith_part = math.ceil(num_candid_cls / max_cls_batch_size)
                        for i_part in range(ith_part):
                            torch.cuda.synchronize()
                            t00=time.time()
                            start_id = i_part * max_cls_batch_size
                            end_id = min((i_part + 1) * max_cls_batch_size, num_candid_cls)
                            # candid_cls_id_part = candid_cls_id[start_id:end_id]
                            torch.cuda.synchronize()
                            t01=time.time()

                            mu_candid_cls_part = mu_candid_cls[start_id:end_id]  # (num_candid_cls, d)
                            L_candid_cls_part = L_candid_cls[start_id:end_id]  # .to(device)  # (num_candid_cls, k, d)
                            D2_candid_cls_part = D2_candid_cls[start_id:end_id]  # .to(device)  # (num_candid_cls, k, k)

                            torch.cuda.synchronize()
                            t1 = time.time()
                            # score_obs1[start_id:end_id], t3 = modules.score_1obs_L_D2(x_batch[obs1_id],
                            #                                                           mu_candid_cls_part,
                            #                                                           L_candid_cls_part,
                            #                                                           D2_candid_cls_part, lda, t)
                            score_obs1[start_id:end_id], t3 = modules.score_1obs_L_D2(x_batch[obs1_id],
                                                                                      mu_candid_cls_part,
                                                                                      L_candid_cls_part,
                                                                                      D2_candid_cls_part, lda, t, dist='Eu') # Tag: k-Means
                            torch.cuda.synchronize()
                            t2 = time.time()
                            # print(num_candid_cls,i_5k,t2-t00, t01 - t00, t1-t01, t2-t1)
                            # print(delta_candid_cls_part.shape, 'score calculation', t2 - t1,t2-t3)
                            # print('score calculation', t2 - t1, t2-t3)
                        # del mu_candid_cls_part
                        # del L_candid_cls_part, D2_candid_cls_part
                        # del delta_candid_cls_part

                    torch.cuda.synchronize()
                    time01_score = time.time()
                    # print(num_candid_cls, 'score calculation',time01_score-time01_unlab_assign, t1-t00, t2 - t1)

                    # # 2 original way
                    # score_obs1_minid = torch.argmin(score_obs1,0)
                    # score_obs1_mincls = candid_cls_id[score_obs1_minid]

                    # # 1
                    # score_obs1_minid = torch.argmin(score_obs1)
                    # score_obs1_mincls = candid_cls_id[score_obs1_minid.item()]
                    # # 2
                    # score_obs1_minid = torch.argmin(score_obs1)
                    # score_obs1_mincls = candid_cls_id[score_obs1_minid]
                    # # 3
                    # score_obs1_minid = np.argmin(score_obs1.cpu().numpy())
                    # score_obs1_mincls = candid_cls_id[score_obs1_minid.item()]
                    # 4
                    score_obs1_minid = np.argmin(score_obs1.cpu().numpy(),0)
                    score_obs1_mincls = candid_cls_id[score_obs1_minid]
                    # tb=time.time()
                    # # 5
                    # score_obs1_minid = np.argmin(score_obs1.tolist())
                    # score_obs1_mincls = candid_cls_id[score_obs1_minid]

                    torch.cuda.synchronize()
                    time01_min=time.time()


                    # time00_argmin+=time01_min-time01_score

                    # ## Task: add 1 obs to its assigned class
                    # if score_obs1[score_obs1_minid] < score_criteria[score_obs1_mincls]:  # if not, out of distribution, throw away
                    #     raves[score_obs1_mincls].add_onlyX(x_batch[obs1_id, :].unsqueeze(0).clone(), mxx_cpu=True)
                    #     raves[score_obs1_mincls].add_score(score_obs1[score_obs1_minid])
                    ## Task: add 1 batch of observations to its assigned class
                    # time00_RAVE=time.time()
                    if score_obs1[score_obs1_minid] < score_criteria[score_obs1_mincls]:  # if not, out of distribution, throw away
                        assigned_cls_map_obs_id[score_obs1_mincls].append(obs1_id)
                        scores_batch[obs1_id]=score_obs1[score_obs1_minid]
                        assigned_y_batch[obs1_id]=score_obs1_mincls


                    time01_RAVE=time.time()
                    # print(time01_RAVE-time00_obs1, time01_min-time01_score,tb-time01_score)
                    # print(time01_RAVE-time00_obs1, time01_score-time01_unlab_assign)

                del mu_candid_cls, L_candid_cls, D2_candid_cls
                del mu_candid_cls_part, L_candid_cls_part, D2_candid_cls_part
                del score_obs1, score_obs1_mincls, score_obs1_minid

                ## Task: add 1 batch of observations to their assigned class
                for k in range(num_prestored_cls, num_cls):
                    obs_id_in_class_k=assigned_cls_map_obs_id[k]
                    # add observations to RAVE
                    raves[k].add_onlyX(x_batch[obs_id_in_class_k,:],mxx_cpu=True)
                    # add scores to RAVE (need to remove labeled obs, their scores are -1 in scores_batch)
                    pos_scores_id=scores_batch[obs_id_in_class_k]>0
                    pos_scores=scores_batch[obs_id_in_class_k][pos_scores_id]
                    raves[k].add_scores(pos_scores)
                    # # add raw data for super class updating
                    # if k in not_full_raw_data_RAVEs_id:
                    #     resp=raves[k].add_raw_data(x_batch[obs_id_in_class_k],assigned_y_batch[obs_id_in_class_k]) # sample 20 obs by default
                    #     if resp=='full':
                    #         not_full_raw_data_RAVEs_id.remove(k)



                    # if time01_unlab_assign-time00_unlab>0.1:
                    #     print('For {}th obs\n'
                    #           ' Find candid classes: {}s\n'
                    #           ' Extract mu_cand and delta_candid: {}s\n'
                    #           ' Assign obs: score: {}s\n'
                    #           ' Assign obs: assign: {}s\n'
                    #           ' Update RAVE: {}s\n'
                    #           ' Total: {}s'.format(obs1_id + 1,
                    #                                time01_unlab_candidclss-time00_unlab,
                    #                                time01_unlab_mu_delta-time01_unlab_candidclss,
                    #                                time01_unlab_assign_score-time01_unlab_mu_delta,
                    #                                time01_unlab_assign - time01_unlab_assign_score,
                    #                                time01_RAVE - time01_unlab_assign,
                    #                                time01_RAVE-time00_unlab))
                    #     print('checkpoint')
                    # if (obs1_id+1)%1000==0:
                    #     time01=time.time()
                    #     print('Avg classification time cost per obs: {}s'.format((time01-time00)/1000))
                    #     time00=time.time()

                # time1=time.time()
                # time_1batch=time1-time0
                # print('batch #{} ({} obs) time cost: {}'.format(batch_id,n_one_batch,time_1batch))

                # ###################  Old Version  ####################################
                # # mah_dist_to_sup_cls_X = torch.zeros((num_supcls,n_one_batch), device=device)
                # candid_supcls_id_X = torch.zeros((num_candid_supcls, n_one_batch), dtype=torch.int)
                # for obs1_id in range(n_one_batch):
                #     ## Task: if labeled, assign to its label
                #     if obs1_id in labeled_local_indices_1batch:
                #         labeled_cls = train_labels[obs1_id]
                #         raves[labeled_cls].add_onlyX(x_batch[obs1_id, :].unsqueeze(0).clone(), mxx_cpu=True)
                #         continue
                #
                #     ## Task: find closest 4 super classes for each obs in this batch
                #     mah_dist, eu_dist = super_classes.mahalanobis(x_batch[obs1_id])  # distance dim: num_supcls
                #     # mah_dist_to_sup_cls_X[:,obs1_id]=mah_dist
                #     _, candid_supcls_id = torch.topk(mah_dist, k=num_candid_supcls, largest=False)
                #     candid_supcls_id_X[:, obs1_id] = candid_supcls_id
                #
                #     ## Task: find closest class from 4 candidate super classes
                #     candid_cls_id = []
                #     for candid_1supcls_id in candid_supcls_id.tolist():
                #         candid_cls_id.extend(super_classes.supcls_cls_dict[candid_1supcls_id])
                #     num_candid_cls = len(candid_cls_id)
                #     score_obs1 = torch.zeros(num_candid_cls)
                #     for ki, k in enumerate(candid_cls_id):
                #         delta_candid_cls_k = delta[k].to(device)
                #         try:
                #             score_obs1[ki], _ = modules.score(x_batch[obs1_id], mu_online[k], delta_candid_cls_k, lda, t)
                #         except:
                #             print('ckpt')
                #     score_obs1_minval, score_obs1_minid = torch.min(score_obs1, dim=0)
                #     score_obs1_mincls = candid_cls_id[score_obs1_minid]
                #     ## Task: add obs to its assigned class
                #     raves[score_obs1_mincls].add_onlyX(x_batch[obs1_id, :].unsqueeze(0).clone(), mxx_cpu=True)
                # ###########################################################################

                # del score_k
                # del score_obs1
                del x_batch#, mah_dist, eu_dist
                # del candid_supcls_id_X
                torch.cuda.empty_cache()

                print('1 batches finished:', time.time()-time0)
                # print('1 batches finished:', time.time()-time0, 'argmin cost:',time00_argmin/n_one_batch)
                # break




            ## Task; Speed-up results
            avg_num_candid_cls=round(sum(num_candid_cls_all_obs)/len(num_candid_cls_all_obs))
            avg_speed_up_emp=np.mean(speedup_all_obs).item()
            # speed_up_TxA=modules.speed_up(num_cls, num_supcls, num_candid_supcls, avg_num_candid_cls/num_candid_supcls)
            speed_up_TxA=modules.speed_up_TA(num_cls, num_supcls, avg_num_candid_cls)
            print('Average #candidate classes: {}\n'
                  'Average speed-up (emp): {}\n'
                  'Average speed-up (uniform): {}'.format(avg_num_candid_cls,avg_speed_up_emp,speed_up_TxA))
            # plt.hist(num_candid_cls_all_obs)
            # plt.title('Distribution of number of candidate classes')
            # plt.xlabel('number of candidate classes')
            # plt.ylabel('count')
            # plt.show()
            #
            # plt.hist(speedup_all_obs)
            # plt.title('Distribution of all speed-ups')
            # plt.xlabel('speed-ups')
            # plt.ylabel('count')
            # plt.show()
            ## Task/

            ## Update score criteria by RAVEs
            for k in range(num_prestored_cls, num_cls):
                # new_score_criteria_1cls = raves[k].calc_score_criteria_normal(sig_level=0.975)
                new_score_criteria_1cls = raves[k].calc_score_criteria_chi(sig_level=0.975)
                score_criteria[k] = new_score_criteria_1cls

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
                mu_online[k] = raves[k].mx.to(device).clone().detach()
                try:
                    cov_online[k] = raves[k].cov_weighted().cpu()
                except:
                    print(123)
            mu_online = mu_online.cpu()  # save GPU memory

            # ## Task: prepare raw data for sup-cls PPCA update
            # # rawdata = [xl, torch.tensor(yl)] # always use labeled data as raw data
            # # summarize and use stored raw data in each RAVE (labeled and unlabeled)
            # x_raw=None
            # y_raw=None
            # for k in range(num_prestored_cls, num_cls): # Task: only include raw data from current session
            #     raw_1cls=raves[k].raw_data
            #     x_raw_1cls=raw_1cls[0]
            #     y_raw_1cls=raw_1cls[1]
            #     if x_raw is None:
            #         assert y_raw is None
            #         x_raw=x_raw_1cls
            #         y_raw=y_raw_1cls
            #     else:
            #         x_raw=torch.cat((x_raw,x_raw_1cls),0)
            #         y_raw=torch.cat((y_raw,y_raw_1cls),0)
            # rawdata=[x_raw,y_raw]
            # ## TODO: when dealing with incremental case, need to add raw data from old classes
            # ## /Task

            del raves
            # del mah_dist_to_sup_cls_X, candid_supcls_id_X
            # del mu_candid_cls, L_candid_cls, D2_candid_cls
            torch.cuda.empty_cache()

            ## Perform SVD
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
                if k % 500 == 0:
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

            # # Super-class Parameter Estimate
            # # super_classes = modules.SuperClasses(mu_online, nc, num_supcls, q_supcls=q, device=device,
            # #                                      PPCA=False,
            # #                                      cov_cls=cov_online)
            # super_classes = modules.SuperClasses(mu_online, nc, num_supcls, q_supcls=q, device=device,
            #                                      PPCA=True,
            #                                      L_cls=L_online,
            #                                      D2_cls=D2_online)
            # super_classes.initialize()
            #
            # ## TODO: when dealing with incremental case, need to add raw data from old classes
            # # super_classes.update_supcls_kMeans(iter=iter_supcls)
            # # super_classes.update_supcls_kMeans_raw(rawdata, iter=iter_supcls) # raw data
            # super_classes.update_supcls_KLdiv_raw(rawdata, iter=iter_supcls) # raw data
            # # super_classes.update_supcls_KLdiv_exp(iter=iter_supcls) # Gaussian




            ## - Save parameters for following incremental sessions
            if (iter_id + 1) % 5 == 0:
                checkpoint_iter_name = 'HkPPCAs_{}_run{}_first{}cls_{}_PCAq{}_{}shot_q{}lda{:.0e}_iter{}_flat.pth'.format(
                    dataset_name, run_id, cls_range[1], self_learner_name, q_ini, num_labeled_per_class, q, lda,
                    iter_id + 1)
                torch.save({'mu_online': mu_online,
                            'L_online': L_online,
                            # 'Sx_online': S_online,
                            'D2_online': D2_online,
                            'lda': lda,
                            't': t,
                            'q': q,
                            'N': N,
                            'nc': nc,
                            'num_cls': num_cls,
                            'trainset': trainset,
                            'val_values': val_values,
                            'val_labels': val_labels,
                            # 'sup_cls': super_classes,
                            'num_supcls': num_supcls,
                            # 'num_candid_supcls': num_candid_supcls,
                            'score_criteria': score_criteria
                            },
                           checkpoint_folder_address + checkpoint_iter_name)

            # del D2_online
            del v_online, vT, S2_online
            del cov_online, cov_online_cls
            # del rawdata, x_raw, y_raw
            torch.cuda.empty_cache()

            # t2 = time.time()

            try:
                ## - Evaluation: Accuracy for val
                print('Evaluating test set from {}th class to {}th class'.format(min(val_labels), max(val_labels)))
                # y_pred_val, loss = modules.prediction_flat_D2_HkPPCAs(val_values, val_labels, num_cls,
                #                                                       mu_online, L_online, D2_online,
                #                                                       lda, t, device,
                #                                                       num_batches=100, max_cls_batch_size=1300)
                y_pred_val, loss = modules.prediction_flat_D2_HkPPCAs(val_values, val_labels, num_cls,
                                                                      mu_online, L_online, D2_online,
                                                                      lda, t, device,
                                                                      num_batches=100, max_cls_batch_size=1300, dist='Eu') # Tag: k-means
                # y_pred_val, loss = modules.prediction_D2_HkPPCAs(val_values, val_labels, num_cls,
                #                                                  mu_online, L_online, D2_online,
                #                                                  super_classes, num_candid_supcls,
                #                                                  lda, t, device,
                #                                                  num_batches=100, max_cls_batch_size=1300)
                y_true_val = torch.tensor(val_labels, dtype=y_pred_val.dtype).to(device)
                cls_err_val = torch.sum(y_pred_val != y_true_val) / (len(val_labels))
                acc_val = 1 - cls_err_val.item()
                print('#{} Accuracy rate - val: {:.4f}'.format(iter_id, acc_val))
                print('#{} loss - on: {}'.format(iter_id, loss.item()))


                # Task//
                ## Calculate KMeans loss
                ## Loss calculation is replaced by prediction_online_s function
                # loss = Kmeans_loss(val_values, y_pred_val_on, mu_online)

                # if (iter_id + 1) == num_iters:
                #     trend_name = 'HkPPCAs_{}_run{}_first{}cls_acc_val_trend.pth'.format(
                #         dataset_name, run_id, cls_range[1])
                #     torch.save(
                #         {'acc_val': acc_val},
                #         checkpoint_folder_address + trend_name)

                del y_pred_val, y_true_val
                torch.cuda.empty_cache()
                print('End of one iteration')

            except:
                print(123)

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

            # if (iter_id + 1) == num_iters:
            #     ### Task: train classification
            #     # y_pred, loss = prediction_online_s_train(train_loader, nc, num_cls, mu_online,  # pi_online,
            #     #                                          L_online, S_online, lda, t,
            #     #                                          device)
            #     # y_pred, loss = modules.prediction_online_D2_train(train_loader, num_cls, mu_online,  # pi_online,
            #     #                                           L_online, D2_online, lda, t,
            #     #                                           device)
            #     y_pred, loss = modules.prediction_D2_train_HkPPCAs(train_loader, num_cls, mu_online,  # pi_online,
            #                                                        L_online, D2_online,
            #                                                        super_classes, num_candid_supcls,
            #                                                        lda, t, device)
            #     y_true = torch.tensor(trainset.labels, dtype=y_pred.dtype).to(device)
            #     cls_err = torch.sum(y_pred != y_true) / (len(trainset.labels))  # (len(val_labels))
            #     print('Accuracy rate - train: {:.4f}'.format(1 - cls_err.item()))
            #     print('Loss - train:', loss.item())
            #
            #     # plt.show()
            print('Done')
            #
            #     # ## Accuracy for each class in Validation set
            #     # N_val=val_labels.__len__()
            #     # N_iter=int(N_val/50)
            #     # val_acc_cls=[]
            #     # for i in range(N_iter):
            #     #     id_cls_start=i*50
            #     #     id_cls_end=(i+1)*50
            #     #     y_pred_i=prediction_online(val_values[id_cls_start:id_cls_end],val_labels[id_cls_start:id_cls_end],
            #     #                          N, num_cls, mu_online, pi_online, L_online, S_online, lda, t,
            #     #                          device)
            #     #     cls_err_i=torch.sum(y_pred_i!=i)/50
            #     #     print('{}th class accuracy - val: {}'.format(i,1-cls_err_i.item()))
            #     #     val_acc_cls.append(1-cls_err_i.item())
            #     # plt.hist(val_acc_cls, bins=[i / 10 for i in range(0, 11, 1)], )


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

    # runs_seeds_list=[[5,88,387,113,471,135],
    #                  [491,122,232,18,470,210],
    #                  [224,77,212,95,10,251],
    #                  [383,191,372,55,192,233],
    #                  [293,328,306,37,137,452]]
    # sessions_list=[[0,500],[500,600],[600,700],[700,800],[800,900],[900,1000]]

    runs_seeds_list = [[5],
                       ]
    # sessions_list = [[0, 1000]]
    sessions_list = [[0, 10450]]
    num_runs = len(runs_seeds_list)
    num_iters = 5

    # feature_address = [os.getcwd() + '\\..\\Benchmarks\\ImageNet_CLIP\\']
    feature_address = [os.getcwd() + '\\..\\..\\Benchmarks\\ImageNet10k_CLIP\\']
    checkpoint_folder_addr = os.getcwd() + '\\..\\..\\Checkpoints\\'
    dataset_name = 'ImageNet10k_288x288'
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
