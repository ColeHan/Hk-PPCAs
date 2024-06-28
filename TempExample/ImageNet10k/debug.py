import os
import time
import torch
import numpy as np
import Modules.Modules as modules


# ## load ckpt
# checkpoint_folder_addr = os.getcwd() + '\\Checkpoints\\'
# addr=checkpoint_folder_addr+'HkPPCAs_ImageNet10k_288x288_run0_first10450cls_CLIP_resnetx4_PCAq2_2shot_q2lda1e-02_iter1.pth'
#
# iter1_result = torch.load(addr)
#
# val_values= iter1_result['val_values']
# val_labels= iter1_result['val_labels']
# num_cls= iter1_result['num_cls']
# mu_online = iter1_result['mu_online']
# L_online = iter1_result['L_online']
# D2_online = iter1_result['D2_online']
# num_candid_supcls = iter1_result['num_candid_supcls']
# lda = iter1_result['lda']
# t = iter1_result['t']
# super_classes = iter1_result['sup_cls']
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# y_pred_val, loss = modules.prediction_D2_HkPPCAs(val_values, val_labels, num_cls,
#                                                  mu_online, L_online, D2_online,
#                                                  super_classes, num_candid_supcls,
#                                                  lda, t, device,
#                                                  num_batches=100, max_cls_batch_size=1300)
# print(123)

# faster slicing
testiter=1000
t1=0
t2=0
t3=0
t4=0
t5=0
l=20000
candid_cls_id=[i for i in range(l)]
for i in range(testiter):
    score_obs1=(torch.rand(l,dtype=torch.float32)*100+40).cuda()
    ta = time.time()
    # torch
    #1
    score_obs1_minid = torch.argmin(score_obs1)
    score_obs1_mincls = candid_cls_id[score_obs1_minid.item()]
    ta1=time.time()
    #2
    score_obs1_minid = torch.argmin(score_obs1,0)
    score_obs1_mincls = candid_cls_id[score_obs1_minid]
    ta2=time.time()

    # np
    #3
    score_obs1_minid = np.argmin(score_obs1.cpu().numpy())
    score_obs1_mincls = candid_cls_id[score_obs1_minid.item()]
    ta3=time.time()
    #4
    score_obs1_minid = np.argmin(score_obs1.cpu().numpy(),0)
    score_obs1_mincls = candid_cls_id[score_obs1_minid]
    ta4=time.time()
    #5
    score_obs1_minid = np.argmin(score_obs1.tolist())
    score_obs1_mincls = candid_cls_id[score_obs1_minid]
    ta5=time.time()

    t1+=ta1-ta
    t2+=ta2-ta1
    t3+=ta3-ta2
    t4+=ta4-ta3
    t5+=ta5-ta4
print(t1/testiter,t2/testiter,t3/testiter,t4/testiter,t5/testiter)
