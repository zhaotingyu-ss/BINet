import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train_util import *


def compute_semantic_pos_loss(prob_in, labxy_feat,  pos_weight=0.003,kernel_size=16 ):

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape

    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_map = reconstr_feat[:, -2:, :, :] - labxy_feat[:, -2:, :, :]

    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-10)
    loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S



    loss_sum = 0.005 * (loss_sem + loss_pos)

    loss_sem_sum = 0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos

    return loss_sum, loss_sem_sum, loss_pos_sum,loss_sem_sum
