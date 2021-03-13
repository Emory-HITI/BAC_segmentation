import os
import numpy as np
import torch
from torch.autograd import Variable

def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t

def dice_loss(y_true, y_pred, epsilon = 1e-6):
    """
    y_true: b x C x X x Y
    y_pred: b x C x X x Y
    """   
    y1_true = y_true[:,0,:,:]
    y1_pred = y_pred[:,0,:,:]
    
    numerator1 = 2.0 * torch.sum((flatten(y1_true)* flatten(y1_pred)))
    denominator1 =  torch.sum((flatten(y1_true).pow(2)+flatten(y1_pred).pow(2)))+epsilon
    
    return 1-torch.mean(numerator1 / denominator1)


def TP(mask, pre_mask):
    return torch.sum((mask & pre_mask),dim=1)
    
def FP(mask, pre_mask):
    return torch.sum((pre_mask - (mask & pre_mask)),dim=1)
    
def TN(mask, pre_mask):
    return torch.sum(((1-mask) & (1-pre_mask)),dim=1)
    
def FN(mask, pre_mask):
    return torch.sum((mask - (mask & pre_mask)),dim=1)


def evaluate(mask, pre_mask):

    mask = mask.contiguous().view(mask.shape[0], -1)
    pre_mask = pre_mask.contiguous().view(pre_mask.shape[0], -1)
    
    tp = TP(mask, pre_mask)
    fp = FP(mask, pre_mask)
    tn = TN(mask, pre_mask)
    fn = FN(mask, pre_mask)

    smooth = 1e-6
    recall = tp/(tp+fn+smooth)
    
    precision = tp/(tp+fp+smooth)
    accuracy = (tp+tn)/(tp+tn+fp+fn+smooth)
    
    f_score = 2*recall*precision/(recall+precision+smooth)
    jaccard = tp/(tp+fn+fp+smooth)
    dice = 2*tp/(2*tp+fn+fp+smooth) 

    return torch.mean(recall), torch.mean(precision), torch.mean(accuracy), torch.mean(f_score), torch.mean(jaccard), torch.mean(dice)
