# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:46:58 2023

@author: Kert PC
"""
import os
import torch
import torch.utils.data as data
from torch import nn
import random
import pandas as pd
import numpy as np
import math
from datetime import date


class LogLikelihoodLoss(nn.Module) :
    
    def __init__(self, gamma=10.0, mu=1e-3) :
        super(LogLikelihoodLoss, self).__init__()
        
        self.gamma = torch.tensor([gamma]).cuda().float()
        self.mu = torch.tensor([mu]).cuda().float()
        
    def get_var_and_mean(self, pred1, pred2) :
        min_pred = torch.minimum(pred1, self.gamma)
        exp_pred = torch.exp(min_pred)
        
        errVar_pred = 1.0 / torch.maximum(exp_pred, self.mu)
        mean_pred = torch.mul(errVar_pred, pred2)
        
        return errVar_pred, mean_pred
        
    def forward(self, gt, pred, mask) :
        pred1 = pred[:, 0, :, :]
        pred2 = pred[:, 1, :, :]
        
        errVar_pred, mean_pred = self.get_var_and_mean(pred1, pred2)
        mean_diff = torch.pow(gt - mean_pred, 2)
    
        var_pred_log = torch.masked_select(torch.log(errVar_pred), mask)
        diff_scaled = torch.masked_select(torch.div(mean_diff, errVar_pred), mask)
        
        loss = (torch.sum(var_pred_log) + torch.sum(diff_scaled)) / diff_scaled.size(dim=0)
        
        """
        error = masked_gt - masked_pred2
        var = torch.var(error)
        std = torch.std(error)
        
        error = torch.div(error, std)
        error = torch.mul(error, error)
        
        loss = (0.5 / error.size(dim=0)) * torch.sum(error) + 0.5 * torch.log(var)
        """
        
        return loss
        
        
            
            