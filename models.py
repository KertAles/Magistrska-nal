# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:21:44 2023

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

from loss_log_likelihood import LogLikelihoodLoss

# U-Net building block
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = nn.ReLU()(x)

        x_pool = self.maxpool(x)
        
        return x, x_pool

# U-Net building block
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = nn.ReLU()(x)

        x_up = self.upsample(x)
        
        return x, x_up
        
# U-Net building block
class OutBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(OutBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_channels, out_channels, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(mid_channels)

    def forward(self, x):

        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = nn.ReLU()(x)
        x = self.conv_out(x)
        
        return x

# U-Net used for either segmentation or colorisation     
class UNet(nn.Module):
    def __init__(self, in_channels=13, out_channels=2, depths=[16, 32, 64, 128]):
        super(UNet, self).__init__()
        # define the neural network blocks for UNet
        # The implementation details of the network are listed in the paper: https://arxiv.org/pdf/1505.04597.pdf
        
        self.downblocks = nn.ModuleList([DownBlock(in_channels, depths[0])])
        
        for i, depth in enumerate(depths[:-2]) :
            self.downblocks.append(DownBlock(depth, depths[i+1]))
        
        self.bottom_block = UpBlock(depths[-2], depths[-1])
        
        self.upblocks = nn.ModuleList([])
        
        for i in range(len(depths)-1, 1, -1) :
            self.upblocks.append(UpBlock(depths[i], depths[i-1]))
    
        self.outblock = OutBlock(depths[1], depths[0], out_channels)

    def forward(self, x):
      xs = []  
      
      for downblock in self.downblocks :
          x_raw, x = downblock(x)
          xs.append(x_raw)

      x_b, x = self.bottom_block(x)
      
      for i, upblock in enumerate(self.upblocks) :
          x = torch.cat([x, xs[len(xs) - i - 1]], dim=1)
          x_b, x = upblock(x)
      
      x = torch.cat([x, xs[0]], dim=1)
      out = self.outblock(x)
      #out = nn.Tanh()(out)
              
      return out
  
    
# U-Net used for either segmentation or colorisation     
class UNetDouble(nn.Module):
    def __init__(self, in_channels=13, out_channels=2, depths=[16, 32, 64, 128], t_window=27):
        super(UNetDouble, self).__init__()
        # define the neural network blocks for UNet
        # The implementation details of the network are listed in the paper: https://arxiv.org/pdf/1505.04597.pdf
        
        self.unet1 = UNet(in_channels=in_channels, out_channels=out_channels, depths=depths)
        self.unet2 = UNet(in_channels=in_channels, out_channels=out_channels, depths=depths)
        
        self.t_window = t_window
        self.out_channels = out_channels
        if self.out_channels == 2 :
            self.log_likelihood_loss = LogLikelihoodLoss()

    def forward(self, x):
      
        x1 = self.unet1(x)
        
        x2 = torch.clone(x)
        if self.out_channels == 1 :
            x2[:, (4 + self.t_window//2), :, :] = x1[:, 0, :, :]
        else :
            pred1 = x1[:, 0, :, :]
            pred2 = x1[:, 1, :, :]
            
            var, mean = self.log_likelihood_loss.get_var_and_mean(pred1, pred2)
            
            x2[:, (4 + self.t_window//2), :, :] = mean[:, :, :]
            
        x3 = self.unet2(x2)
        
        return x3, x1

        
        
      