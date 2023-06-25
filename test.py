# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:52:29 2023

@author: Kert PC
"""

from torch.utils.data import DataLoader
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.std import tqdm
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary

from models import UNet, UNetDouble
from dataset_grid import SatDataset
from loss_log_likelihood import LogLikelihoodLoss

TRAIN_DATA_PATH = './data/npy/train/'
VAL_DATA_PATH = './data/npy/val/'
TEST_DATA_PATH = './data/npy/test/'
MODEL_PATH = './models/'
T_WINDOW = 27

EPOCHS = 20

BATCH_SIZE = 1

def test(net, testloader, out_channels) :
    mse_loss = torch.nn.MSELoss(reduction='sum')
    log_likelihood_loss = LogLikelihoodLoss()
    
    running_loss = 0.0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs;
            sat_data = data['satData']
            gt = data['gt']
            mask = data['mask']
            
            # Get the inputs to the GPU
            sat_data = torch.from_numpy(sat_data).cuda().float()
            sat_data = sat_data.unsqueeze(0)
            gt = torch.from_numpy(gt).cuda().float()
            mask = mask.cuda().bool()
        
            outputs = net(sat_data)
                          
            if out_channels == 1 :
                outputs_masked = outputs[:, 0, :, :] * mask
                loss = mse_loss(outputs_masked, gt) / torch.sum(mask)
            else :
                pred1 = outputs[:, 0, :, :]
                pred2 = outputs[:, 1, :, :]
                              
                var, mean = log_likelihood_loss.get_var_and_mean(pred1, pred2)
                              
                mean_masked = mean * mask
                loss = mse_loss(mean_masked, gt) / torch.sum(mask)
                        
                
            RMS = torch.sqrt(loss)
                          
            running_loss += RMS.data.cpu().detach().numpy()
                          
        running_loss /= (i+1)
        print(f'Test RMS for single UNet, with ch {out_channels} : {running_loss}')
    return running_loss


def test_double(net, testloader, out_channels) :

    log_likelihood_loss = LogLikelihoodLoss()
    mse_loss = torch.nn.MSELoss(reduction='sum')
    
    running_loss = 0.0
    with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                  # get the inputs;
                  sat_data = data['satData']
                  gt = data['gt']
                  mask = data['mask']
    
                  # Get the inputs to the GPU
                  sat_data = torch.from_numpy(sat_data).cuda().float()
                  sat_data = sat_data.unsqueeze(0)
                  gt = torch.from_numpy(gt).cuda().float()
                  mask = mask.cuda().bool()

                  output_end, output_mid = net(sat_data)
                  
                  if out_channels == 1 :
                      output_end_masked = output_end[:, 0, :, :] * mask
                      loss_end = mse_loss(output_end_masked, gt)
                  else :
                      pred1_end = output_end[:, 0, :, :]
                      pred2_end = output_end[:, 1, :, :]
                      
                      var, mean_end = log_likelihood_loss.get_var_and_mean(pred1_end, pred2_end)
                      mean_end_masked = mean_end * mask
                      loss_end = mse_loss(mean_end_masked, gt)
                      
                  RMS = torch.sqrt(loss_end / torch.sum(mask))
                  
                  running_loss += RMS.data.cpu().detach().numpy()
                  
            running_loss /= (i+1)
            print(f'Test RMS for double UNet, with ch {out_channels} : {running_loss}')

    return running_loss



if __name__ == "__main__":
    
    testset = SatDataset(TEST_DATA_PATH, T_WINDOW)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=1)
    
    
    in_channels = T_WINDOW + 4
    out_channels_list = [1, 2]
    
    for out_channels in out_channels_list :
        net = UNetDouble(in_channels=in_channels, out_channels=out_channels, depths=[32, 64, 128, 256]).cuda().float()
        net.load_state_dict(torch.load(MODEL_PATH + f'model_ch{out_channels}_double_final_fine'))
        net.cuda().float()
            
        test_double(net, testset, out_channels)

            
        #net = UNet(in_channels=in_channels, out_channels=out_channels, depths=[32, 64, 128, 256]).cuda().float()
        #net.load_state_dict(torch.load(MODEL_PATH + f'model_ch{out_channels}_40'))
        #net.cuda().float()

        #test(net, testset, out_channels)
    