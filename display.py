# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 21:22:56 2023

@author: Kert PC
"""
from torch.utils.data import Dataset, DataLoader
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.std import tqdm
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
import matplotlib
from models import UNet, UNetDouble
from dataset_grid import SatDataset
from loss_log_likelihood import LogLikelihoodLoss

TRAIN_DATA_PATH = './data/npy/train/'
VAL_DATA_PATH = './data/npy/val/'
MODEL_PATH = './models/'
MASK_PATH = './data/mask/mask.npy'
T_WINDOW = 27

BATCH_SIZE = 1

def display_preds_2(net, testset) :
    net.eval()
    mask = np.load(MASK_PATH)
    mask_disp = np.rot90(mask)
    
    fig=plt.figure(figsize=(28, 12))
    columns = 1
    rows = 3
    
    loss_log_likelihood = LogLikelihoodLoss()
    
    for i in range(rows):
      batch = testset[(i+1)*80]
      satData = batch['satData']
      satData_tensor = torch.from_numpy(satData).cuda().float()
      satData_tensor = satData_tensor.unsqueeze(0)
      
      date = batch['date'].split('.')[0]
      
      preds = net(satData_tensor)
      pred1 = preds[:, 0, :, :]
      pred2 = preds[:, 1, :, :]
      
      var, mean = loss_log_likelihood.get_var_and_mean(pred1, pred2)
      
      pred1 = pred1.cpu().detach().numpy() * mask
      pred2 = pred2.cpu().detach().numpy() * mask
      var = var.cpu().detach().numpy() * mask
      mean = mean.cpu().detach().numpy() * mask
      
      pred1 = np.rot90(pred1[0, :, :])
      pred2 = np.rot90(pred2[0, :, :])
      var = np.rot90(var[0, :, :])
      mean = np.rot90(mean[0, :, :])
    
      """
      fig.add_subplot(rows, columns, i*columns+1)
      plt.title(f'{date} : variance')
      plt.imshow(var, cmap='plasma')
      plt.imshow(mask_disp, cmap='gray', alpha=1.0*(mask_disp == 0), interpolation='none')
      """
      fig.add_subplot(rows, columns, i*columns+1)
      plt.title(f'{date} : reconstruction')
      ssh = plt.imshow(mean, cmap='plasma', vmin=-0.4, vmax=0.4)
      plt.imshow(mask_disp, cmap='gray', alpha=1.0*(mask_disp == 0), interpolation='none')
      plt.colorbar(ssh)

    plt.show()


def display_preds_1(net, testset) :
    net.eval()
    mask = np.load(MASK_PATH)
    mask_disp = np.rot90(mask)
    
    fig=plt.figure(figsize=(28, 12))
    columns = 1
    rows = 3
    
    loss_log_likelihood = LogLikelihoodLoss()
    
    for i in range(rows):
      batch = testset[(i+1)*80]
      satData = batch['satData']
      satData_tensor = torch.from_numpy(satData).cuda().float()
      satData_tensor = satData_tensor.unsqueeze(0)
      
      date = batch['date'].split('.')[0]
      
      preds = net(satData_tensor)
      pred = preds[:, 0, :, :]
      
      pred = pred.cpu().detach().numpy() * mask
      pred = np.rot90(pred[0, :, :])
      
      
      
      fig.add_subplot(rows, columns, i*columns+1)
      plt.title(f'{date} : reconstruction')
      ssh = plt.imshow(pred, cmap='plasma', vmin=-0.4, vmax=0.4, interpolation='none')
      plt.imshow(mask_disp, cmap='gray', alpha=1.0*(mask_disp == 0), interpolation='none')
      plt.colorbar(ssh)
      

    plt.show()
    
    
    
def display_preds_double_2(net, testset) :
    net.eval()
    mask = np.load(MASK_PATH)
    mask_disp = np.rot90(mask)
    
    fig=plt.figure(figsize=(28, 12))
    columns = 2
    rows = 3
    
    loss_log_likelihood = LogLikelihoodLoss()
    
    for i in range(rows):
      batch = testset[(i+1)*80]
      satData = batch['satData']
      satData_tensor = torch.from_numpy(satData).cuda().float()
      satData_tensor = satData_tensor.unsqueeze(0)
      
      date = batch['date'].split('.')[0]
      
      preds_end, preds_mid = net(satData_tensor)
      pred1_mid = preds_mid[:, 0, :, :]
      pred2_mid = preds_mid[:, 1, :, :]
      
      var_mid, mean_mid = loss_log_likelihood.get_var_and_mean(pred1_mid, pred2_mid)
      
      pred1_end = preds_end[:, 0, :, :]
      pred2_end = preds_end[:, 1, :, :]
      
      var_end, mean_end = loss_log_likelihood.get_var_and_mean(pred1_end, pred2_end)
      
    
      var_mid = var_mid.cpu().detach().numpy() * mask
      mean_mid = mean_mid.cpu().detach().numpy() * mask
      var_end = var_end.cpu().detach().numpy() * mask
      mean_end = mean_end.cpu().detach().numpy() * mask
      
      var_mid = np.rot90(var_mid[0, :, :])
      mean_mid = np.rot90(mean_mid[0, :, :])
      var_end = np.rot90(var_end[0, :, :])
      mean_end = np.rot90(mean_end[0, :, :])
      
      """
      fig.add_subplot(rows, columns, i*columns+1)
      plt.title(f'{date} : mid variance')
      plt.imshow(var_mid, cmap='plasma')
      plt.imshow(mask_disp, cmap='gray', alpha=1.0*(mask_disp == 0), interpolation='none')
      """
      fig.add_subplot(rows, columns, i*columns+1)
      plt.title(f'{date} : mid reconstruction')
      ssh = plt.imshow(mean_mid, cmap='plasma', vmin=-0.4, vmax=0.4)
      plt.imshow(mask_disp, cmap='gray', alpha=1.0*(mask_disp == 0), interpolation='none')
      plt.colorbar(ssh)
      
      """
      fig.add_subplot(rows, columns, i*columns+3)
      plt.title(f'{date} : end variance')
      plt.imshow(var_end, cmap='plasma')
      plt.imshow(mask_disp, cmap='gray', alpha=1.0*(mask_disp == 0), interpolation='none')
      """
      fig.add_subplot(rows, columns, i*columns+2)
      plt.title(f'{date} : end reconstruction')
      ssh = plt.imshow(mean_end, cmap='plasma', vmin=-0.4, vmax=0.4)
      plt.imshow(mask_disp, cmap='gray', alpha=1.0*(mask_disp == 0), interpolation='none')
      plt.colorbar(ssh)

    plt.show()


def display_preds_double_1(net, testset) :
    net.eval()
    mask = np.load(MASK_PATH)
    mask_disp = np.rot90(mask)
    
    fig=plt.figure(figsize=(28, 12))
    columns = 2
    rows = 3
    
    loss_log_likelihood = LogLikelihoodLoss()
    
    for i in range(rows):
          batch = testset[(i+1)*80]
          satData = batch['satData']
          satData_tensor = torch.from_numpy(satData).cuda().float()
          satData_tensor = satData_tensor.unsqueeze(0)
          
          date = batch['date'].split('.')[0]
          
          preds_mid, preds_end = net(satData_tensor)
          pred_mid = preds_mid[:, 0, :, :]
          pred_end = preds_end[:, 0, :, :]
          
          pred_mid = pred_mid.cpu().detach().numpy() * mask
          pred_end = pred_end.cpu().detach().numpy() * mask
          
          pred_mid = np.rot90(pred_mid[0, :, :])
          pred_end = np.rot90(pred_end[0, :, :])
          
          fig.add_subplot(rows, columns, i*columns+1)
          plt.title(f'{date} : mid reconstruction')
          ssh = plt.imshow(pred_mid, cmap='plasma', vmin=-0.4, vmax=0.4)
          plt.imshow(mask_disp, cmap='gray', alpha=1.0*(mask_disp == 0), interpolation='none')
          plt.colorbar(ssh)
          
          fig.add_subplot(rows, columns, i*columns+2)
          plt.title(f'{date} : end reconstruction')
          ssh = plt.imshow(pred_end, cmap='plasma', vmin=-0.4, vmax=0.4)
          plt.imshow(mask_disp, cmap='gray', alpha=1.0*(mask_disp == 0), interpolation='none')
          plt.colorbar(ssh)

    plt.show()

if __name__ == "__main__" :
    valset = SatDataset(VAL_DATA_PATH, T_WINDOW)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=1)
    
    in_channels = T_WINDOW + 4
    train_double_net = True
    out_channels = 2
    
    if train_double_net :
        net = UNetDouble(in_channels=in_channels, out_channels=out_channels, depths=[32, 64, 128, 256]).cuda().float()
        net.load_state_dict(torch.load(MODEL_PATH + f'model_ch{out_channels}_double_final'))
        net.cuda().float()
        
        if out_channels == 1 :
            display_preds_double_1(net, valset)
        else :
            display_preds_double_2(net, valset)
    else :
        net = UNet(in_channels=in_channels, out_channels=out_channels, depths=[32, 64, 128, 256]).cuda().float()
        net.load_state_dict(torch.load(MODEL_PATH + f'model_ch{out_channels}_final'))
        net.cuda().float()
    
        if out_channels == 1 :
            display_preds_1(net, valset)
        else :
            display_preds_2(net, valset)